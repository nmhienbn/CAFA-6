import argparse
import glob
import os
import sys
import joblib
import numpy as np
import pandas as pd
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch import nn
import tqdm


sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

try:
    from protnn.utils import get_labels, get_goa_data, CAFAEvaluator
    from protnn.dataset import StackDataset, StackDataLoader
    from protnn.stacker import GCNStacker
    from protnn.swa import SWA
    from protlib.metric import obo_parser, Graph, ia_parser, get_topk_targets
except ImportError:
    print('Import Error. Make sure PYTHONPATH is set correctly.')
    pass

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--ontology', type=str, required=True)
parser.add_argument('-c', '--config-path', type=str, required=True)

parser.add_argument('--batch-size', type=int, default=4096,
                    help="Batch size PER GPU")
parser.add_argument('--num-workers', type=int, default=16,
                    help="Number of CPU workers per GPU")
parser.add_argument('--log-to-file', action='store_true',
                    help="If true, tqdm will be disabled to avoid clutter in logs")

ont_dict = {'bp': 0, 'mf': 1, 'cc': 2}


def train_ddp(model, swa, train_dl, val_dl, evaluator, train_sampler, rank, n_ep=20, lr=1e-3, clip_grad=1, use_tqdm=True):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if rank == 0:
        print(f"🚀 Start training DDP for {n_ep} epochs...")

    for epoch in range(n_ep):

        train_sampler.set_epoch(epoch)
        model.train()

        iterator = train_dl
        if rank == 0 and use_tqdm:
            iterator = tqdm.tqdm(train_dl, desc=f"Epoch {epoch}")

        total_loss = 0
        steps = 0

        for batch in iterator:
            optimizer.zero_grad()

            output = model(batch)
            loss = loss_fn(output, batch['y'])

            loss.backward()

            if clip_grad is not None:
                nn.utils.clip_grad_value_(
                    model.parameters(), clip_value=clip_grad)

            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if rank == 0 and use_tqdm:
                iterator.set_postfix({'loss': total_loss/steps})

        dist.barrier()

        if rank == 0:
            avg_loss = total_loss / steps
            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.5f}")

            score = evaluator(model.module, val_dl)
            print(f'[Epoch {epoch}] Val CAFA5 score: {score:.5f}')
            swa.add_checkpoint(model.module, score=score)

        dist.barrier()

    return model


if __name__ == '__main__':
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", local_rank)

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    NOUT = ont_dict[args.ontology]

    models_path = os.path.join(config['base_path'], config['models_path'])
    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    ia_path = os.path.join(config['base_path'], 'IA.txt')
    helpers_path = os.path.join(config['base_path'], config['helpers_path'])
    temporal_path = os.path.join(
        config['base_path'], config['temporal_path'], "ver228")

    work_dir = os.path.join(models_path, 'gcn', args.ontology)
    temp_dir = os.path.join(work_dir, 'temp')
    swa_dir = os.path.join(work_dir, 'swa')

    if local_rank == 0:
        os.makedirs(temp_dir, exist_ok=True)
    dist.barrier()

    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, ia_parser(ia_path), True))
    G = ontologies[NOUT]

    root_id = [x['id'] for x in G.terms_list if len(x['adj']) == 0][0]
    flist = sorted(glob.glob(os.path.join(
        helpers_path, f'real_targets/{G.namespace}/part*')))

    target = pd.concat([pd.read_parquet(x, columns=[x['id'] for x in G.terms_list])
                       for x in flist], ignore_index=True).fillna(0)

    train_sl = np.nonzero(target[root_id].values == 1)[0]
    target = target.values[train_sl]

    prior_cnd = joblib.load(os.path.join(
        helpers_path, f'real_targets/{G.namespace}/prior.pkl'))
    nulls = joblib.load(os.path.join(
        helpers_path, f'real_targets/{G.namespace}/nulls.pkl'))
    prior_raw = prior_cnd * (1 - nulls)

    prot_id = pd.read_feather(os.path.join(
        helpers_path, 'fasta/train_seq.feather'), columns=['EntryID'])['EntryID'].values
    prot_id = prot_id[train_sl]
    goa_data = get_goa_data(temporal_path, 'train', prot_id, G)

    nn_cfg = config['gcn'][args.ontology]
    models_config = []
    for mod in nn_cfg['preds']:
        models_config.append([
            os.path.join(models_path, mod),
            [config['base_models'][mod]['bp'], config['base_models']
                [mod]['mf'], config['base_models'][mod]['cc']],
            config['base_models'][mod]['conditional']
        ])

    preds = []
    for folder, split, cnd in models_config:
        path = os.path.join(folder, 'oof_pred.pkl')
        oof_pred = joblib.load(path)[train_sl][:, sum(
            split[:NOUT]): sum(split[:NOUT]) + split[NOUT]]
        idx = get_topk_targets(G, split[NOUT], train_path=os.path.join(
            config['base_path'], 'Train/'))
        preds.append((oof_pred, idx, cnd))

    test_labels = pd.concat([
        pd.read_csv(os.path.join(temporal_path,
                    'labels/prop_test_leak_no_dup.tsv'), sep='\t'),

    ]).drop_duplicates().reset_index(drop=True)

    if local_rank == 0:
        test_labels.to_csv(os.path.join(
            temp_dir, 'labels.tsv'), index=False, sep='\t')

    ids_to_take = test_labels['EntryID'].drop_duplicates().values
    sl = pd.read_feather(os.path.join(helpers_path, 'fasta/test_seq.feather'),
                         columns=['EntryID']).reset_index().set_index('EntryID').loc[ids_to_take]
    ids_to_take = sl.index.values
    sl = sl.values[:, 0]
    test_goa_data = get_goa_data(temporal_path, 'test', ids_to_take, G)

    test_preds = []
    for n, (folder, split, cnd) in enumerate(models_config):
        path = os.path.join(folder, 'test_pred.pkl')
        test_pred = joblib.load(path)[sl][:, sum(
            split[:NOUT]): sum(split[:NOUT]) + split[NOUT]]
        idx = get_topk_targets(G, split[NOUT], train_path=os.path.join(
            config['base_path'], 'Train/'))
        test_preds.append((test_pred, idx, cnd))

    for side_pred in nn_cfg['side_preds']:
        side_path = os.path.join(
            models_path, side_pred, config['public_models'][side_pred]['source'])
        data_side = joblib.load(side_path)
        split = data_side['borders']
        oof_pred = data_side['pred'][train_sl][:, sum(
            split[:NOUT]): sum(split[:NOUT]) + split[NOUT]]
        test_pred_val = data_side['test_pred'][sl][:, sum(
            split[:NOUT]): sum(split[:NOUT]) + split[NOUT]]
        idx = data_side['idx'][sum(split[:NOUT]): sum(
            split[:NOUT]) + split[NOUT]]

        preds.append((oof_pred, idx, False))
        test_preds.append((test_pred_val, idx, False))

    train_ds = StackDataset(preds, G.idxs, prior_raw, prior_cnd,
                            G, goa_list=goa_data, p_goa=.5, targets=target)

    train_sampler = DistributedSampler(train_ds, shuffle=True)

    train_dl = StackDataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_ds = StackDataset(test_preds, G.idxs, prior_raw, prior_cnd,
                          G, goa_list=test_goa_data, p_goa=1, targets=None)
    val_dl = StackDataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = GCNStacker(
        5, 1, G.idxs,
        hidden_size=nn_cfg['hidden_size'],
        n_layers=nn_cfg['n_layers'],
        embed_size=nn_cfg['embed_size']
    ).to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    evaluator = None
    swa = None
    if local_rank == 0:
        evaluator = CAFAEvaluator(
            os.path.join(config['base_path'], config['rapids-env']),
            os.path.join(temp_dir, 'labels.tsv'),
            config['base_path'],
            os.path.join(config['base_path'], 'Train/go-basic.obo'),
            os.path.join(config['base_path'], 'IA.txt'),
            ids_to_take, G,
            batch_size=3000,
            device=local_rank,
            temp_dir=temp_dir
        )
        swa = SWA(nn_cfg['store_swa'], path=swa_dir, rewrite=True)

    train_ddp(
        model, swa, train_dl, val_dl, evaluator, train_sampler,
        rank=local_rank,
        n_ep=nn_cfg['n_ep'],
        lr=1e-3,
        clip_grad=1e-1,
        use_tqdm=not args.log_to_file
    )

    if local_rank == 0:
        print("Saving final model...")
        joblib.dump(swa, os.path.join(work_dir, f'swa.pkl'))

        model_module = model.module
        model_module = swa.set_weights(model_module, 3, weighted=False)
        torch.save(model_module.state_dict(),
                   os.path.join(work_dir, f'checkpoint.pth'))

        final_score = evaluator(model_module, val_dl)
        print(f'✅ FINAL CAFA5 SCORE ({args.ontology.upper()}): {final_score}')

    dist.destroy_process_group()
