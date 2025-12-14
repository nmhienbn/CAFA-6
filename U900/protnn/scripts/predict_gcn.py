import argparse
import os
import sys

import joblib
import pandas as pd
import yaml

# Thêm path để import modules
sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str, required=True)
parser.add_argument('-d', '--device', type=str, default='0')
# THÊM: Tham số để chỉ định chạy TTA index nào
parser.add_argument('--run-index', type=int, default=-1, help='Index của vòng lặp TTA cần chạy (0, 1, 2...). -1 là chạy hết.')
parser.add_argument('--batch-size', type=int, default=2048, help='Batch size lớn cho A100')
parser.add_argument('--num-workers', type=int, default=32, help='Số CPU workers load data')

if __name__ == '__main__':

    args = parser.parse_args()
    
    # Thiết lập GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    import torch

    try:
        from protnn.utils import get_labels, get_goa_data, CAFAEvaluator, make_submission
        from protnn.dataset import StackDataset, StackDataLoader
        from protnn.stacker import GCNStacker
        from protnn.swa import SWA
        from protnn.train import train

        from protlib.metric import obo_parser, Graph, ia_parser
        from protlib.metric import get_topk_targets

    except ImportError:
        print('Alarm: Import Error in protnn/protlib')
        pass

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    # Load Ontology
    ontologies = []
    # Lưu ý: Cần đảm bảo path này đúng trong environment của bạn
    obo_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    for ns, terms_dict in obo_parser(obo_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    # Load Test Index
    test_idx = pd.read_feather(
        os.path.join(config['base_path'], 'helpers/fasta/test_seq.feather'),
        columns=['EntryID']
    )['EntryID'].values

    # --- LOOP CHÍNH ---
    for nout, ontology in enumerate(['bp', 'mf', 'cc']):
        mode = 'w' if nout == 0 else 'a'
        nn_cfg = config['gcn'][ontology]
        work_dir = os.path.join(config['base_path'], 'models/gcn', ontology)
        models_path = os.path.join(config['base_path'], config['models_path'])

        G = ontologies[nout]

        # Load GCN Stacker Model
        model = GCNStacker(
            5, 1,
            G.idxs,
            hidden_size=nn_cfg['hidden_size'],
            n_layers=nn_cfg['n_layers'],
            embed_size=nn_cfg['embed_size']
        )
        model.load_state_dict(torch.load(os.path.join(work_dir, 'checkpoint.pth')))
        model = model.cuda()

        # --- VÒNG LẶP TTA (CẦN CHIA NHỎ) ---
        # nn_cfg['tta'] thường là dict, enumerate sẽ trả về (index, key)
        for k, tta_cfg in enumerate(nn_cfg['tta']):
            
            # [LOGIC MỚI] Kiểm tra xem worker này có nhiệm vụ chạy index k không
            if args.run_index != -1 and k != args.run_index:
                continue

            print(f"Running Inference: Ontology={ontology}, TTA_Index={k}, Config={tta_cfg}")

            output_path = os.path.join(config['base_path'], 'models/gcn', f'pred_tta_{k}.tsv')
            
            # Nếu chạy song song, file mode 'a' (append) có thể gây lỗi race condition nếu nhiều process cùng ghi vào 1 file.
            # Tuy nhiên code này tách file theo `pred_tta_{k}.tsv`, nên mỗi worker ghi 1 file riêng -> An toàn.
            # Nhưng lưu ý: mode='w' if nout == 0 else 'a' áp dụng cho cùng 1 file output_path.
            # Vì ta giữ nguyên logic chạy tuần tự các ontology (bp->mf->cc) trong 1 worker, logic này vẫn đúng.
            
            model_ids = nn_cfg['tta'][tta_cfg]
            models_config = []

            for mod in model_ids:
                models_config.append([
                    os.path.join(models_path, mod),
                    [
                        config['base_models'][mod]['bp'],
                        config['base_models'][mod]['mf'],
                        config['base_models'][mod]['cc']
                    ],
                    config['base_models'][mod]['conditional']
                ])
                
            # print(models_config) # Có thể comment lại cho đỡ rác log

            # get features
            test_preds = []

            prior_cnd = joblib.load(
                os.path.join(config['base_path'], f'helpers/real_targets/{G.namespace}/prior.pkl')
            )
            nulls = joblib.load(
                os.path.join(config['base_path'], f'helpers/real_targets/{G.namespace}/nulls.pkl')
            )
            prior_raw = prior_cnd * (1 - nulls)

            for folder, split, cnd in models_config:
                path = os.path.join(folder, 'test_pred.pkl')
                # Load file pkl có thể nặng RAM, chú ý nếu chạy 8 GPU cùng lúc
                test_pred = joblib.load(path)[:, sum(split[:nout]): sum(split[:nout]) + split[nout]]
                idx = get_topk_targets(G, split[nout], train_path=os.path.join(config['base_path'], 'Train/'))
                test_preds.append((test_pred, idx, cnd))
            
            # add side models prediction
            for side_pred_name in nn_cfg['side_preds']: # đổi tên biến tránh trùng
                side_path = os.path.join(
                    models_path,
                    side_pred_name,
                    config['public_models'][side_pred_name]['source'],
                )
                
                side_data = joblib.load(side_path)
                split = side_data['borders']

                test_pred = side_data['test_pred'][:, sum(split[:nout]): sum(split[:nout]) + split[nout]]
                idx = side_data['idx'][sum(split[:nout]): sum(split[:nout]) + split[nout]]
                test_preds.append((test_pred, idx, False))

            test_goa_data = get_goa_data(os.path.join(config['base_path'], 'temporal'), 'test', test_idx, G)
            
            test_ds = StackDataset(
                test_preds,
                G.idxs,
                prior_raw,
                prior_cnd,
                G,
                goa_list=test_goa_data,
                p_goa=1,
                targets=None
            )
            
            # Giảm num_workers của DataLoader nếu chạy quá nhiều process GPU để tránh quá tải CPU
            test_dl = StackDataLoader(
                test_ds, 
                batch_size=args.batch_size,
                shuffle=False, 
                num_workers=args.num_workers
            )

            make_submission(
                model,
                test_dl,
                G,
                test_idx,
                output_path,
                mode=mode,
                topk=500,
                tau=0.01
            )