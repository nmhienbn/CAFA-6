import argparse
import os
import sys
import joblib
import numpy as np
import yaml


sys.path.append(os.path.abspath(os.path.join(__file__, '../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str, required=True)
parser.add_argument('-m', '--model-name', type=str, required=True)
parser.add_argument('-d', '--device', type=str, default="0")
parser.add_argument('--fold', type=int, required=True,
                    help="Fold ID to run (0-4)")

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    try:
        from protlib.metric import obo_parser, Graph, get_topk_targets
        from protlib.models.prepocess import get_features_simple, get_targets_from_parquet
        from protlib.models.gbdt import BCEWithNaNLoss, BCEwithNaNMetric
    except ImportError:
        print('Alarm: Import Error in protlib')
        pass

    from py_boost import GradientBoosting
    from py_boost.multioutput.sketching import RandomProjectionSketch

    print(
        f"--- Worker started: Model {args.model_name} | Fold {args.fold} on GPU {args.device} ---")

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    model_config = config['base_models'][args.model_name]
    graph_path = os.path.join(config['base_path'], 'Train/go-basic.obo')
    embeds_path = os.path.join(config['base_path'], config['embeds_path'])
    helpers_path = os.path.join(config['base_path'], config['helpers_path'])

    output = os.path.join(config['base_path'],
                          config['models_path'], args.model_name)
    os.makedirs(output, exist_ok=True)

    ontologies = []
    for ns, terms_dict in obo_parser(graph_path).items():
        ontologies.append(Graph(ns, terms_dict, None, True))

    split = [model_config['bp'], model_config['mf'], model_config['cc']]
    cols = []
    for n, i in enumerate(split):
        cols.extend(get_topk_targets(
            ontologies[n], i,
            train_path=os.path.join(config['base_path'], 'Train')
        ))

    fillna = not model_config['conditional']
    Y = get_targets_from_parquet(
        os.path.join(helpers_path, 'real_targets'),
        ontologies, split, ids=cols, fillna=fillna
    )

    train_embeds = [os.path.join(embeds_path, x, 'train_embeds.npy')
                    for x in model_config['embeds']]
    test_embeds = [os.path.join(embeds_path, x, 'test_embeds.npy')
                   for x in model_config['embeds']]

    X, train_idx = get_features_simple(os.path.join(
        helpers_path, 'fasta/train_seq.feather'), train_embeds)
    X_test, test_idx = get_features_simple(os.path.join(
        helpers_path, 'fasta/test_seq.feather'), test_embeds)

    N_FOLDS = 5
    key = np.array(list(map(hash, X.sum(axis=1))))
    np.random.seed(42)
    folds = np.unique(key)
    np.random.shuffle(folds)
    folds = np.array_split(folds, N_FOLDS)

    f = args.fold
    print(f"Processing Fold {f}...")

    test_sl = np.isin(key, folds[f])
    tr_idx, ts_idx = np.nonzero(~test_sl)[0], np.nonzero(test_sl)[0]

    model = GradientBoosting(
        BCEWithNaNLoss(), BCEwithNaNMetric(),
        ntrees=20000, lr=0.03, verbose=100, es=300, lambda_l2=10, gd_steps=1,
        subsample=.8, colsample=0.8, min_data_in_leaf=10, use_hess=False,
        max_bin=256, max_depth=6,
        multioutput_sketch=RandomProjectionSketch(3)
    )

    model.fit(X[tr_idx], Y[tr_idx], eval_sets=[
              {'X': X[ts_idx], 'y': Y[ts_idx]}])

    joblib.dump(model, os.path.join(output, f'model_{f}.pkl'))

    fold_oof = np.zeros((X.shape[0], len(cols)), dtype=np.float32)
    fold_oof[ts_idx] = model.predict(X[ts_idx], batch_size=5000)
    joblib.dump(fold_oof, os.path.join(output, f'temp_oof_fold_{f}.pkl'))

    fold_test = model.predict(X_test, batch_size=5000)
    joblib.dump(fold_test, os.path.join(output, f'temp_test_fold_{f}.pkl'))

    print(f"--- Finished Fold {f} ---")
