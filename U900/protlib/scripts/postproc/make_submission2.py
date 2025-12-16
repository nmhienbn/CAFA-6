import argparse
import os
import sys
import pandas as pd
import yaml

sys.path.append(os.path.abspath(os.path.join(__file__, '../../../../')))

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config-path', type=str, required=True)
parser.add_argument('-d', '--device', type=str, default="0")
parser.add_argument('-i', '--input-file', type=str,
                    required=True, help="Path to your single submission file")

if __name__ == '__main__':
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import cudf

    try:
        from protlib.metric import CAFAMetric
        from protlib.cafa_utils import obo_parser, Graph
    except Exception:
        print("Warning: protlib not found.")
        CAFAMetric, obo_parser, Graph = [None] * 3

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    base_path = config['base_path']
    graph_path = os.path.join(base_path, 'Train/go-basic.obo')
    temporal_path = os.path.join(base_path, config.get(
        'temporal_path', 'temporal'), "ver228")

    sub_path = os.path.join(base_path, 'sub')
    os.makedirs(sub_path, exist_ok=True)

    print(f"Reading main submission: {args.input_file}")
    pred = cudf.read_csv(args.input_file, sep='\t', header=None, names=[
                         'EntryID', 'term', 'prob'])

    # goa leak
    print("Loading GOA leak...")
    goa_path = os.path.join(temporal_path, 'labels/prop_test_leak_no_dup.tsv')
    if os.path.exists(goa_path):
        goa = cudf.read_csv(goa_path, sep='\t', usecols=['EntryID', 'term'])
        goa['prob'] = 0.99
    else:
        print(f"Warning: GOA file not found at {goa_path}")
        goa = cudf.DataFrame(columns=['EntryID', 'term', 'prob'])

    # gq51 dataset
    print("Loading QuickGO51...")
    qg_path = os.path.join(temporal_path, 'prop_quickgo51.tsv')
    if os.path.exists(qg_path):
        qg = cudf.read_csv(qg_path, sep='\t', usecols=['EntryID', 'term'])
        qg['prob'] = 0.99
    else:
        print(f"Warning: QuickGO51 file not found at {qg_path}")
        qg = cudf.DataFrame(columns=['EntryID', 'term', 'prob'])

    # diff
    print("Loading Diff terms...")
    diff_path = os.path.join(temporal_path, 'cafa-terms-diff.tsv')
    if os.path.exists(diff_path):
        diff = cudf.read_csv(diff_path, header=None, sep='\t', names=[
                             'EntryID', 'term', 'prob'])
    else:
        print(f"Warning: Diff file not found at {diff_path}")
        diff = cudf.DataFrame(columns=['EntryID', 'term', 'prob'])

    print("Concatenating all sources...")
    pred = cudf.concat([
        pred,
        qg,
        goa,
        diff,
    ], ignore_index=True)

    mapper = []
    if os.path.exists(graph_path):
        print("Parsing OBO graph for namespaces...")
        for n, (ns, terms_dict) in enumerate(obo_parser(graph_path).items()):
            G = Graph(ns, terms_dict, None, True)
            terms = [x['id'] for x in G.terms_list]
            mapper.append(pd.Series([n] * len(terms), index=terms))

        mapper = cudf.from_pandas(pd.concat(mapper))
        pred['ns'] = pred['term'].map(mapper)
        missing_count = pred['ns'].isna().sum()
        if missing_count > 0:
            print(
                f"Warning: {missing_count} terms could not be mapped to a namespace (not in OBO). Dropping them.")
            pred = pred.dropna(subset=['ns'])
        pred['ns'] = pred['ns'].astype(int)
    else:
        print("Error: OBO file not found. Cannot perform ranking.")
        sys.exit(1)

    pred = pred.groupby(['EntryID', 'term']).mean().reset_index()

    pred['rank'] = pred.groupby(['EntryID', 'ns'])['prob'].rank(
        method='dense', ascending=False) - 1

    pred = pred.query('rank < 500')

    output_file = os.path.join(sub_path, 'submission.tsv')
    print(f"Saving final submission to: {output_file}")

    pred[['EntryID', 'term', 'prob']].to_csv(
        output_file, header=False, index=False, sep='\t'
    )
    print("Done.")
