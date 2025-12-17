import argparse
import os
import pandas as pd
import tqdm
import yaml


ASPECT_DEFAULT_MAP = {
    'F': 'enables',
    'P': 'involved_in',
    'C': 'located_in'
}


def get_final_qualifier(row):
    raw_q = str(row['type'])
    if raw_q != 'nan' and raw_q.strip() != '':
        return raw_q

    return ASPECT_DEFAULT_MAP.get(row['z'], 'other')


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str,
                    help='GAF file (e.g.: goa_uniprot_all.gaf.gz)')
parser.add_argument('-c', '--config-path', type=str,
                    help='config.yaml file path')
parser.add_argument('-o', '--output', type=str,
                    default='./output', help='Output directory')

if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    print("---Loading ID from Feather file ---")

    train_idx = set(
        pd.read_feather(
            os.path.join(config['base_path'],
                         'helpers/fasta/train_seq.feather'),
            columns=['EntryID']
        )['EntryID']
    )

    test_idx = set(
        pd.read_feather(
            os.path.join(config['base_path'],
                         'helpers/fasta/test_seq.feather'),
            columns=['EntryID']
        )['EntryID']
    )
    idxs = train_idx.union(test_idx)

    col_names = [
        'x', 'EntryID', 'xx', 'type', 'term', 'y', 'source', 'yyy',
        'z', 'zz', 'zzz', 'a', 'aa', 'date', 'b', 'bb', 'bbb'
    ]

    use_cols = ['EntryID', 'term', 'source', 'type', 'z']

    reader = pd.read_csv(
        os.path.join(config['base_path'], config['temporal_path'], args.file),
        sep='\t',
        header=None,
        comment='!',
        names=col_names,
        usecols=use_cols,
        chunksize=1_000_000,
        dtype=str
    )

    store = []

    print("--- Processing GAF ---")
    for n, batch in tqdm.tqdm(enumerate(reader)):

        if n == 0:
            batch = batch.dropna(subset=['EntryID', 'term'])

        filtered = batch[(batch['EntryID'].isin(idxs))].copy()

        if filtered.empty:
            continue

        filtered['qualifier'] = filtered.apply(get_final_qualifier, axis=1)

        filtered = filtered[['EntryID', 'term',
                             'qualifier', 'source']].drop_duplicates()

        if len(store) > 0 and len(filtered) > 0 and \
                (store[-1].iloc[-1].values == filtered.iloc[0].values).all():
            filtered = filtered.iloc[1:]

        store.append(filtered)

    store = pd.concat(store, ignore_index=True)

    output_dir = os.path.join(
        config['base_path'], config['temporal_path'], args.output)
    os.makedirs(output_dir, exist_ok=True)

    export_df = store.rename(
        columns={'EntryID': 'protein_id', 'term': 'go_term'})

    export_df[['protein_id', 'go_term', 'qualifier']].to_csv(
        os.path.join(output_dir, 'goa_uniprot_all.tsv'), index=False, sep='\t'
    )

    print("Processing complete!")
