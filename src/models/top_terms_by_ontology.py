import os
import numpy as np
import pandas as pd

TRAIN_TERMS = 'data/raw/cafa6/Train/train_terms.tsv'
IA_FILE     = 'data/raw/cafa6/IA.tsv'
TOP_K_LABELS = 10000
SAVE_DIR = 'features/top_terms_by_aspect'
os.makedirs(SAVE_DIR, exist_ok=True)

df_terms = pd.read_csv(TRAIN_TERMS, sep='\t', header=0,
                       names=['EntryID', 'term', 'aspect'])
df_ia = pd.read_csv(IA_FILE, sep='\t', names=['term', 'ia'])
ia_dict = dict(zip(df_ia['term'], df_ia['ia']))

for aspect, aspect_name in [('F', 'MF'), ('P', 'BP'), ('C', 'CC')]:
    df_a = df_terms[df_terms['aspect'] == aspect]

    term_counts = df_a['term'].value_counts().reset_index()
    term_counts.columns = ['term', 'freq']

    term_counts['ia'] = term_counts['term'].map(ia_dict).fillna(0.0)
    term_counts['score'] = term_counts['freq'] * term_counts['ia']

    top_terms_df = term_counts.sort_values(
        by='score', ascending=False
    ).head(TOP_K_LABELS)

    top_terms = top_terms_df['term'].values
    out_path = os.path.join(SAVE_DIR, f'top_terms_{aspect_name}.npy')
    np.save(out_path, top_terms)
    print(aspect_name, len(top_terms), 'terms ->', out_path)
