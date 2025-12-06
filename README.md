### Repo Layout

```text
cafa6/
  configs/
    data.yaml
    protboost.yaml
    mlp_softf1.yaml
    seq_only_lr.yaml
    gcn_stack.yaml
    ensemble.yaml
  data/
    raw/
      cafa6/   # from Kaggle 
      external/  # GOA, STRING, Pfam,...
    processed/
      mapping/   # id2protein, id2go, splits
  features/
    ankh3/
    esm2/
    protT5/
    taxon/
    goa/
    ppi/
    structure/   # Foldseek, AlphaFold index,...
  src/
    data/
      dataset.py
      go_graph.py
      split.py
    features/
      extract_single_model.py
      embedding.txt
    models/
      base.py
      seq_lr.py          # logistic / linear
      seq_mlp_bce.py     # NN BCE
      seq_mlp_softf1.py  # NN soft-F1
      pyboost_mt.py      # ProtBoost-style
      gcn_stack.py       # GCN trên GO
      postprocess.py     # CondProbMod + DAG fix
    eval/
      cafa_metric.py
      threshold_search.py
    runners/
      train.py
      infer.py
      kfold_oof.py
      make_submission.py
  outputs/
    logs/
    ckpts/
    preds/
    submissions/
```
