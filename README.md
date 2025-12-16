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
      external/  # UniProt, STRING,...
    processed/
      mapping/   # id2protein, id2go, splits
  features/
    ankh3/
    esm2/
    protT5/
    taxon/
    goa/
    ppi/
    structure/   # Foldseek, Blast,...
  src/
    data/
      <!-- Extract models -->
    features/
      <!-- Extract models -->
    notebooks/
      <!-- All my old notebooks -->
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






https://geneontology.org/docs/guide-go-evidence-codes/