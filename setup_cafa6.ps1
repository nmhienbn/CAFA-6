$root = "D:\CAFA-6"

$dirs = @(
    ".",                        # cafa6/
    "configs",
    "data",
    "data\raw",
    "data\raw\cafa6",
    "data\raw\cafa6\Train",
    "data\raw\cafa6\Test",
    "data\raw\external",
    "data\processed",
    "data\processed\mapping",
    "data\processed\splits",
    "features",
    "features\ankh3",
    "features\esm2",
    "src",
    "src\data",
    "src\features",
    "src\models",
    "src\eval",
    "src\runners",
    "outputs",
    "outputs\logs",
    "outputs\ckpts",
    "outputs\preds",
    "outputs\submissions"
)

foreach ($d in $dirs) {
    $fullDir = Join-Path $root $d
    New-Item -ItemType Directory -Path $fullDir -Force | Out-Null
}

$files = @(
    # configs
    "configs\data.yaml",
    "configs\seq_mlp_bce.yaml",
    "configs\seq_mlp_softf1.yaml",
    "configs\pyboost_mt.yaml",
    "configs\ensemble.yaml",

    # data/raw/cafa6
    "data\raw\cafa6\Train\train_terms.tsv",
    "data\raw\cafa6\Train\train_sequences.fasta",
    "data\raw\cafa6\Test\testsuperset.fasta",
    "data\raw\cafa6\go.obo",

    # data/processed/mapping
    "data\processed\mapping\protein2idx.json",
    "data\processed\mapping\idx2protein.npy",
    "data\processed\mapping\go2idx_BP.json",
    "data\processed\mapping\go2idx_MF.json",
    "data\processed\mapping\go2idx_CC.json",
    "data\processed\mapping\Y_BP.npz",
    "data\processed\mapping\Y_MF.npz",

    # data/processed/splits
    "data\processed\splits\train_folds.npy",

    # features
    "features\ankh3\train.npy",
    "features\ankh3\test.npy",
    "features\esm2\train.npy",
    "features\esm2\test.npy",

    # src/data
    "src\data\build_dataset.py",
    "src\data\dataset.py",
    "src\data\go_graph.py",
    "src\data\split.py",

    # src/features
    "src\features\gen_ankh3.py",
    "src\features\gen_esm2.py",
    "src\features\gen_prott5.py",

    src/models
    "src\models\base.py",
    "src\models\seq_mlp_bce.py",
    "src\models\seq_mlp_softf1.py",
    "src\models\pyboost_mt.py",
    "src\models\postprocess.py",
    "src\models\ensemble.py",

    # src/eval
    "src\eval\cafa_metric.py",
    "src\eval\threshold_search.py",

    # src/runners
    "src\runners\train.py",
    "src\runners\infer.py",
    "src\runners\kfold_oof.py",
    "src\runners\make_submission.py"
)

foreach ($f in $files) {
    $fullFile = Join-Path $root $f
    if (-not (Test-Path $fullFile)) {
        New-Item -ItemType File -Path $fullFile | Out-Null
    }
}

Write-Host "Done. Structure created at $root"
