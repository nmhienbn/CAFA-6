import pandas as pd

# Đổi path cho đúng của bạn
mf_path = "output/base_model_kfold/submission_MF.tsv"
bp_path = "output/base_model_kfold/submission_BP.tsv"
cc_path = "output/base_model_kfold/submission_CC.tsv"

out_path = "output/base_model_kfold/submission.tsv"

# Không có header → header=None, đặt tên cột cho dễ xử lý
cols = ["EntryID", "term", "score"]

df_mf = pd.read_csv(mf_path, sep="\t", header=None, names=cols)
df_bp = pd.read_csv(bp_path, sep="\t", header=None, names=cols)
df_cc = pd.read_csv(cc_path, sep="\t", header=None, names=cols)

# Gộp
df = pd.concat([df_mf, df_bp, df_cc], ignore_index=True)

# Bỏ trùng theo (protein, term) nếu có
df = df.drop_duplicates(subset=["EntryID", "term"], keep="first")

# Ghi lại đúng format CAFA/Kaggle: không header, tab, 3 cột
df.to_csv(out_path, sep="\t", header=False, index=False)

print("Done, saved to:", out_path)
