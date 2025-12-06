import os
import argparse
import gc

import numpy as np
import torch
from Bio import SeqIO

from transformers import (
    AutoTokenizer, AutoModel,
    T5Tokenizer, T5EncoderModel,
    EsmTokenizer,
)

# pip install -e . in folder SaProt
try:
    from model.saprot.base import SaprotBaseModel
    HAS_SAPROT = True
except ImportError:
    HAS_SAPROT = False


def load_fasta(fasta_path):
    seqs, ids = [], []
    for r in SeqIO.parse(fasta_path, "fasta"):
        seqs.append(str(r.seq))
        ids.append(str(r.id))
    return seqs, ids


def mean_pool(last_hidden, attention_mask):
    # last_hidden: (B, L, D), attention_mask: (B, L)
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    sum_vec = torch.sum(last_hidden * mask, dim=1)
    denom = torch.clamp(mask.sum(1), min=1e-9)
    mean = sum_vec / denom
    return mean


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True,
                        help="HuggingFace model name or SaProt config path")
    parser.add_argument("--short_name", type=str, required=True,
                        help="Short name for saving, e.g. esm2_650M")
    parser.add_argument("--fasta_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--use_fp16", action="store_true")

    parser.add_argument("--is_prott5", action="store_true")
    parser.add_argument("--is_protbert", action="store_true")
    parser.add_argument("--is_saprot", action="store_true")
    parser.add_argument("--is_ankh", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {args.model_name} ({args.short_name})")
    print(f"FASTA:  {args.fasta_path}")

    # -------------------------------------------------
    # 1) Load sequences
    # -------------------------------------------------
    sequences, ids = load_fasta(args.fasta_path)
    print(f"Found {len(sequences)} sequences")

    # -------------------------------------------------
    # 2) Detect model type
    # -------------------------------------------------
    model_name_lower = args.model_name.lower()

    is_saprot = args.is_saprot or ("saprot" in model_name_lower or "saprot" in args.short_name.lower())
    is_prott5 = args.is_prott5 or ("prot_t5" in model_name_lower)
    is_protbert = args.is_protbert or ("prot_bert" in model_name_lower)
    is_ankh     = args.is_ankh     or ("ankh3"    in model_name_lower or "ankh-" in model_name_lower)

    # -------------------------------------------------
    # 3) Load tokenizer + model
    # -------------------------------------------------
    if is_saprot:
        if not HAS_SAPROT:
            raise ImportError(
                "SaProt not importable. Make sure you cloned SaProt repo and "
                "`pip install -e .` trong thư mục SaProt."
            )

        print("Loading SaProt model...")
        cfg = {
            "task": "base",
            "config_path": args.model_name,  # model_name is local PATH to folder SaProt_650M_AF2
            "load_pretrained": True,
        }
        model = SaprotBaseModel(**cfg)
        tokenizer = EsmTokenizer.from_pretrained(cfg["config_path"])

        preprocess = lambda s: s  # SaProt seq can be used directly (if you have encoded # pLDDT)

        # FP16 for SaProt
        if args.use_fp16 and device.type == "cuda":
            model = model.half()

    elif is_prott5:
        print("Loading ProtT5-XL (T5Tokenizer + T5EncoderModel)...")
        tokenizer = T5Tokenizer.from_pretrained(args.model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(args.model_name)
        preprocess = lambda s: " ".join(list(s))

        if args.use_fp16 and device.type == "cuda":
            model = model.half()

    elif is_ankh:
        print("Loading Ankh (T5-style encoder)...")
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        model = T5EncoderModel.from_pretrained(args.model_name)
        preprocess = lambda s: "[S2S]" + s

        if args.use_fp16 and device.type == "cuda":
            model = model.half()
    else:
        # ESM2 / ESM1b / ProtBERT / ... : HF AutoModel
        print("Loading HuggingFace model via AutoModel...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            do_lower_case=False,
            use_fast=True
        )
        model = AutoModel.from_pretrained(args.model_name)

        if is_protbert:
            preprocess = lambda s: " ".join(list(s))
        else:
            preprocess = lambda s: s

        if args.use_fp16 and device.type == "cuda":
            model = model.half()

    model.to(device)
    model.eval()

    # -------------------------------------------------
    # 4) Embedding loop
    # -------------------------------------------------
    all_emb_blocks = []

    with torch.no_grad():
        n = len(sequences)
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            batch_seqs_raw = sequences[start:end]

            batch_seqs = [preprocess(s) for s in batch_seqs_raw]

            # tokenize
            inputs = tokenizer(
                batch_seqs,
                padding=True,
                truncation=True,
                max_length=args.max_len,
                return_tensors="pt"
            )

            # move to device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            # forward
            if is_saprot:
                emb = model.get_hidden_states(inputs, reduction="mean")  # (B, D)
                pooled = emb

            else:
                outputs = model(**inputs)
                if hasattr(outputs, "last_hidden_state"):
                    last_hidden = outputs.last_hidden_state  # (B, L, D)
                else:
                    raise ValueError(f"Model {args.short_name} does not provide last_hidden_state")

                attn = inputs["attention_mask"]
                pooled = mean_pool(last_hidden, attn)  # (B, D)

            emb_np = pooled.detach().cpu().numpy().astype(np.float16)
            all_emb_blocks.append(emb_np)

            # cleanup
            del inputs, pooled
            if not is_saprot:
                del outputs
            torch.cuda.empty_cache()
            gc.collect()

            print(f"Processed {end}/{n}", end="\r", flush=True)

    # -------------------------------------------------
    # 5) Save
    # -------------------------------------------------
    final_emb = np.concatenate(all_emb_blocks, axis=0) if all_emb_blocks else np.zeros((0, 0), dtype=np.float16)
    print("\nEmbedding shape:", final_emb.shape)
    
    save_path = os.path.join(args.save_dir, args.short_name)
    os.makedirs(save_path, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.fasta_path))[0]

    emb_path = os.path.join(save_path, f"{base_name}_emb.npy")
    ids_path = os.path.join(save_path, f"{base_name}_ids.npy")

    np.save(emb_path, final_emb)
    np.save(ids_path, np.array(ids))

    print(f"Saved embeddings: {emb_path}")
    print(f"Saved ids:        {ids_path}")

    # cleanup
    model.to("cpu")
    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
