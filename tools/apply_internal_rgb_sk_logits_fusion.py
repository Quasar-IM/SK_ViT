import argparse
import os
import zipfile

import numpy as np


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    required = {"ids", "rgb_logits", "sk_logits"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} missing keys: {sorted(missing)}")
    return {
        "ids": data["ids"],
        "rgb_logits": data["rgb_logits"].astype(np.float32),
        "sk_logits": data["sk_logits"].astype(np.float32),
    }


def internal_to_submission_label(pred_label):
    return 31 if int(pred_label) == 0 else int(pred_label) - 1


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def main(args):
    data = load_npz(args.input_npz)
    fused_logits = (
        float(args.alpha_rgb) * data["rgb_logits"]
        + float(args.beta_sk) * data["sk_logits"]
    ).astype(np.float32)
    fused_probs = softmax(fused_logits).astype(np.float32)
    fused_preds = fused_logits.argmax(axis=1).astype(np.int64)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{args.output_basename}.csv")
    zip_path = os.path.join(output_dir, f"{args.output_basename}.zip")
    npz_path = os.path.join(output_dir, f"{args.output_basename}.npz")

    with open(csv_path, "w") as f:
        f.write("Id,Target\n")
        for sample_id, pred in zip(data["ids"], fused_preds):
            f.write(f"{sample_id},{internal_to_submission_label(pred)}\n")

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, os.path.basename(csv_path))

    np.savez_compressed(
        npz_path,
        ids=data["ids"],
        rgb_logits=data["rgb_logits"],
        sk_logits=data["sk_logits"],
        logits=fused_logits,
        probs=fused_probs,
        preds=fused_preds,
        alpha_rgb=np.float32(args.alpha_rgb),
        beta_sk=np.float32(args.beta_sk),
    )

    print(f"saved csv: {csv_path}")
    print(f"saved zip: {zip_path}")
    print(f"saved npz: {npz_path}")
    print(f"num_samples: {len(fused_preds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply learned internal RGB/SK logit fusion weights to test predictions.")
    parser.add_argument("--input_npz", type=str, required=True)
    parser.add_argument("--alpha_rgb", type=float, required=True)
    parser.add_argument("--beta_sk", type=float, required=True)
    parser.add_argument("--output_dir", type=str, default="./submission/fusion")
    parser.add_argument("--output_basename", type=str, default="Submission_fusion_internal_rgb_sk")
    main(parser.parse_args())
