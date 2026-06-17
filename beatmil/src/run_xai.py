"""Run Grad-CAM XAI evaluation against LUDB expert annotations.

Loads the Beat-MIL intra-DB checkpoint, generates saliency maps on
each LUDB record, computes IoU/Dice against the expert P/QRS/T masks.

Run: python run_xai.py
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from xai import evaluate_xai


def main():
    ckpt = Path.home() / "beatmil/checkpoints/beatmil_intra-db/best.pt"
    ludb = Path.home() / "beatmil/data/ludb"
    out_dir = Path.home() / "beatmil/outputs/eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt.exists():
        print(f"[error] {ckpt} not found — train Beat-MIL intra-DB first")
        sys.exit(1)
    if not ludb.exists():
        print(f"[error] {ludb} not found — download LUDB first")
        sys.exit(1)

    print(f"[xai] checkpoint: {ckpt}")
    print(f"[xai] LUDB: {ludb}")
    summary, _ = evaluate_xai(model_ckpt=ckpt, ludb_root=ludb)
    out_path = out_dir / "xai_summary.json"
    payload = {"beatmil_gradcam": summary}
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\n[xai] saved to {out_path}")


if __name__ == "__main__":
    main()
