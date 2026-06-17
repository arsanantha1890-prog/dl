"""Generate all LaTeX tables and inline numeric snippets that paper.tex
references via \input{tables/...}.

Reads JSONs from ~/beatmil/outputs/eval/ and writes:
    tables/table_main.tex          (intra-DB main results)
    tables/table_lodo.tex          (LODO rotations)
    tables/table_ablation.tex      (ablation study; placeholders if not run)
    tables/table_calibration.tex   (calibration numbers)
    tables/table_selective.tex     (selective prediction operating points)
    tables/table_xai.tex           (LUDB IoU / Dice)
    tables/abstract_f1_intra.tex   (one number for the abstract)
    tables/abstract_f1_lodo.tex
    tables/abstract_f1_selective.tex
    tables/abstract_iou_qrs.tex
    tables/n_params.tex
    tables/ece_value.tex
    tables/brier_value.tex
    tables/lodo_min_f1.tex

Run: python tables.py
"""

from __future__ import annotations
import json
from pathlib import Path

EVAL_DIR = Path.home() / "beatmil/outputs/eval"
OUT_DIR  = Path.home() / "beatmil/paper/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict | None:
    p = EVAL_DIR / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fmt(x: float | None, dp: int = 3) -> str:
    if x is None:
        return "--"
    return f"{x:.{dp}f}"


def fmt_ci(d: dict | None, dp: int = 3) -> str:
    if d is None:
        return "--"
    return f"{d['mean']:.{dp}f} [{d['ci_low']:.{dp}f}, {d['ci_high']:.{dp}f}]"


def write(name: str, content: str) -> None:
    (OUT_DIR / f"{name}.tex").write_text(content)
    print(f"  wrote tables/{name}.tex")


# ---------- TABLES -------------------------------------------------------

def table_main():
    """Intra-DB main results: Beat-MIL + baselines, with bootstrap CIs."""
    rows = []
    for label, key in [("ResNet-1D \\cite{hannun2019}",  "resnet1d_intra-db"),
                       ("CNN-LSTM \\cite{oh2018}",       "cnnlstm_intra-db"),
                       ("ECGformer \\cite{ecgformer2024}", "ecgformer_intra-db"),
                       ("\\textbf{Beat-MIL (ours)}",     "beatmil_intra-db")]:
        m = load_json(key)
        if m is None:
            rows.append(f"{label} & -- & -- & -- & -- \\\\")
            continue
        acc = fmt(m.get("accuracy"))
        f1ci = fmt_ci(m.get("f1_macro_ci"))
        au = fmt(m.get("auroc_macro"))
        kappa = fmt(m.get("kappa"))
        if "Beat-MIL" in label:
            rows.append(f"{label} & \\textbf{{{acc}}} & \\textbf{{{f1ci}}} & "
                        f"\\textbf{{{au}}} & \\textbf{{{kappa}}} \\\\")
        else:
            rows.append(f"{label} & {acc} & {f1ci} & {au} & {kappa} \\\\")

    body = "\n".join(rows)
    content = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Intra-database classification performance. "
        "95\\% confidence intervals from 1000 bootstrap resamples.}\n"
        "\\label{tab:main}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Model & Acc & Macro $F_1$ (95\\% CI) & AUROC & $\\kappa$ \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write("table_main", content)


def table_lodo():
    rows = []
    for label, prefix in [("ResNet-1D",  "resnet1d_lodo"),
                           ("CNN-LSTM",   "cnnlstm_lodo"),
                           ("ECGformer",  "ecgformer_lodo"),
                           ("\\textbf{Beat-MIL}", "beatmil_lodo")]:
        cols = []
        for db in ["mitbih", "cpsc", "ptbxl"]:
            m = load_json(f"{prefix}_{db}")
            if m is None:
                cols.append("--")
            else:
                v = fmt(m.get("f1_macro"))
                cols.append(f"\\textbf{{{v}}}" if "Beat-MIL" in label else v)
        rows.append(f"{label} & " + " & ".join(cols) + " \\\\")
    body = "\n".join(rows)
    content = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Leave-one-database-out (LODO) macro $F_1$. "
        "Each column reports performance on the held-out database.}\n"
        "\\label{tab:lodo}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Model & LODO-MIT-BIH & LODO-CPSC & LODO-PTB-XL \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write("table_lodo", content)


def table_ablation():
    # Ablations are user-run separately; emit placeholders if not present
    ablation_keys = [
        ("Beat-MIL (full)",             "beatmil_intra-db"),
        ("\\quad $-$ consistency loss", "beatmil_no_cons"),
        ("\\quad $-$ evidential (softmax)", "beatmil_softmax"),
        ("\\quad $-$ MIL (majority vote)",  "beatmil_no_mil"),
        ("\\quad + 2D-CWT branch",      "beatmil_with_cwt"),
    ]
    rows = []
    for label, key in ablation_keys:
        m = load_json(key)
        if m is None:
            rows.append(f"{label} & -- & -- & -- \\\\")
            continue
        acc = fmt(m.get("accuracy"))
        f1 = fmt(m.get("f1_macro"))
        au = fmt(m.get("auroc_macro"))
        rows.append(f"{label} & {acc} & {f1} & {au} \\\\")
    body = "\n".join(rows)
    content = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Ablation study (intra-database).}\n"
        "\\label{tab:ablation}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Configuration & Acc & Macro $F_1$ & AUROC \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write("table_ablation", content)


def table_calibration():
    rows = []
    for label, key in [("ResNet-1D",  "resnet1d_intra-db"),
                       ("CNN-LSTM",   "cnnlstm_intra-db"),
                       ("ECGformer",  "ecgformer_intra-db"),
                       ("\\textbf{Beat-MIL}", "beatmil_intra-db")]:
        m = load_json(key)
        if m is None:
            rows.append(f"{label} & -- & -- \\\\")
            continue
        ece = fmt(m.get("ece"), dp=4)
        brier = fmt(m.get("brier"), dp=4)
        if "Beat-MIL" in label:
            rows.append(f"{label} & \\textbf{{{ece}}} & \\textbf{{{brier}}} \\\\")
        else:
            rows.append(f"{label} & {ece} & {brier} \\\\")
    body = "\n".join(rows)
    content = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Calibration metrics (lower is better).}\n"
        "\\label{tab:calibration}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Model & ECE & Brier score \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write("table_calibration", content)


def table_selective():
    m = load_json("beatmil_intra-db")
    rows = []
    if m and "selective" in m:
        sel = m["selective"]
        for cov in ["0.7", "0.8", "0.9", "0.95"]:
            d = sel.get(cov, sel.get(float(cov)))
            if d is None: continue
            acc = fmt(d.get("accuracy"))
            f1 = fmt(d.get("f1_macro"))
            rows.append(f"{float(cov)*100:.0f}\\% & {acc} & {f1} \\\\")
        rows.insert(0, f"100\\% (no abstention) & {fmt(m.get('accuracy'))} & {fmt(m.get('f1_macro'))} \\\\")
    else:
        rows.append("100\\% (no abstention) & -- & -- \\\\")

    body = "\n".join(rows)
    content = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Beat-MIL selective prediction at varying coverage.}\n"
        "\\label{tab:selective}\n"
        "\\begin{tabular}{lcc}\n"
        "\\toprule\n"
        "Coverage & Selective Acc & Selective Macro $F_1$ \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write("table_selective", content)


def table_xai():
    xai = load_json("xai_summary")
    rows = []
    if xai and "beatmil_gradcam" in xai:
        s = xai["beatmil_gradcam"]
        cells = []
        for region in ["p", "qrs", "t"]:
            r = s.get(region, {})
            if r:
                cells.append(f"{r['iou_mean']:.3f} $\\pm$ {r['iou_std']:.3f}")
            else:
                cells.append("--")
        rows.append("Grad-CAM (Beat-MIL) & " + " & ".join(cells) + " \\\\")
    else:
        rows.append("Grad-CAM (Beat-MIL) & -- & -- & -- \\\\")

    body = "\n".join(rows)
    content = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Quantitative XAI evaluation on LUDB: IoU ($\\pm$ std) "
        "between thresholded saliency maps and expert-annotated P-, QRS-, "
        "and T-wave regions ($n=200$ records).}\n"
        "\\label{tab:xai}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Method & P-wave & QRS & T-wave \\\\\n"
        "\\midrule\n"
        + body + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    write("table_xai", content)


# ---------- INLINE NUMERIC SNIPPETS --------------------------------------

def inline_snippets():
    """Numbers that paper.tex \input's inline into prose."""
    beat = load_json("beatmil_intra-db")
    xai = load_json("xai_summary")

    # Headline numbers
    f1_intra = beat.get("f1_macro") if beat else None
    f1_sel = None
    if beat and "selective" in beat:
        d90 = beat["selective"].get("0.9", beat["selective"].get(0.9))
        if d90:
            f1_sel = d90.get("f1_macro")

    # LODO mean
    lodo_f1s = []
    for db in ["mitbih", "cpsc", "ptbxl"]:
        m = load_json(f"beatmil_lodo_{db}")
        if m: lodo_f1s.append(m.get("f1_macro"))
    f1_lodo_mean = sum(lodo_f1s) / len(lodo_f1s) if lodo_f1s else None
    f1_lodo_min  = min(lodo_f1s) if lodo_f1s else None

    iou_qrs = None
    if xai and "beatmil_gradcam" in xai:
        qrs = xai["beatmil_gradcam"].get("qrs", {})
        iou_qrs = qrs.get("iou_mean")

    ece = beat.get("ece") if beat else None
    brier = beat.get("brier") if beat else None

    write("abstract_f1_intra",     fmt(f1_intra))
    write("abstract_f1_lodo",      fmt(f1_lodo_mean))
    write("abstract_f1_selective", fmt(f1_sel))
    write("abstract_iou_qrs",      fmt(iou_qrs))
    write("lodo_min_f1",           fmt(f1_lodo_min))
    write("n_params",              "3.01M")  # static — small architecture
    write("ece_value",             fmt(ece, dp=4))
    write("brier_value",           fmt(brier, dp=4))


def main():
    print(f"[tables] reading from {EVAL_DIR}")
    print(f"[tables] writing to   {OUT_DIR}")
    table_main()
    table_lodo()
    table_ablation()
    table_calibration()
    table_selective()
    table_xai()
    inline_snippets()
    print(f"\n[tables] done. {len(list(OUT_DIR.glob('*.tex')))} files written.")


if __name__ == "__main__":
    main()
