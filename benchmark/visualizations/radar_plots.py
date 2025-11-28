import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"   # keep text as text (not paths)

# ======================== CONFIG ========================
DATASETS        = ["orion", "hemit", "pannuke", "lizard", "pathocell"]
BASELINE_DIRS   = ("baseline", "baselines")

# Baseline line styles (no fill)
BASELINE_STYLES = {
    "Upper Bound": {"linestyle": (0, (6, 3)), "fill": False},  # long dashes
    "Random":      {"linestyle": (0, (2, 2)), "fill": False},  # short dashes
}

# ===================== YOUR ORDERS ======================
ORION_PIXEL_ORDER = [
    "Hoechst", "Pan-CK", "E-cadherin", "CD45", "CD3e", "CD20", "CD4",
    "CD45RO", "CD8a", "FOXP3", "PD-L1", "CD68", "CD163", "Ki67", "CD31", "SMA"
]
ORION_CELL_ORDER = [
    "Pan-CK_pos", "E-cadherin_pos", "CD45_pos", "CD3e_pos", "CD20_pos", "CD4_pos", "CD45RO_pos",
    "CD8a_pos", "FOXP3_pos", "PD-L1_pos", "CD68_pos", "CD163_pos", "Ki67_pos", "CD31_pos", "SMA_pos"
]
PATHOCELL_CELL_ORDER = [
    "Tumor cells", "Smooth muscle", "T cells", "B cells", "NK cells",
    "Dendritic cells", "Background", "Adipocytes", "Nerves", "Other cells",
    "Plasma cells", "Stroma", "Macrophages/Monocytes", "Granulocytes",
    "Vasculature/Lymphatics",
]
LIZARD_CELL_ORDER = [
    "Epithelial", "Connective tissue", "Plasma", "Eosinophil", "Neutrophil", "Lymphocyte"
]
PANNNUKE_CELL_ORDER = [
    "Epithelial", "Connective/Soft tissue cells", "Inflammatory",
    "Neoplastic cells", "Dead cells", "Background"
]

# Map dataset/space -> template
ORDER_TEMPLATES = {
    ("orion",      "pixel"): ORION_PIXEL_ORDER,
    ("orion",      "cell"):  ORION_CELL_ORDER,
    ("pathocell",  "cell"):  PATHOCELL_CELL_ORDER,
    ("lizard",     "cell"):  LIZARD_CELL_ORDER,
    ("pannuke",    "cell"):  PANNNUKE_CELL_ORDER,
    # hemit: no enforced order → alphabetical
}

# Optional hard exclusions (exact label names as they appear in CSVs)
EXCLUDE_LABELS = {
    ("orion", "cell"): {"PD-1_pos"},   # 🚫 hide PD-1+ for ORION cell plots
    # If you also want to hide PD-1 at pixel level, add:
    # ("orion", "pixel"): {"PD-1"},
}

CUSTOM_COLORS = {
    "MIPHEI-vit":          "#0072B2",  # blue
    "MIPHEI-convnext":     "#008D67",  # vermillion
    "pix2pix":             "#56B4E9",  # green
    "rosie_orion":         "#8D0F9E",  # reddish purple
    "diffusion_ft":        "#EEE460",  # yellow
    "HEMIT":               "#F5AC74",  # light blue
    "nuclear_morphometry": "#F14747",  # gray
}

# ==================== SMALL HELPERS =====================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip spaces and normalize key column names."""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    lower = {c.lower(): c for c in df.columns}
    if "marker" in lower: df = df.rename(columns={lower["marker"]: "marker"})
    if "f1 score" in lower: df = df.rename(columns={lower["f1 score"]: "F1 Score"})
    if "f1" in lower:       df = df.rename(columns={lower["f1"]: "F1 Score"})
    return df

def find_baseline_dir(root: str):
    for name in BASELINE_DIRS:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    return None

def random_f1_candidates(baseline_dir: str, dataset: str):
    return [
        os.path.join(baseline_dir, f"{dataset}_random_f1.csv"),
        os.path.join(baseline_dir, f"{dataset}_f1_random.csv"),
    ]

# --------- name normalization just for ordering/display ----------
def _canon_label(label: str) -> str:
    """Normalize a label for comparison/ordering (hyphen variants, connective_tissue, etc.)."""
    x = label.strip()
    # Common harmonizations
    x = x.replace("Ecadherin", "E-cadherin")
    x = x.replace("E-Cadherin", "E-cadherin")
    x = x.replace("Pan CK", "Pan-CK").replace("Pan-CK", "Pan-CK")
    x = x.replace("Connective_tissue", "Connective tissue")
    x = x.replace("Dead Cells", "Dead cells")
    # Keep _pos as-is for cell order (display will add + later)
    return x

def order_labels(labels, dataset: str, space: str):
    """
    Order raw labels according to template for (dataset, space).
    Unknowns go after template, alphabetically by canonical form.
    """
    template = ORDER_TEMPLATES.get((dataset, space))
    if not template:
        # No template → alphabetical by canonical form
        return sorted(labels, key=lambda s: _canon_label(s).lower())

    # Build index for quick lookup on canonicalized template
    t_index = { _canon_label(t): i for i, t in enumerate(template) }

    def key_fn(lbl):
        c = _canon_label(lbl)
        return (0, t_index[c]) if c in t_index else (1, c.lower())

    return sorted(labels, key=key_fn)

# ==================== DATA LOADING ======================
def load_all_metrics(checkpoint_dir):
    cell_perf, pixel_perf = {}, {}

    # Regular models (skip baseline folders)
    for model_name in os.listdir(checkpoint_dir):
        mpath = os.path.join(checkpoint_dir, model_name)
        if not os.path.isdir(mpath) or model_name.lower().startswith("baseline"):
            continue

        for ds in DATASETS:
            logreg = os.path.join(mpath, f"{ds}_logreg.csv")
            if os.path.exists(logreg):
                df = normalize_columns(pd.read_csv(logreg))
                if "marker" in df.columns:
                    cell_perf.setdefault(ds, {})[model_name] = df

            pixel = os.path.join(mpath, f"{ds}_pixel_metrics.csv")
            if os.path.exists(pixel):
                df = normalize_columns(pd.read_csv(pixel))
                if "marker" in df.columns:
                    pixel_perf.setdefault(ds, {})[model_name] = df

    # Baselines → Upper Bound (+ Random for F1)
    bdir = find_baseline_dir(checkpoint_dir)
    if bdir:
        for ds in DATASETS:
            ub_cell = os.path.join(bdir, f"{ds}_logreg.csv")
            if os.path.exists(ub_cell):
                df = normalize_columns(pd.read_csv(ub_cell))
                if "marker" in df.columns:
                    cell_perf.setdefault(ds, {})["Upper Bound"] = df

            ub_pixel = os.path.join(bdir, f"{ds}_pixel_metrics.csv")
            if os.path.exists(ub_pixel):
                df = normalize_columns(pd.read_csv(ub_pixel))
                if "marker" in df.columns:
                    pixel_perf.setdefault(ds, {})["Upper Bound"] = df

            found = None
            for cand in random_f1_candidates(bdir, ds):
                if os.path.exists(cand):
                    found = cand
                    break
            if found:
                df = normalize_columns(pd.read_csv(found))
                if {"marker", "F1 Score"}.issubset(df.columns):
                    cell_perf.setdefault(ds, {})["Random"] = df
                else:
                    print(f"⚠ Random F1 exists but missing columns in {found}")

    return cell_perf, pixel_perf

# ===================== PLOTTING =========================
def radar_plot(df_dict, metric, title, save_path, model_colors, dataset: str, space: str):
    TITLE_FSIZE, LABEL_FSIZE, YTICK_FSIZE, LEGEND_FSIZE = 16, 12, 11, 11

    if not df_dict:
        print(f"⚠ SKIP {title} — no models.")
        return

    # Union of markers from all models in this plot
    raw_labels = sorted(set().union(*[
        df["marker"].tolist() for df in df_dict.values() if "marker" in df.columns
    ]))
    if not raw_labels:
        print(f"⚠ SKIP {title} — no markers.")
        return

    # 🚫 Drop hard-excluded labels for this dataset/space
    excluded = EXCLUDE_LABELS.get((dataset, space), set())
    raw_labels = [lbl for lbl in raw_labels if lbl not in excluded]

    # Order them by your template (dataset, space)
    labels = order_labels(raw_labels, dataset, space)

    # Display: replace _pos → +
    labels_display = [lbl[:-4] + "+" if lbl.endswith("_pos") else lbl for lbl in labels]

    # angles
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)
    ax.set_title(title, fontsize=TITLE_FSIZE, pad=20)

    plotted = False
    for name, df in df_dict.items():
        if metric not in df.columns:  # ignore models missing this metric (e.g., Random for non-F1)
            continue

        arr = (df.set_index("marker")
                 .reindex(labels)[metric]
                 .astype(float)
                 .fillna(0.0)
                 .to_numpy())
        arr = np.concatenate([arr, [arr[0]]])

        color = model_colors.get(name)
        style = BASELINE_STYLES.get(name, {})
        linestyle = style.get("linestyle", "-")
        do_fill = style.get("fill", True)

        ax.plot(angles, arr, label=name, color=color, linewidth=2, linestyle=linestyle)
        if do_fill:
            ax.fill(angles, arr, alpha=0.15, color=color)
        plotted = True

    if not plotted:
        print(f"⚠ No model had column '{metric}' in {title}.")
        plt.close(fig)
        return

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_display, fontsize=LABEL_FSIZE)
    ax.tick_params(axis='x', pad=10)

    # === Adaptive radial limits and tick labels ===
    if metric.lower() == "psnr":
        # PSNR → start from 0 for fair visual comparison
        all_values = np.concatenate([
            df[metric].values for df in df_dict.values() if metric in df.columns
        ])
        vmax = np.nanmax(all_values)
        ax.set_ylim(0, vmax + 2)

        # Set reasonable tick spacing depending on max value
        step = 10 if vmax > 40 else 5
        yticks = np.arange(0, vmax + step, step)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.0f}" for y in yticks], fontsize=YTICK_FSIZE)
    else:
        # Default for normalized metrics (AUC, F1, SSIM, Pearson, etc.)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["", "0.4", "", "0.8", ""], fontsize=YTICK_FSIZE)

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=LEGEND_FSIZE)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    root, _ = os.path.splitext(save_path)
    """svg_path = f"{root}.svg"
    plt.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0.2, transparent=True)"""
    plt.close()

# ====================== DRIVER ==========================
def run(checkpoint_dir, save_dir):
    cell_perf, pixel_perf = load_all_metrics(checkpoint_dir)

    # Consistent colors across all plots (Matplotlib default cycle)
    all_models = sorted(set(
        m for d in list(cell_perf.values()) + list(pixel_perf.values()) for m in d.keys()
    ))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_colors = {}
    for i, m in enumerate(all_models):
        if m in CUSTOM_COLORS:
            print(m, CUSTOM_COLORS[m])
            model_colors[m] = CUSTOM_COLORS[m]
        else:
            print(m)
            model_colors[m] = default_colors[i % len(default_colors)]
    model_colors["Upper Bound"] = "black"
    model_colors["Random"] = "gray"

    # Cell plots (space="cell")
    for ds, models in cell_perf.items():
        for metric in ["F1 Score", "ROC AUC", "AUPRC"]:
            out = os.path.join(save_dir, f"{ds}_cell_{metric}.png")
            radar_plot(models, metric, f"{ds} — Cell {metric}", out, model_colors, dataset=ds, space="cell")
            print(f"✅ Saved {out}")

    # Pixel plots (space="pixel")
    for ds, models in pixel_perf.items():
        for metric in ["psnr", "ssim", "pearson_r"]:
            out = os.path.join(save_dir, f"{ds}_pixel_{metric}.png")
            radar_plot(models, metric, f"{ds} — Pixel {metric}", out, model_colors, dataset=ds, space="pixel")
            print(f"✅ Saved {out}")

    print("\n✅ DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    run(args.checkpoints_dir, args.save_dir)
