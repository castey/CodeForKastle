# DR imports
from sklearn.manifold import TSNE
import umap.umap_ as umap
from pacmap import PaCMAP # try to get this work at some point (also might be easier to use this for PCA)
from sklearn.decomposition import PCA

# pykeen imports
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.utils import set_random_seed

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
# import matplotlib.colors as mcolors

# others
import pandas as pd
import random, torch, os, re
import numpy as np

# seed random and torch with predefined fixed seed
seed = 42
random.seed(seed)
torch.manual_seed(seed)
set_random_seed(seed)

# create embeddings and call dimentionality reduction and plotter function
# we only need to call this function
def embed_and_plot(model, seed, training_epochs, dimensionality):
    
    # find synthetic KGs
    input_dir = "KGs"
    os.makedirs(input_dir, exist_ok=True)

    print("Loading all KG TSV files...")

    tsv_files = [f for f in os.listdir(input_dir) if f.endswith(".tsv")]

    if not tsv_files:
        raise FileNotFoundError("No .tsv files found in ./KGs directory.")

    # for every tsv file, define the training, testing, and validation sets and embed
    for tsv_file in tsv_files:
        base_name = os.path.splitext(tsv_file)[0]
        path = os.path.join(input_dir, tsv_file)
        
        # pull filename for entity and their initalization values from windower
        val_file = f"{tsv_file}.val"

        print(f"\n--- Processing {base_name} ---")

        # define training, testing, validation sets
        tf = TriplesFactory.from_path(path)
        training, testing, validation = tf.split([0.8, 0.1, 0.1])

        # create embeddings
        print(f"Training {model}...")
        result = pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model=model,
            random_seed=seed,
            model_kwargs= { 'embedding_dim': dimensionality },
            training_kwargs={ "num_epochs": training_epochs }
        )

        print(f"Training complete for {base_name}!")

        # dimensionality reduction and plotting
        reduce_and_plot(result=result, val_file=val_file, basename=f"{base_name}_{model}_DE-{dimensionality}", only_entities=True)

def plot_distances(values, mean_val, std_val,
                   cos_sims=None, mean_cos=None, std_cos=None,
                   metric_title="L₂ Distance",
                   comparison_label="Embedded vs. Expected E₃",
                   outpath=None):
    """
    Generic plotter for distance or similarity metrics.

    Parameters
    ----------
    values : list or np.ndarray
        Metric values to plot (L2, cosine, etc.)
    mean_val : float
        Mean of the primary metric.
    std_val : float
        Standard deviation of the primary metric.
    cos_sims : list, optional
        Optional cosine similarity values for a second plot.
    mean_cos, std_cos : float, optional
        Mean and SD for the cosine similarities.
    metric_title : str
        Axis label for the y-axis ("L₂ Distance", "Cosine Similarity", etc.)
    comparison_label : str
        Used in titles/filenames to describe what’s being compared.
    outpath : str
        Output path for saving plots (as SVGs).
    """
    plt.ioff()

    # === 1. PRIMARY METRIC (bars) ===
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = np.arange(len(values))
    ax.bar(indices, values, color="steelblue", alpha=0.85,
           edgecolor="black", linewidth=0.4)

    # Annotate mean ± SD
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1.2,
               label=f"Mean = {mean_val:.3f}")
    ax.axhline(mean_val + std_val, color="gray", linestyle=":",
               linewidth=0.8, label=f"±SD = {std_val:.3f}")
    ax.axhline(mean_val - std_val, color="gray", linestyle=":", linewidth=0.8)

    ax.set_title(f"{metric_title} — {comparison_label}", fontsize=13)
    ax.set_xlabel("Triplet Index", fontsize=11)
    ax.set_ylabel(metric_title, fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()

    if outpath:
        base = os.path.splitext(outpath)[0]
        svg_path = f"{base}_{comparison_label.replace(' ', '_')}_L2.svg"
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"[Saved {metric_title} bar plot] {svg_path}")
    plt.close(fig)

    # === 2. COSINE SIMILARITY (optional) ===
    if cos_sims is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.arange(len(cos_sims))
        ax.bar(indices, cos_sims, color="orange", alpha=0.8,
               edgecolor="black", linewidth=0.4)

        ax.axhline(mean_cos, color="red", linestyle="--", linewidth=1.2,
                   label=f"Mean cos = {mean_cos:.3f}")
        ax.axhline(mean_cos + std_cos, color="gray", linestyle=":", linewidth=0.8,
                   label=f"±SD = {std_cos:.3f}")
        ax.axhline(mean_cos - std_cos, color="gray", linestyle=":", linewidth=0.8)

        ax.set_title(f"Cosine Similarity — {comparison_label}", fontsize=13)
        ax.set_xlabel("Triplet Index", fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()

        if outpath:
            base = os.path.splitext(outpath)[0]
            svg_path = f"{base}_{comparison_label.replace(' ', '_')}_cosine.svg"
            plt.savefig(svg_path, format="svg", bbox_inches="tight")
            print(f"[Saved cosine bar plot] {svg_path}")
        plt.close(fig)
    
# --- classify node type ---
def classify_node(name):
    if name == "Window_root":
        return "root"
    elif re.match(r"^E\d+$", str(name)):
        return "entity"
    elif "Window" in str(name):
        return "window"
    else:
        return "other"

# --- dimensionality reduction helpers ---
def compute_reductions(entity_emb):
    """Return dictionary of DR results for t-SNE, UMAP, PaCMAP."""
    reduced_tSNE = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000).fit_transform(entity_emb)
    reduced_UMAP = umap.UMAP(n_components=2, random_state=seed).fit_transform(entity_emb)
    reduced_PCA = PCA(n_components=2, random_state=seed).fit_transform(entity_emb)
    #reduced_PaCMAP = PaCMAP(random_state=seed, n_neighbors=5).fit_transform(entity_emb)
    return { "PCA": reduced_PCA, "tSNE": reduced_tSNE, "UMAP": reduced_UMAP } #, "PaCMAP": reduced_PaCMAP }

# --- plotting helper ---
def plot_embedding(entity_df, reduced, title="Embedding Projection", outpath=None):
    """
    Generate two 2D embedding plots:
    1. Gradient version (dodgerblue → red) based on 'initial_values'.
    2. Categorical version using 8 discrete cool→warm color bins.

    Windows (gray) and root nodes (red) are shown if present.
    Displays entity/window counts on each plot.
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    import numpy as np
    import pandas as pd

    entity_df = entity_df.copy()
    entity_df[["x", "y"]] = reduced

    # === Split by group type ===
    has_group = "group" in entity_df.columns
    entities = entity_df[entity_df["group"] == "entity"] if has_group else entity_df
    windows = entity_df[entity_df["group"] == "window"] if has_group else pd.DataFrame(columns=entity_df.columns)
    roots = entity_df[entity_df["group"] == "root"] if has_group else pd.DataFrame(columns=entity_df.columns)

    # === Extract numeric values ===
    if "initial_values" not in entities.columns:
        print("[WARN] No 'initial_values' column found — skipping color encoding.")
        vals = np.zeros(len(entities))
    else:
        vals = entities["initial_values"].astype(float)

    mean_val = vals.mean() if len(vals) > 0 else 0
    cmap_grad = LinearSegmentedColormap.from_list("blue_red", ["dodgerblue", "red"])

    # ==========================================================
    # 1. GRADIENT PLOT
    # ==========================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Windows (if any) ---
    if len(windows) > 0:
        ax.scatter(
            windows.x,
            windows.y,
            s=40,
            color="lightgray",
            edgecolor="k",
            linewidths=0.3,
            alpha=0.7,
            label="Windows",
            zorder=1,
        )

    # --- Roots (if any) ---
    if len(roots) > 0:
        ax.scatter(
            roots.x,
            roots.y,
            s=90,
            color="red",
            marker="^",
            edgecolor="k",
            linewidths=0.4,
            alpha=0.9,
            label="Root",
            zorder=2,
        )

    # --- Entities ---
    if len(entities) > 0:
        sc = ax.scatter(
            entities.x,
            entities.y,
            c=vals,
            cmap=cmap_grad,
            s=60,
            edgecolor="k",
            linewidths=0.4,
            label="Entities",
            zorder=3,
        )

        # Annotate entities with numeric values
        for _, row in entities.iterrows():
            val_str = f"{row.initial_values:.3f}" if not pd.isna(row.initial_values) else ""
            ax.text(
                row.x, row.y, val_str,
                ha="center", va="center",
                fontsize=2.5,
                fontweight="medium",
                color="white" if (not pd.isna(row.initial_values) and row.initial_values > mean_val) else "black",
                zorder=4,
            )

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Initialization Value", fontsize=12)

    ax.set_title(f"{title} — Gradient", fontsize=14)
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)

    # === Entity/Window counts ===
    n_entities = len(entities)
    n_windows = len(windows) + len(roots)
    count_text = f"Entities: {n_entities:,} | Windows: {n_windows:,}"
    ax.text(
        0.02, 0.98, count_text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="semibold",
        color="black",
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.4, alpha=0.8),
    )

    plt.tight_layout()

    if outpath:
        svg_path = outpath.replace(".png", "_gradient.svg")
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"[Saved Gradient SVG] {svg_path}")

    plt.close(fig)

    # ==========================================================
    # 2. CATEGORICAL (8 bins)
    # ==========================================================
    if len(entities) > 0:
        label_colors = [
            "#13315C", "#155E75", "#208B82", "#34A853",
            "#B7950B", "#DAA520", "#E6B422", "#FEE191",
        ]
        bins = np.linspace(vals.min(), vals.max(), 9)
        bin_indices = np.digitize(vals, bins) - 1
        bin_indices = np.clip(bin_indices, 0, 7)

        cmap_cat = ListedColormap(label_colors)
        fig, ax = plt.subplots(figsize=(10, 8))

        # Windows
        if len(windows) > 0:
            ax.scatter(
                windows.x,
                windows.y,
                s=40,
                color="lightgray",
                edgecolor="k",
                linewidths=0.3,
                alpha=0.7,
                label="Windows",
                zorder=1,
            )

        # Roots
        if len(roots) > 0:
            ax.scatter(
                roots.x,
                roots.y,
                s=90,
                color="red",
                marker="^",
                edgecolor="k",
                linewidths=0.4,
                alpha=0.9,
                label="Root",
                zorder=2,
            )

        # Entities
        sc = ax.scatter(
            entities.x,
            entities.y,
            c=bin_indices,
            cmap=cmap_cat,
            s=60,
            edgecolor="k",
            linewidths=0.4,
            label="Entities",
            zorder=3,
        )

        for _, row in entities.iterrows():
            val_str = f"{row.initial_values:.3f}" if not pd.isna(row.initial_values) else ""
            ax.text(
                row.x, row.y, val_str,
                ha="center", va="center",
                fontsize=2.5,
                fontweight="medium",
                color="white" if (not pd.isna(row.initial_values) and row.initial_values > mean_val) else "black",
                zorder=4,
            )

        cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(0.5, 8.5, 1))
        cbar.ax.set_yticklabels([f"Bin {i+1}" for i in range(8)])
        cbar.set_label("Initialization Value (Categorical)", fontsize=12)

        ax.set_title(f"{title} — Categorical Bins", fontsize=14)
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)

        # Entity/Window count
        ax.text(
            0.02, 0.98, count_text,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="semibold",
            color="black",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.4, alpha=0.8),
        )

        plt.tight_layout()

        if outpath:
            svg_path = outpath.replace(".png", "_categories.svg")
            plt.savefig(svg_path, format="svg", bbox_inches="tight")
            print(f"[Saved Category SVG] {svg_path}")

        plt.close(fig)
        
# --- main pipeline ---
def reduce_and_plot(result, val_file, outdir="embedding_plots", basename=None, only_entities=True):
    """
    Compute t-SNE, UMAP projections for a PyKEEN result,
    and save each as PNG image in the given output directory.

    Parameters
    ----------
    result : pykeen.pipeline.PipelineResult
        The result of a trained PyKEEN pipeline.
    outdir : str
        Directory to save images (will create subdir per model if basename given).
    basename : str, optional
        Used to name subdirectory and output files (e.g., the KG filename).
    only_entities : bool
        If True, filter out any Window or Root nodes.
    """
    import os
    import pandas as pd
    import numpy as np

    if basename:
        outdir = os.path.join(outdir, basename)
    os.makedirs(outdir, exist_ok=True)

    # --- Extract embeddings and metadata ---
    entity_emb = result.model.entity_representations[0]().detach().numpy()
    entities = list(result.training.entity_to_id.keys())
    
    # Sanity check alignment
    assert entity_emb.shape[0] == len(entities), "Entity count and embedding matrix mismatch!"
    
    print(entities)
    entity_df = pd.DataFrame(entity_emb, index=entities)
    entity_df["group"] = [classify_node(e) for e in entity_df.index]
    
    def natural_sort_key(label):
        match = re.match(r"([A-Za-z]+)(\d+)", label)
        if match:
            prefix, num = match.groups()
            return (prefix, int(num))
        return (label, 0)

    # kick out windows
    entity_df_4_mean = entity_df[entity_df["group"] == "entity"]
    entity_df_4_mean = entity_df_4_mean.select_dtypes(include=[np.number])
    
    sorted_pairs = sorted(zip(entity_df_4_mean.index, entity_df_4_mean.values), key=lambda x: natural_sort_key(x[0]))
    entity_tuples = [(label, np.array(vec, dtype=float)) for label, vec in sorted_pairs]
    
    # --- Extended mean relationship test ---
    L2_expected, L2_embedded = [], []
    L2_parent_expected, L2_parent_embedded = [], []
    cos_sims = []
    triplet_labels = []

    for i in range(0, len(entity_tuples) - 2, 3):
        e1, v1 = entity_tuples[i]
        e2, v2 = entity_tuples[i + 1]
        e3, v3 = entity_tuples[i + 2]   # embedded E3

        # Expected mean vector
        calc_avg = (v1 + v2) / 2

        # --- Main metric: E₃_expected ↔ E₃_embedded ---
        L2_exp_emb = np.linalg.norm(calc_avg - v3)
        L2_expected.append(L2_exp_emb)

        # --- Additional diagnostics ---
        # Distance from E₁/E₂ to expected E₃
        L2_parent_expected.extend([
            np.linalg.norm(v1 - calc_avg),
            np.linalg.norm(v2 - calc_avg)
        ])

        # Distance from E₁/E₂ to embedded E₃
        L2_parent_embedded.extend([
            np.linalg.norm(v1 - v3),
            np.linalg.norm(v2 - v3)
        ])

        # Cosine similarity of expected vs. embedded E₃
        cos_sim = np.dot(calc_avg, v3) / (np.linalg.norm(calc_avg) * np.linalg.norm(v3))
        cos_sims.append(cos_sim)

        triplet_labels.append((e1, e2, e3))

    # --- Summaries ---
    def stats(arr):
        return np.mean(arr), np.std(arr)

    avg_exp, std_exp = stats(L2_expected)
    avg_pExp, std_pExp = stats(L2_parent_expected)
    avg_pEmb, std_pEmb = stats(L2_parent_embedded)
    avg_cos, std_cos = stats(cos_sims)

    print(f"\nE₃_expected ↔ E₃_embedded: {avg_exp:.4f} ± {std_exp:.4f}")
    print(f"E₁/E₂ ↔ E₃_expected: {avg_pExp:.4f} ± {std_pExp:.4f}")
    print(f"E₁/E₂ ↔ E₃_embedded: {avg_pEmb:.4f} ± {std_pEmb:.4f}")
    print(f"Cos sim (expected, embedded E₃): {avg_cos:.4f} ± {std_cos:.4f}")

    # --- Plots ---
    plot_distances(L2_expected, avg_exp, std_exp,
                cos_sims=cos_sims, mean_cos=avg_cos, std_cos=std_cos,
                metric_title="L₂ Distance",
                comparison_label="E₃ Expected vs. Embedded",
                outpath=os.path.join(outdir, f"{basename}_E3_expected_vs_embedded.svg"))

    plot_distances(L2_parent_expected, avg_pExp, std_pExp,
                metric_title="L₂ Distance",
                comparison_label="Parents vs. E₃ Expected",
                outpath=os.path.join(outdir, f"{basename}_parent_to_expected.svg"))

    plot_distances(L2_parent_embedded, avg_pEmb, std_pEmb,
                metric_title="L₂ Distance",
                comparison_label="Parents vs. E₃ Embedded",
                outpath=os.path.join(outdir, f"{basename}_parent_to_embedded.svg"))

    # --- Filter and merge initialization values ---
    if only_entities:
        # Just entities
        entity_df = entity_df[entity_df["group"] == "entity"]
        print("[INFO] Visualizing entities only.")
    elif only_entities is False:
        # Keep all nodes (entities + windows + roots)
        print("[INFO] Visualizing all node types (entities, windows, roots).")

    # Load initialization values (these only apply to 'entity' nodes)
    vals_df = pd.read_csv(f"KGs/{val_file}", sep="\t", header=None, names=["entity", "initial_values"])
    vals_df = vals_df.set_index("entity")

    # Merge the initialization values onto the dataframe (non-entities will get NaN)
    entity_df = entity_df.join(vals_df, how="left")

    # --- Debug info ---
    print("\n[DEBUG] entity_df summary after merge:")
    print("Shape:", entity_df.shape)
    print("Group counts:\n", entity_df["group"].value_counts())
    print("Columns:", list(entity_df.columns))
    print("\nColumn dtypes:\n", entity_df.dtypes)

    print("\nSample initialization values (entity → init_value):")
    print(entity_df["initial_values"].dropna().head(10))

    missing_mask = entity_df["initial_values"].isna()
    missing_count = missing_mask.sum()
    print(f"\n[INFO] Missing initialization values (non-entities expected): {missing_count}")


        # Optional full table if you want to inspect entity–value pairs
        # print(entity_df[["initial_values"]])
        
    # --- Compute reductions ---
    # Get embedding dimensionality directly from the model
    d = result.model.entity_representations[0]().shape[1]

    # Extract only the embedding columns
    embedding_matrix = entity_df.iloc[:, :d].values

    # Run your DR
    reductions = compute_reductions(embedding_matrix)

    # --- Plot and save each ---
    for name, reduced in reductions.items():
        outpath = os.path.join(outdir, f"{basename or 'embedding'}_{name}.png")
        plot_embedding(entity_df, reduced, f"{name} Projection ({basename})", outpath)

    print(f"\n[Saved all plots for {basename}] → {os.path.abspath(outdir)}")
    return entity_df


# MuRE dimensionality default = 200 range(50,300)
# TransE dimensionality default = 50 range(50,300)
# TransR dimensionality default = 50 range not given 
# TransD dimensionality default = 50 range not given

embed_and_plot(model="MuRE", seed=seed, training_epochs=100, dimensionality=300)