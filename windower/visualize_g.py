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
import random, torch, os
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
            dimensions=dimensionality,
            testing=testing,
            validation=validation,
            model=model,
            random_seed=seed,
            training_kwargs=dict(num_epochs=training_epochs),
        )

        print(f"Training complete for {base_name}!")

        # dimensionality reduction and plotting
        reduce_and_plot(result=result, val_file=val_file, basename=f"{base_name}_{model}_DE-{dimensionality}", only_entities=True)

# --- classify node type ---
def classify_node(name):
    if name == "Window_root":
        return "root"
    elif name.startswith("E"):
        return "entity"
    elif "Window" in name:
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
    
    Both include the numeric initialization values printed inside the points:
        - white text for above-mean values
        - black text for below-mean values
    
    Each saved as an SVG for infinite zoom clarity.
    """
    entity_df = entity_df.copy()
    entity_df[["x", "y"]] = reduced

    if "initial_values" not in entity_df.columns:
        print("[WARN] No 'initial_values' column found — skipping color encoding.")
        return

    vals = entity_df["initial_values"].astype(float)
    mean_val = vals.mean()
    plt.ioff()  # no GUI window

    # ==========================================================
    # 1. GRADIENT PLOT (continuous dodgerblue → red)
    # ==========================================================
    cmap_grad = LinearSegmentedColormap.from_list("blue_red", ["dodgerblue", "red"])
    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        entity_df.x,
        entity_df.y,
        c=vals,
        cmap=cmap_grad,
        s=60,
        alpha=0.9,
        edgecolor="k",
        linewidths=0.4,
        zorder=2,
    )

    # Annotate values (white if above mean, black if below)
    for _, row in entity_df.iterrows():
        val_str = f"{row.initial_values:.3f}"
        ax.text(
            row.x, row.y, val_str,
            ha="center", va="center",
            fontsize=2.5,
            fontweight="medium",
            color="white" if row.initial_values > mean_val else "black",
            zorder=3,
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Initialization Value", fontsize=12)

    ax.set_title(f"{title} — Gradient", fontsize=14)
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    plt.tight_layout()

    if outpath:
        svg_path = outpath.replace(".png", "_gradient.svg")
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"[Saved Gradient SVG] {svg_path}")

    plt.close(fig)

    # ==========================================================
    # 2. CATEGORICAL PLOT (8 color bins)
    # ==========================================================
    label_colors = [
        "#13315C",  # dark navy
        "#155E75",  # deep teal
        "#208B82",  # sea green
        "#34A853",  # muted green
        "#B7950B",  # gold tone
        "#DAA520",  # goldenrod
        "#E6B422",  # amber
        "#FEE191",  # pastel yellow
    ]

    bins = np.linspace(vals.min(), vals.max(), 9)
    bin_indices = np.digitize(vals, bins) - 1
    bin_indices = np.clip(bin_indices, 0, 7)

    cmap_cat = ListedColormap(label_colors)
    fig, ax = plt.subplots(figsize=(10, 8))

    sc = ax.scatter(
        entity_df.x,
        entity_df.y,
        c=bin_indices,
        cmap=cmap_cat,
        s=60,
        alpha=0.9,
        edgecolor="k",
        linewidths=0.4,
        zorder=2,
    )

    # Annotate values again
    for _, row in entity_df.iterrows():
        val_str = f"{row.initial_values:.3f}"
        ax.text(
            row.x, row.y, val_str,
            ha="center", va="center",
            fontsize=2.5,
            fontweight="medium",
            color="white" if row.initial_values > mean_val else "black",
            zorder=3,
        )

    cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(0.5, 8.5, 1))
    cbar.ax.set_yticklabels([f"Bin {i+1}" for i in range(8)])
    cbar.set_label("Initialization Value (Categorical)", fontsize=12)

    ax.set_title(f"{title} — Categorical Bins", fontsize=14)
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
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
    if basename:
        outdir = os.path.join(outdir, basename)
    os.makedirs(outdir, exist_ok=True)

    # --- Extract embeddings and metadata ---
    entity_emb = result.model.entity_representations[0]().detach().numpy()
    entities = list(result.training.entity_to_id.keys())
    entity_df = pd.DataFrame(entity_emb, index=entities)
    entity_df["group"] = [classify_node(e) for e in entity_df.index]

    # --- Filter to only entities ---
    if only_entities:
        entity_df = entity_df[entity_df["group"] == "entity"]
        
        # grab the values from the windower initialization to merge to dataframe
        vals_df = pd.read_csv(f"KGs/{val_file}", sep="\t", header=None, names=["entity", "initial_values"])
        vals_df = vals_df.set_index("entity")
        
        # merge by "entity" as index
        entity_df = entity_df.join(vals_df, how="left")
        
        # === Debug printout ===
        print("\n[DEBUG] entity_df summary after merge:")
        print("Shape:", entity_df.shape)
        print("Columns:", list(entity_df.columns))
        print("\nColumn dtypes:\n", entity_df.dtypes)

        # Show a few sample initialization values with entity labels
        print("\nSample initialization values (entity → init_value):")
        print(entity_df["initial_values"].head(10))  # top 10 entities
        print("\nRandom sample of initialization values:")
        print(entity_df["initial_values"].sample(10, random_state=42))

        # Show missing (NaN) values, if any
        # Show missing (NaN) values, if any
        missing_mask = entity_df["initial_values"].isna()
        missing_count = missing_mask.sum()

        print(f"\n[INFO] Missing initialization values: {missing_count}")

        if missing_count > 0:
            missing_entities = entity_df.index[missing_mask].tolist()
            print("[DEBUG] Entities missing initialization values:")
            for e in missing_entities:
                print(f"  - {e}")
                entity_df = entity_df.dropna(subset=["initial_values"])

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

# MuRE dimensionality default = 200
# TransE dimensionality default = 50 
# TransR dimensionality default = 50 
# TransD dimensionality default = 50 

embed_and_plot(model="MuRE", seed=seed, training_epochs=100, dimensionality=200)