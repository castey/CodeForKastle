from sklearn.manifold import TSNE
import umap.umap_ as umap
from pacmap import PaCMAP
import pandas as pd
import matplotlib.pyplot as plt
import os

seed = 42

# --- classify node type ---
def classify_node(name):
    if name == "Window_root":
        return "Root"
    elif name.startswith("E"):
        return "Entity"
    elif "Window" in name:
        return "Window"
    else:
        return "Other"

# --- dimensionality reduction helpers ---
def compute_reductions(entity_emb):
    """Return dictionary of DR results for t-SNE, UMAP, PaCMAP."""
    reduced_tSNE = TSNE(n_components=2, random_state=seed, perplexity=30, max_iter=1000).fit_transform(entity_emb)
    reduced_UMAP = umap.UMAP(n_components=2, random_state=seed).fit_transform(entity_emb)
    #reduced_PaCMAP = PaCMAP(random_state=seed, n_neighbors=5).fit_transform(entity_emb)
    return { "tSNE": reduced_tSNE, "UMAP": reduced_UMAP } #"PaCMAP": reduced_PaCMAP}

# --- plotting helper ---
import matplotlib.pyplot as plt
import numpy as np

def plot_embedding(entity_df, reduced, title="Embedding Projection", outpath=None):
    """
    Plot a 2D embedding with fine-tuned style:
    - smaller points
    - alpha transparency
    - black edge outlines for crispness
    """
    entity_df = entity_df.copy()
    entity_df[["x", "y"]] = reduced

    colors = {"Entity": "dodgerblue", "Window": "orange", "Root": "crimson"}
    plt.figure(figsize=(10, 8))

    for group, subset in entity_df.groupby("group"):
        plt.scatter(
            subset.x, subset.y,
            label=group,
            color=colors.get(group, "gray"),
            alpha=0.7,           # transparency
            s=10 if group == "Entity" else 20,  # smaller entities
            edgecolor='k',       # black outline for contrast
            linewidths=0.2,
        )

    plt.title(title, fontsize=14)
    plt.xlabel("Component 1", fontsize=12)
    plt.ylabel("Component 2", fontsize=12)
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, format="png", dpi=300)
        print(f"[Saved] {outpath}")

    plt.close()

# --- main pipeline ---
def reduce_and_plot(result, outdir="embedding_plots", basename=None, only_entities=True):
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
        entity_df = entity_df[entity_df["group"] == "Entity"]

    # --- Compute reductions ---
    reductions = compute_reductions(entity_df.iloc[:, :-1].values)

    # --- Plot and save each ---
    for name, reduced in reductions.items():
        outpath = os.path.join(outdir, f"{basename or 'embedding'}_{name}.png")
        plot_embedding(entity_df, reduced, f"{name} Projection ({basename})", outpath)

    print(f"\n[Saved all plots for {basename}] → {os.path.abspath(outdir)}")
    return entity_df

