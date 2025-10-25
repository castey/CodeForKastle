# DR imports
from sklearn.manifold import TSNE
import umap.umap_ as umap
from pacmap import PaCMAP # try to get this work at some point (also might be easier to use this for PCA)

# pykeen imports
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.utils import set_random_seed

# matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# others
import pandas as pd
import random, torch, os

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
        reduce_and_plot(result=result, val_file=val_file, basename=f"{base_name}_{model}", only_entities=True)

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
    #reduced_PaCMAP = PaCMAP(random_state=seed, n_neighbors=5).fit_transform(entity_emb)
    return { "tSNE": reduced_tSNE, "UMAP": reduced_UMAP } #"PaCMAP": reduced_PaCMAP}

# --- plotting helper ---
def plot_embedding(entity_df, reduced, title="Embedding Projection", outpath=None):
    """
    Plot 2D embedding with color gradient from dodgerblue → red
    based on 'initial_values'.
    """
    entity_df = entity_df.copy()
    entity_df[["x", "y"]] = reduced

    # Define custom gradient: dodgerblue → red (go dodgers!)
    cmap = LinearSegmentedColormap.from_list("blue_red", ["dodgerblue", "red"])
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()

    if "initial_values" in entity_df.columns:
        vals = entity_df["initial_values"].astype(float)
        
        sc = ax.scatter(
            entity_df.x,
            entity_df.y,
            c=vals,
            cmap=cmap,
            s=60,
            alpha=0.9,
            edgecolor="k",
            linewidths=0.4,
            zorder=2
        )
        
        cbar = plt.colorbar(sc)
        cbar.set_label("Initialization Value", fontsize=12)
        
        # annotate each point with its value (centered)
        for _, row in entity_df.iterrows():
            val_str = f"{row.initial_values:.3f}"
            ax.text(
                row.x, row.y, val_str,
                ha="center", 
                va="center",
                fontsize=2.0, 
                fontweight="medium",
                color="white" if row.initial_values > vals.mean() else "black",
                zorder=3,
            )
    else:
        ax.scatter(entity_df.x, entity_df.y, s=12, alpha=0.7, color="gray", edgecolor="k", linewidths=0.2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    plt.tight_layout()

    if outpath:
        svg_path = outpath.replace(".png", ".svg")
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"[Saved SVG] {svg_path}")

    plt.close()

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
        missing = entity_df["initial_values"].isna().sum()
        print(f"\n[INFO] Missing initialization values: {missing}")

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
embed_and_plot("MuRE", seed, 100, dimensionality=None)