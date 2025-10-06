"""
Windower: Hierarchical Windowing and RDF Knowledge Graph Export

This module builds a hierarchical "window" tree over a set of entities
with numeric values. Entities are first normalized (min–max scaling),
then recursively split into subwindows to a given depth. The structure,
entity relationships, and normalized values are exported directly to an
RDF file in N-Triples format.

Usage (command line):
    python windower.py --n_entities 10 --depth 3 --D uniform --low 0 --high 10 \
        --sort_input True --verbose True --outfile output.nt

Arguments:
    --n_entities   Number of entities to generate (default: 10)
    --depth        Depth of the window tree (default: 4)
    --D            Distribution for random values: "uniform" or "normal" (default: uniform)
    --low          Lower bound for distribution (default: 0.0)
    --high         Upper bound for distribution (default: 10.0)
    --verbose      Print details of the pipeline (default: True)
    --outfile      Path to save the RDF graph (N-Triples format, .nt recommended)
    --precision    Optional precision step size (e.g., 1e-4).

Example:
    python windower.py --n_entities 5 --depth 2 --D normal --low 0 --high 100 \
        --outfile mygraph.nt

This will generate 5 entities with random values from a normal distribution
between 0 and 100, build a depth-2 window tree, print details, and save the
RDF graph to "mygraph.nt".
"""


import argparse
from typing import Sequence, Optional, Tuple, List, Literal
from rdflib import Namespace
import tools

EX = Namespace("http://example.org/")

class Window:
    def __init__(self, elements: Sequence[Tuple[str, float]], n: int, path="root"):
        self.elements: List[Tuple[str, float]] = list(elements)
        self.n = n
        self.path = path
        self.left: Optional["Window"] = None
        self.right: Optional["Window"] = None

    def __repr__(self):
        return f"Window(path={self.path}, n={self.n}, size={len(self.elements)})"

def windower(pairs: Sequence[Tuple[str, float]], n: int = 1, path="root") -> Window:
    node = Window(pairs, n, path=path)
    if n <= 1 or len(pairs) <= 1:
        return node
    left, right = tools.split_pairs(pairs)
    node.left = windower(left, n - 1, path + "->L")
    node.right = windower(right, n - 1, path + "->R")
    return node

def print_windows(node: Window, ndigits=3):
    if node is None:
        return
    elems = [(e, round(v, ndigits)) for e, v in node.elements]
    print(f"Window path={node.path}, depth={node.n}, elems={elems}")
    if node.left:
        print_windows(node.left, ndigits)
    if node.right:
        print_windows(node.right, ndigits)

# ========= RDF Export =========
def build_kg(root: Window, norm_pairs: List[Tuple[str, float]], outfile: str, lt_mode: Literal["sequential", "pairwise"]):
    """
    Write triples to a TSV file usable by PyKEEN.
    Excludes hasValue datatype triples for embedding use.
    """
    
    vals_file = f"{outfile}.val.txt"

    with open(outfile, "w") as f, open(vals_file, "w") as vf:
        # --------- Write window triples ---------
        def add_window_triples(node: Window):
            window_label = f"Window_{node.path.replace('->','_')}"
            f.write(f"{window_label}\ttype\tWindow\n")

            for e, v in node.elements:
                f.write(f"{e}\ttype\tEntity\n")
                f.write(f"{e}\tinWindow\t{window_label}\n")

            if node.left:
                add_window_triples(node.left)
            if node.right:
                add_window_triples(node.right)

        add_window_triples(root)

        # --------- Write lessThan edges only ---------
        sorted_pairs = sorted(norm_pairs, key=lambda x: x[1])

        if lt_mode == "pairwise":
            for i, (e1, _) in enumerate(sorted_pairs):
                for e2, _ in sorted_pairs[i+1:]:
                    f.write(f"{e1}\tlessThan\t{e2}\n")

        elif lt_mode == "sequential":
            for (e1, _), (e2, _) in zip(sorted_pairs, sorted_pairs[1:]):
                f.write(f"{e1}\tlessThan\t{e2}\n")
                
        # --------- Log values with entity labels ---------
        for e, v in sorted_pairs:
            vf.write(f"{e}\t{v}\n")

# ========= Pipeline =========
def run_windower_pipeline(
    entities_with_values: dict,
    depth: int,
    lt_mode: str,
    verbose: bool,
    outfile: str
):
    pairs = list(entities_with_values.items())

    norm_pairs = tools.minmax_scale_pairs(pairs)
    root = windower(norm_pairs, depth)

    if verbose:
        print("Raw entity values:", pairs)
        print("Normalized values:", [(e, round(v, 3)) for e, v in norm_pairs])
        print("\nWindows:")
        print_windows(root)

    # Build RDF graph
    build_kg(root=root, norm_pairs=norm_pairs, outfile=outfile, lt_mode=lt_mode)


# ========= Main =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run windower pipeline on entities.")
    parser.add_argument("--n_entities", type=int, default=10,
                        help="Number of entities to generate (int > 0).")
    parser.add_argument("--depth", type=int, default=4,
                        help="Depth of the window tree (int > 0).")
    parser.add_argument("--D", type=str, default="uniform",
                        choices=["uniform", "normal"],
                        help="Distribution type.")
    parser.add_argument("--low", type=float, default=0.0,
                        help="Lower bound for distribution.")
    parser.add_argument("--high", type=float, default=10.0,
                        help="Upper bound for distribution (must be > low).")
    parser.add_argument("--verbose", type=lambda x: (str(x).lower() == "true"),
                        default=True,
                        choices=[True, False],
                        help="Print details (True/False).")
    parser.add_argument("--precision", type=float, default=None,
                        help="Optional precision step size (e.g., 1e-4).")
    parser.add_argument("--lt_mode", type=str, required=True,
                        choices=["pairwise", "sequential"],
                        help="Save less-than edges as pairwise (complete) or sequential (chain).")
    parser.add_argument("--outfile", type=str, required=True,
                        help="Path to save the RDF graph (N-Triples format, .nt recommended).")

    args = parser.parse_args()

    # generate entity array
    entity_array = tools.gen_entity_array(args.n_entities)

    # assign random values using chosen distribution and range
    entities_with_rand_nums = {
        entity: tools.random_from_distribution(args.D, args.low, args.high, args.precision)
        for entity in entity_array
    }

    # run pipeline
    run_windower_pipeline(
        entities_with_rand_nums,
        depth=args.depth,
        lt_mode=args.lt_mode,
        verbose=args.verbose,
        outfile=args.outfile
    )
