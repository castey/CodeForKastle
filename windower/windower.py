import argparse
import numpy as np
from typing import Sequence, Optional, Tuple, List, Dict
import tools
from rdflib import Graph, Namespace, RDF, URIRef, Literal, XSD

EX = Namespace("http://example.org/")

class Window:
    def __init__(self, elements: Sequence[Tuple[str, float]], n: int, path="root"):
        self.elements: List[Tuple[str, float]] = list(elements)
        self.n = n
        self.path = path  # unique identifier (root, root->L, etc.)
        self.left: Optional["Window"] = None
        self.right: Optional["Window"] = None

    def __repr__(self):
        return f"Window(path={self.path}, n={self.n}, size={len(self.elements)})"

# ========= Helpers =========
def minmax_scale_pairs(pairs: Sequence[Tuple[str, float]]):
    if not pairs:
        return []
    values = np.array([v for _, v in pairs], dtype=float)
    mn, mx = values.min(), values.max()
    if mn == mx:
        return [(e, 0.0) for e, _ in pairs]
    normed = (values - mn) / (mx - mn)
    return [(e, float(v)) for (e, _), v in zip(pairs, normed)]

def split_pairs(pairs: Sequence[Tuple[str, float]]):
    mid = len(pairs) // 2
    return pairs[:mid], pairs[mid:]

def windower(pairs: Sequence[Tuple[str, float]], n: int = 1, path="root") -> Window:
    node = Window(pairs, n, path=path)
    if n <= 1 or len(pairs) <= 1:
        return node
    left, right = split_pairs(pairs)
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

def compute_less_than(pairs: Sequence[Tuple[str, float]]):
    result = {e: [] for e, _ in pairs}
    for e1, v1 in pairs:
        for e2, v2 in pairs:
            if v1 < v2:
                result[e1].append(e2)
    return result

# ========= RDF Export =========
def build_kg(root: Window, lt_map: Dict[str, List[str]], norm_pairs: List[Tuple[str, float]]) -> Graph:
    g = Graph()
    g.bind("ex", EX)

    # Recursively add windows + memberships
    def add_window_triples(node: Window):
        window_uri = EX[f"Window_{node.path.replace('->','_')}"]
        g.add((window_uri, RDF.type, EX.Window))

        for e, v in node.elements:
            entity_uri = EX[e]
            g.add((entity_uri, RDF.type, EX.Entity))
            g.add((entity_uri, EX.inWindow, window_uri))

        if node.left:
            add_window_triples(node.left)
        if node.right:
            add_window_triples(node.right)

    add_window_triples(root)

    # Add lessThan relations
    for e1, smaller_than_list in lt_map.items():
        for e2 in smaller_than_list:
            g.add((EX[e1], EX.lessThan, EX[e2]))

    # Add numeric values (normalized)
    for e, v in norm_pairs:
        g.add((EX[e], EX.hasValue, Literal(v, datatype=XSD.float)))

    return g

# ========= Pipeline =========
def run_windower_pipeline(
    entities_with_values: dict,
    depth: int = 3,
    sort_input: bool = True,
    verbose: bool = True,
):
    pairs = list(entities_with_values.items())

    if sort_input:
        pairs = sorted(pairs, key=lambda x: x[1])

    norm_pairs = minmax_scale_pairs(pairs)
    root = windower(norm_pairs, depth)

    if verbose:
        print("Raw entity values:", pairs)
        print("Normalized values:", [(e, round(v, 3)) for e, v in norm_pairs])
        print("\nWindows:")
        print_windows(root)

        print("\nLess-than relationships:")
        lt_map = compute_less_than(norm_pairs)
        for e, smaller_than in lt_map.items():
            print(f"{e} < {smaller_than}")
    else:
        lt_map = compute_less_than(norm_pairs)

    # Build RDF graph
    g = build_kg(root, lt_map, norm_pairs)
    return norm_pairs, root, g

# ========= Main =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run windower pipeline on entities.")
    parser.add_argument("--n_entities", type=int, default=10,
                        help="Number of entities to generate.")
    parser.add_argument("--depth", type=int, default=4,
                        help="Depth of the window tree.")
    parser.add_argument("--D", type=str, default="uniform",
                        choices=["uniform", "normal"],
                        help="Distribution type (uniform or normal).")
    parser.add_argument("--low", type=float, default=0.0,
                        help="Lower bound for distribution.")
    parser.add_argument("--high", type=float, default=10.0,
                        help="Upper bound for distribution.")
    parser.add_argument("--sort_input", type=lambda x: (str(x).lower() == "true"),
                        default=True,
                        help="Sort input before normalization (True/False).")
    parser.add_argument("--verbose", type=lambda x: (str(x).lower() == "true"),
                        default=True,
                        help="Print details (True/False).")
    parser.add_argument("--save", type=str, default=None,
                        help="Optional path to save KG as Turtle file.")
    args = parser.parse_args()

    # generate entity array
    entity_array = tools.gen_entity_array(args.n_entities)

    # assign random values using chosen distribution and range
    entities_with_rand_nums = {
        entity: tools.random_from_distribution(args.D, args.low, args.high)
        for entity in entity_array
    }

    # run pipeline
    norm_pairs, root, g = run_windower_pipeline(
        entities_with_rand_nums,
        depth=args.depth,
        sort_input=args.sort_input,
        verbose=args.verbose,
    )

    # Save KG if requested
    if args.save:
        g.serialize(args.save, format="turtle")
        print(f"\nKG saved to {args.save}")
