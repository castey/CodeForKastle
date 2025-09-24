# Windower pipeline

- generates n entities 
- generates a random value within a range associated with each entity
- min max scales the values in that range 
- recursively "windows" entities in that range
- outputs a turtle file of the entities related to their windows and which entities their associated value are lesser than

- [turtle file from sample run 9/24](./kg.ttl)

## Usage and Example Output

```
# run with 10 "entities" associated with random values randomly sampled from a uniform distribution at a range of 0 to 10 with turtle file 

(env) dave@owners-MacBook-Pro-6 windower % python windower.py --n_entities 10 --depth 4 --D uniform --low 0 --high 10 --save ./kg.ttl

Raw entity values: [('E3', 0.6056438387309071), ('E1', 0.9915127044792194), ('E2', 2.021741087344683), ('E4', 2.6787168328860886), ('E6', 2.759826627930768), ('E9', 4.095050011403403), ('E7', 6.987929626153724), ('E8', 8.427303978065627), ('E5', 8.535579654426279), ('E0', 9.835451635704274)]
Normalized values: [('E3', 0.0), ('E1', 0.042), ('E2', 0.153), ('E4', 0.225), ('E6', 0.233), ('E9', 0.378), ('E7', 0.691), ('E8', 0.847), ('E5', 0.859), ('E0', 1.0)]

Windows:
Window path=root, depth=4, elems=[('E3', 0.0), ('E1', 0.042), ('E2', 0.153), ('E4', 0.225), ('E6', 0.233), ('E9', 0.378), ('E7', 0.691), ('E8', 0.847), ('E5', 0.859), ('E0', 1.0)]
Window path=root->L, depth=3, elems=[('E3', 0.0), ('E1', 0.042), ('E2', 0.153), ('E4', 0.225), ('E6', 0.233)]
Window path=root->L->L, depth=2, elems=[('E3', 0.0), ('E1', 0.042)]
Window path=root->L->L->L, depth=1, elems=[('E3', 0.0)]
Window path=root->L->L->R, depth=1, elems=[('E1', 0.042)]
Window path=root->L->R, depth=2, elems=[('E2', 0.153), ('E4', 0.225), ('E6', 0.233)]
Window path=root->L->R->L, depth=1, elems=[('E2', 0.153)]
Window path=root->L->R->R, depth=1, elems=[('E4', 0.225), ('E6', 0.233)]
Window path=root->R, depth=3, elems=[('E9', 0.378), ('E7', 0.691), ('E8', 0.847), ('E5', 0.859), ('E0', 1.0)]
Window path=root->R->L, depth=2, elems=[('E9', 0.378), ('E7', 0.691)]
Window path=root->R->L->L, depth=1, elems=[('E9', 0.378)]
Window path=root->R->L->R, depth=1, elems=[('E7', 0.691)]
Window path=root->R->R, depth=2, elems=[('E8', 0.847), ('E5', 0.859), ('E0', 1.0)]
Window path=root->R->R->L, depth=1, elems=[('E8', 0.847)]
Window path=root->R->R->R, depth=1, elems=[('E5', 0.859), ('E0', 1.0)]

Less-than relationships:
E3 < ['E1', 'E2', 'E4', 'E6', 'E9', 'E7', 'E8', 'E5', 'E0']
E1 < ['E2', 'E4', 'E6', 'E9', 'E7', 'E8', 'E5', 'E0']
E2 < ['E4', 'E6', 'E9', 'E7', 'E8', 'E5', 'E0']
E4 < ['E6', 'E9', 'E7', 'E8', 'E5', 'E0']
E6 < ['E9', 'E7', 'E8', 'E5', 'E0']
E9 < ['E7', 'E8', 'E5', 'E0']
E7 < ['E8', 'E5', 'E0']
E8 < ['E5', 'E0']
E5 < ['E0']
E0 < []

KG saved to ./kg.ttl

```
