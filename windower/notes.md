Tuesday must do 
embedder: murE -> something something something Embeddings
plots without windows

lt relationships pick 5 at random for each entity x

look over windower logic make sure its good

# 10/21 notes

PCA - priority PUST DO PCA ON EMBEDDINGS

TransE dimensionality 300 

MuRE dimensionality try extremes

Color code based on windows
Color gradient of values for all PCA, Tsne, UMAP

Look into values inside datapoints on plot

# 10/28 notes
## pre-meeting notes:
- lobe clusters are related to the rand5_lt relationships
- dimensionality of MuRE seems to have a negligible effect 

## post-meeting notes:
generate e1, e2, -> e3 will be the midpoint between e1 and e2 plus some perturbation 

calculate vector midpoint distance between the embeddings of e1 and e2 to e3

calculate the distance between this expected vector and the actual learned e3 embedding

do for all e1 to e3, e4 to e6, etc

from Andrea:
Umap sequential only, then put 5, plus 10 and pairwise , technically plus 1, plus 5, plus 10 and then pairwise —> 16 and 32 depth 
E2 to be midpoint of e1 and e3 in terms of initialization values  we want to see the distance between the points