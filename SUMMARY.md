# Understanding the code base 

## 7/17: 
* so far I have checked out **coboundary** in extension.py, and have tracked down where this is used. The coboundary function takes as input the **restriction maps**, something called the **edge_index** (figure out what this is!), and an optional param for relabeling (not sure what this is for, but seems less important than the others), and produces the coboundary matrix mapping between the vertex space and the edge space of the relation (**TODO**: sanity check with dimensions. Is this right?)
* * Main calculation I don't understand is the construction of the indices of non-zero values. Look @ the paper for this + write out some simple examples by hand.

* I looked at KGExtender, the base class for
sheaf-based kg embedding. This doesn't implement the full range of assumed methods, so I looked at SEEmbedding, which seems to be the simplest KG embedding approach. 
* the basic usage of models that subclass KGExtender is as follows: 
* * select optimal hyperparameters
* * do training with ^^ on some knowledge graph G
* * given new pieces of the graph that we don't know, diffuse the embeddings from known edges/vertices to the unfamiliar parts by iteratively applying the sheaf laplacian.

* *TODO*: what exactly is harmonic extension lol? is it just the above process? 
* *TODO*: explore pykeen + run some simple models (ex: SE aka structured Embedding) on some sample data to get a feel for how they work (as well as expected methods/properties). 
* *TODO*: what does it mean to diffuse the *interior* of something? Find out what this actually means haha
