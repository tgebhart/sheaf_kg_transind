# Understanding the code base 

## 7/17: 
* so far I have checked out **coboundary** in extension.py, and have tracked down where this is used. The coboundary function takes as input the **restriction maps**, something called the **edge_index** (figure out what this is!), and an optional param for relabeling (not sure what this is for, but seems less important than the others), and produces the coboundary matrix mapping between the vertex space and the edge space of the relation (**TODO**: sanity check with dimensions. Is this right?)
    * Main calculation I don't understand is the construction of the indices of non-zero values. Look @ the paper for this + write out some simple examples by hand.

* I looked at KGExtender, the base class for
sheaf-based kg embedding. This doesn't implement the full range of assumed methods, so I looked at SEEmbedding, which seems to be the simplest KG embedding approach. 
* the basic usage of models that subclass KGExtender is as follows: 
    * select optimal hyperparameters
    * do training with ^^ on some knowledge graph G
    * given new pieces of the graph that we don't know, diffuse the embeddings from known edges/vertices to the unfamiliar parts by iteratively applying the sheaf laplacian.

* *TODO*: what exactly is harmonic extension lol? is it just the above process? UPDATE: harmonic extension is the process that iteratively applying the laplacian is **approximating**. 
* *TODO*: explore pykeen + run some simple models (ex: SE aka structured Embedding) on some sample data to get a feel for how they work (as well as expected methods/properties). 
* *TODO*: what does it mean to diffuse the *interior* of something? Find out what this actually means haha

## 7/20: 

* Using full_run_from_inductive_hpo as my starting point, I looked at `diffuse_interior` in extension.py. 
    * What defines the interior/boundary? The data, it seems. I wanted to track down where `interior_ent_msk` comes from. 

* I looked at `expand_entity_embeddings` and `expand_model_to_inductive_graph` in utils.py, wrote some doc strings, and added some type hints. These are key functions used in process of extending embeddings to unseen entities. 

* I also looked at complex_data_info.py. Not sure what this is for yet. Too many single-letter variables 😰

## 7/23: 
* tracking `extend_best_from_hpo`.
    1. call `get_train_eval_inclusion_data` to get data structure that is a combo of the training + evaluation knowledge graphs: 
        ```json
        {
            "orig": {"graph": orig_graph, "triples": orig_triples},
            "eval": {"graph": eval_graph, "triples": eval_triples},
            "inclusion": {
                "entities": orig_eval_entity_inclusion,
                "relations": orig_eval_relation_inclusion,
            },
        }
        ```
        * Question: what is the difference between a **graph** and a **triples factory**? We provide both in this data structure, but in the code they're both TriplesFactories that are intialized by the same data. Seems redundant.
        * the inclusion maps `orig_eval_entity_inclusion` and `orig_eval_relation_inclusion` are just mappings of entities (resp. relations) that exist in the base knowledge graph to entities (resp. relations) in the extended knowledge graph. 
    
    2. Load the trained model. 
    3. Extend the trained model to the extended knowledge graph using `expand_model_to_inductive_graph`. 
    4. Diffuse the embeddings of the **known** entities (the *boundary* entities) to the **unknown** ones (the *interior* entities). 

* Found a copy of the dataset https://github.com/DeepGraphLearning/pLogicNet/tree/master/data/FB15k-237
    * this was helpful just for understanding what the structure of the dataset is like.

* Think I figured out what "interior" vertices are: They are the entities from our extended graph which we have not seen in the original graph. The terminology had me confused for a minute!

* Added a lot of docstrings + type hints in `data_tools.py`.

* TODO: Start adding type hints + docstrings to `SEExtender`. 
    * TODO: talk to tom about what makes the diffusion process unstable. 
    * Consider pre-computing powers of the Laplacian using an optimized matrix exponentiation implementation, like https://pytorch.org/docs/stable/generated/torch.linalg.matrix_power.html, then applying to embedding once. Would this make a difference? 
