There are two evaluation datasets. 

1. MATRIX-BASED

Start with matrix of some size (the default is 10x10) with random values. 
Each levels gives some rule to manipulate the matric with increasing difficulty. 
There are 28 levels. The score is calculated by how many levels the model passes. 
The code contains the core logic for the dataset generation

2. TEXT-BASED
The evaluation mirrors the structure of the existing matrix-based benchmark.
30-level text-based instruction-following evaluation benchmark designed to be less dependent on arithmetic or spatial reasoning!
Each task begins with a reproducibly randomized list of synthetic tokens and applies a fixed sequence of increasingly complex transformation instructions.
