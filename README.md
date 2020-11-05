# veRNAml
Repository for my comp402 project

**Supervisor:** Jerome Waldispuhl

**Coordinator:** Yue Li

# Project Goals
The overall goal of this project will be to combine, tune and expand upon these programs to provide a tool which is interpretable for biologists. Currently RNAMigos predicts small molecule binding partners however I would like to extend this to predict proteins and functional annotations. This task will use motif fingerprints from VeRNAl and data from PDB, Rfam and GO. We will rely on interpretable prediction models at first (ie. decisions trees) to determine motif importance. The overall result will be cleaned and curated data sources; tuned parameters for VeRNAl motif extraction; trained classification models for RNA-{protein/rna/small molecule/function} prediction and novel functional insights on conserved RNA structural patterns.

## Associated Repositories:
[VeRNAl](https://github.com/cgoliver/vernal)

[RNAMigos](https://github.com/cgoliver/RNAmigos)

# 1. Data Preparation
The training data for VernaML consists of netowrkx graphs which are sliced into portions containing RNA interfaces and their respective complements.
To generate this data, find interfaces on a representative set of RCSB PDB structures using the prepare\_data package.

This process can take some time loading all the PDB structures then subsequently slicing their graphs. Alternatively the pre-built following datasets can be downloaded from MEGA:
| Dataset | # Graphs | # Avg. Nodes | # Edges | Download Link |
| -------- | -------- | -------- | ----- | ---- |
| RNA-RNA     |      |      |       |      |
| RNA-Protein |      |      |       |      |
| RNA-Small Mol. |      |      |       |      |
| RNA-Ion |      |      |       |      |

|	Dataset 	    |Graphs | Edges| Nodes  |Avg. Nodes | Avg. Edges|Link|
|---------------------------|-------:|------:|--------:|-------:|-----------:|-------|
|ALL                         |2679   | 447225 | 641968  |166.9|239.6|
|ALL complement              |9034   | 195395 | 228261 | 21.6 |25.3|
|RNA-Ion                     |572    | 3490   | 3764   | 6.1  |6.6|
|RNA-Ion complement          |493    | 3691   | 3993   | 7.5  |8.1|
|RNA-Protein                 |2750   | 411487 | 587961 | 149.6|213.8|
|RNA-Protein complement      |8265   | 241611 | 322324 | 29.2 |39.0|
|RNA-RNA                     |2737   | 59333  | 79116  | 21.7 |28.9|
|RNA-RNA complement          |2483   | 55001  | 70551  | 22.2 |28.4|
|RNA-Small\_Mol.             |166    | 981    | 1004   | 5.9 |6.0|
|RNA-Small\_Mol. complement  |140    | 973    | 1038   | 7.0|7.4|

## Download PDB Structures
To avoid redundancies in the training data the BGSU representative set of RNAs are used.
They can be downloaded from [here](http://rna.bgsu.edu/rna3dhub/nrlist/release/3.145) [1]

Make a directory to store the structures
`mkdir data/structures`

Then run the following command to retrieve the PDB structures from the RCSB database
`python prepare_data/retrieve_structures.py <BGSU file> data/structures`

## 




# References
1. Leontis, N. B., & Zirbel, C. L. (2012). Nonredundant 3D Structure Datasets for RNA Knowledge Extraction and Benchmarking. In RNA 3D Structure Analysis and Prediction N. Leontis & E. Westhof (Eds.), (Vol. 27, pp. 281–298). Springer Berlin Heidelberg. doi:10.1007/978-3-642-25740-7\_13

