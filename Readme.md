# OPTransE

This is the source code for paper ''Representation Learning with Ordered Relation Paths for Knowledge Graph Completion (EMNLP 2019)''.
For paper: https://www.aclweb.org/anthology/D19-1268/

### Dataset and Model
For dataset and model, please refer to https://drive.google.com/open?id=1Jo39CqhwtigD1J3hoADZ6o7xpqKKqkwT.

### Compile
```sh
$ g++ -I ./ train_OPTransE.cpp -o train_OPTransE -O2 -fopenmp -lpthread
$ g++ -I ./ test_OPTransE.cpp -o test_OPTransE -O2 -fopenmp -lpthread
```
### Run
Train
```sh
$ ./train_OPTransE -datapath FB15K/
```
Test
```sh
$ ./test_OPTransE  -datapath FB15K/
```

Note that it will take a long time for training and testing.
