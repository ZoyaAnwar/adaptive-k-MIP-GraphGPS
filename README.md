# k-Maximum Inner Product Attention for Graph Transformers and the Expressive Power of GraphGPS

> Adaptive-*k* extension notes (what we changed / how to run on Colab): [`README_ADAPTIVE_K.md`](README_ADAPTIVE_K.md)

[![OpenReview](https://img.shields.io/badge/OpenReview-b31b1b.svg)](https://openreview.net/forum?id=4Y5kxbH2fI)
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/pdf/2604.03815)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

In this work, we introduce k-MIP self-attention for graph transformers. This codebase contains all code required to reproduce the experiments in the paper.

Main contributions:
- We introduce k-Maximum Inner Product (k-MIP) self-attention for graph transformers, which achieves linear memory complexity and yields up to a ten-fold speedup over a PyTorch implementation of full attention. As a result, k-MIP attention enables the processing of graphs with over 500k nodes on a single A100 GPU.
- We show that k-MIP attention can be seamlessly integrated into the GraphGPS framework and provide a theoretical analysis of its expressive power, establishing an upper bound on the graph-distinguishing capability of GraphGPS in terms of the S-SEG-WL test [(Zhu et al., 2023)](https://dl.acm.org/doi/pdf/10.1145/3580305.3599451). This analysis clarifies how positional and structural encodings enable expressivity in graph transformers.
- We prove that k-MIP transformers can approximate any full-attention transformer to arbitrary precision, thereby guaranteeing that the proposed sparsification does not reduce the expressive power of transformer-based architectures.
- We empirically demonstrate competitive performance against other scalable graph transformers on a range of benchmarks, including the Long Range Graph Benchmark (LRGB) [(Dwivedi et al., 2022)](https://arxiv.org/abs/2206.08164), the City-Networks benchmark [(Liang et al., 2025)](https://arxiv.org/abs/2503.09008), and two custom large-scale inductive point cloud datasets based on ShapeNet-Part [(Yi et al., 2016)](https://dl.acm.org/doi/10.1145/2980179.2980238) and S3DIS [(Armeni et al., 2016)](https://ieeexplore.ieee.org/document/7780539).


## Environment setup with conda

In the project's main directory, run:
```bash
CONDA_OVERWRITE_FILES=1 conda env create -f environment.full.yml -n kmipattn
conda activate kmipattn
```

## Navigating the codebase

We highlight some important files and folders in the codebase.

| File/Folder                          | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `configs`                            | Contains configuration files for all experiments and datasets.        |
| `run/experiments.py`              | Implements the command-line interface that was used to run the experiments.             |
| `graphgps/layer/sparse_attention_layer.py` | Implementation of the sparse attention layer used in the k-MIP Graph Transformer. |
| `profiling-experiment` | This subfolder contains the code that was used to benchmark different attention mechanisms. |


## Using the CLI

The CLI in `run/experiments.py` can be used to run all experiments. 

A pick of the most important arguments:

```
positional arguments:
  {print,test,real}     Mode to run: print (just print commands that will be executed), test (run commands in test setting), or real (execute commands)

optional arguments:
  --datasets            {COCO,PeptStruct,PeptFunc,Pascal,PCQM,Paris,Shanghai,LA,London,ShapeNet,S3DIS,all,CityNetworks,LRGB,PointClouds}
  --methods             {GAT,GatedGCN,GCN,GINE,BigBird,kmip,Performer,Transformer,Exphormer,all,LRGB,GTs,GNNs}
  --commands COMMANDS   Either a mode ("hp-search", "best-of-each", "tuned", "tuned-so-far", "defaults") or a path to a JSON file containing commands to run
  --cfg_overrides CFG_OVERRIDES
                        Additional config overrides as a string
```


## Reproducing the experiments

### Hyperparameter Sweeps

```bash
# For the hyperparameter sweeps
python run/experiments.py real --datasets *dataset_names* --methods *method_names* --commands hp-search --cfg_overrides "wandb.use False"

# For the tuned configs
python run/experiments.py real --datasets *dataset_names* --methods *method_names* --commands tuned --cfg_overrides "wandb.use False"
```
You may override any config key by adding it as extra argument, as we have done above for `wandb.use`


### Controlled Experiment for Computational Efficiency

In section 5.1 of the paper, we benchmark full attention, k-MIP attention (naive) and k-MIP attention (symbolic matrices). The code that was used to execute this experiment, as well as instructions to reproduce the results, can be found in the subfolder [`profiling-experiment`](profiling-experiment/README.md).


## Citation

If this work is useful for your research, please consider citing it using the following bibtex:

```bibtex
@inproceedings{deschouwer26,
  title     = {k-{M}aximum {I}nner {P}roduct {A}ttention for {G}raph {T}ransformers and the {E}xpressive {P}ower of {G}raphGPS},
  author    = {De Schouwer, Jonas and Sáez de Ocáriz Borde, Haitz and Dong, Xiaowen},
  booktitle = {ICLR 2026 Workshop on Geometry-grounded Representation Learning and Generative Modeling (GRaM)},
  year      = {2026},
  note      = {Accepted; available on OpenReview},
  url       = {https://openreview.net/forum?id=4Y5kxbH2fI}
}
```
