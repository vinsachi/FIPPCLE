# Filtered Inner Product Projection
Repository for "Filtered Inner Product Projection for Crosslingual Embedding Alignment" (ICLR 2021) [https://openreview.net/forum?id=A2gNouoXE7]

## Method Code 
The entirety of the code for the FIPP algorithm is contained within the method `project_fipp` in `xling-bli/code/projection.py`.

## Reproduction of Paper Results

All code necessary for reproduction of BLI results and runtime profiling from the paper is contained in `xling-bli/setup.sh`. This code will download the necessary fastText embeddings, serialize them, and produce BLI and runtime results for all 28 language pairs (for both 1K and 5K dictionaries).

To run the reproduction run the below code. If you use the self-learning framework, please first install pytorch for your appropriate CUDA build from https://pytorch.org. 

```
bash xling-bli/setup.sh
```

### BLI Reproduction (XLING 1K and 5K)

BLI reproduction results from the head of this repo for both XLING 1K and 5K are included in `xling-bli/bli_reprod_1K.json` and `xling-bli/bli_reprod_5K.json`. 

- One can reconstruct results by running `fipp_bli_reprod_1k.py` and `fipp_bli_reprod_5k.py` respectively after running the preceding lines in `xling-bli/setup.sh`. 
- Results from reproduction are within [-0.001, +0.003] from numbers reported in the paper. Averaged MAP over all 28 language pairs is provided in the table below

Dataset | Dictionary Size (incl. any Augmentation) | MAP
------------ | ------------- | -------------
XLING 1K (Self Learning + 14K) | 15K |0.407
XLING 5K (No Self Learning) | 5K |0.442

### Runtime Statistics
Runtime statistics results from the head of this repo for both XLING 1K and 5K are included in `xling-bli/runtime_reprod_1k.json` and `xling-bli/runtime_reprod_1k.json`. All results include time taken to load embeddings, construct dictionaries, and save embeddings after alignment. 

- One can reconstruct results by running `fipp_runtime_reprod_1k.py` and `fipp_runtime_reprod_5k.py` respectively after running the preceding lines in `xling-bli/setup.sh`. 
- During productionalization we have made code optimizations to reduce the runtime of XLING 5K by ~30% from 23 seconds to 16.8 seconds
- In our paper, we profile on the EN-DE language pair; the below table includes runtimes averaged over all 28 language pairs. 

Dataset | Dictionary Size (incl. Augmentation) | Runtime (seconds)
------------ | ------------- | -------------
XLING 1K (Self Learning + 14K) | 15K | 180.92
XLING 5K (No Self Learning) | 5K | 16.80

## Environment
- Dependencies for FIPP are included below, although we only utilize standard linear algebraic procedures (Eigendecomposition, Least Squares, SVD) note that different versions of `numpy, scipy` will have variations in their implementation of these procedures. 
- Additionally, `pytorch` with `fp32` precision is utilized to speed up our self-learning framework
- We use only matrix multiplications in `pytorch` and expect variations in results to be minimal across builds and CUDA versions. 

All reported results use Python 3.6.10 (code is backwards compatible with Python 2) with the following version set:
```
numpy==1.14.2
scipy==0.17.0
torch==1.5.1 # (CUDA Version 9.0.176)
```

## Citation
If you are using FIPP in your work, please cite using the following Bibtex entry:

```
@inproceedings{
sachidananda2021filtered,
title={Filtered Inner Product Projection for Crosslingual Embedding Alignment},
author={Vin Sachidananda and Ziyi Yang and Chenguang Zhu},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=A2gNouoXE7}
}
```

### Acknowledgements 
We thank Professor Goran Glavaš (Universität Mannheim) for correspondence during experiments and for construction of the XLING-BLI datasets and repo. If you use the XLING datasets in your work, please cite Glavaš et. al (ACL 2019):
```
@inproceedings{glavas-etal-2019-properly,
    title = "How to (Properly) Evaluate Cross-Lingual Word Embeddings: On Strong Baselines, Comparative Analyses, and Some Misconceptions",
    author = "Glava{\v{s}}, Goran  and
      Litschko, Robert  and
      Ruder, Sebastian  and
      Vuli{\'c}, Ivan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    doi = "10.18653/v1/P19-1070",
    pages = "710--721"
}
```
