# Improved-ITV
The official implementation of our paper [Improving Interpretable Embeddings for Ad-hoc Video Search with Generative Captions and Multi-word Concept Bank ](https://arxiv.org/abs/2404.06173) accepted in ICMR2024.
## Environment

We used Anaconda to setup a workspace with PyTorch 1.8. Run the following script to install the required packages.

```shell
conda create -n IITV python==3.8 -y
conda activate IITV
git clone https://github.com/nikkiwoo-gh/Improved-ITV.git
cd Improved-ITV
pip install -r requirements.txt
```

### Stanford coreNLP server for concept bank construction
```shell
./do_install_StanfordCoreNLIP.sh
```

## Downloads

### Data

[WebVid-genCap7M dataset](https://drive.google.com/file/d/18Dh20_ZlSGJ_XAFM2P5dpd3qSIR-vSBJ/view)

to be released

### Checkpoints

to be released

## Usages


### 1. build bag of word vocabulary and concept bank
```shell
./do_get_vocab_and_concept.sh $collection
```

e.g.,
```shell
./do_get_vocab_and_concept.sh tgif-msrvtt10k-VATEX
```

### 2. prepare the data
to be released

### 3. train the model from pre-trained checkpoint
```shell
./do_train_from_pretrain.sh
```

### 4. Inference on TRECVid datasets
```shell
./do_predition_iacc.3.sh
./do_predition_v3c1.sh
./do_predition_v3c2.sh
```

### 5. Evalution
Remember to set the score_file correctly to your own path.
```shell
cd tv-avs-eval/
do_eval_iacc.3.sh
do_eval_v3c1.sh
do_eval_v3c2.sh
```

## Citation

```latex
@inproceedings{ICMR2024_WU_improvedITV,
author = {Wu, Jiaxin and Ngo, Chong-Wah and Chan, Wing-Kwong},
title = {Improving Interpretable Embeddings for Ad-hoc Video Search with Generative Captions and Multi-word Concept Bank},
year = {2024},
booktitle = {The Annual ACM International Conference on Multimedia Retrieval},
pages = {1-10},
}
```



## Contact
jiaxin.wu@my.cityu.edu.hk
