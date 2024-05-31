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

### Pretraining Dataset

[WebVid-genCap7M dataset](https://drive.google.com/file/d/18Dh20_ZlSGJ_XAFM2P5dpd3qSIR-vSBJ/view)

### Concept bank for tgif-msrvtt10k-VATEX

[concept_phrase.zip](https://portland-my.sharepoint.com/:u:/g/personal/jiaxinwu9-c_my_cityu_edu_hk/EZZ4l3eo675DmXh0afsPRF8B6rIp8V02WBOJKtv8tPkaxw?e=2iq38f)

### Video-level concept annoation for tgif-msrvtt10k-VATEX

 [tgif-msrvtt10k-VATEX-videl_level_concept annotation](https://portland-my.sharepoint.com/:u:/g/personal/jiaxinwu9-c_my_cityu_edu_hk/EbKBY5x-zqNIhvpgIabA20IBIqhlFd8Yu6rQNEXkNkhynw?e=ihU9je) 

### Model Checkpoints
[Improved_ITV model pretrained on_WebVid-genCap7M](https://drive.google.com/file/d/1hif1yS_8H4ap-FtZqdQx4Ug_Joh8CRtM/view?usp=sharing)

[Improved_ITV model finetuned on tgif-msrvtt10k-VATEX](https://drive.google.com/file/d/1fB-U6XrCFfj_n23oB6kvCtO7nw8JQsh_/view?usp=sharing)

## Usages


### 1. build bag of word vocabulary and concept bank
```shell
./do_get_vocab_and_concept.sh $collection
```

e.g.,
```shell
./do_get_vocab_and_concept.sh tgif-msrvtt10k-VATEX 
```
or download from [concept_phrase.zip](https://portland-my.sharepoint.com/:u:/g/personal/jiaxinwu9-c_my_cityu_edu_hk/EZZ4l3eo675DmXh0afsPRF8B6rIp8V02WBOJKtv8tPkaxw?e=2iq38f), and unzip to the folder $rootpath/tgif-msrvtt10k-VATEX/TextData/
### 2. prepare the data
build up video-level concept annotation (script to be released), or download from [here](https://portland-my.sharepoint.com/:u:/g/personal/jiaxinwu9-c_my_cityu_edu_hk/EbKBY5x-zqNIhvpgIabA20IBIqhlFd8Yu6rQNEXkNkhynw?e=ihU9je)   


### 3. train the Improved ITV model

#### 3.1 train from pre-trained checkpoint
```shell
./do_train_from_pretrain.sh
```
#### 3.2 train without pre-training
```shell
./do_train.sh
```
### 4. Inference on TRECVid datasets
```shell
./do_prediction_iacc.3.sh
./do_prediction_v3c1.sh
./do_prediction_v3c2.sh
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
