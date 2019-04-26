# Quantifying and Alleviating the Language Prior Problem in Visual Question Answering.
This the official implementation for our SIGIR 19 paper. We tested our idea on three different works. Since the code is easy to implement and expand, we only provide the ```score regularization``` based on the ```Strong baseline``` [paper](https://arxiv.org/abs/1704.03162). This repository is built upon the [code](https://github.com/Cyanogenoid/vqa-counting.git) provided by @Yan Zhang. Thanks for him generously sharing the code.

### Get repo: 
``` git clone https://github.com/guoyang9/vqa-prior.git --recursive```

## Prerequisites

	* python==3.6.8
	* numpy==1.16.2
	* pytorch==1.0.1
	* torchvision==0.2.2
	* nltk==3.4
	* bcolz==1.2.1
	* tqdm==4.31.1

## Dataset
First of all, make all the data in the right position according to the config.py.

* The VQA dataset can be downloaded at the 
[official website](https://visualqa.org/download.html). This repository only implemented the model on the VQA 1.0 dataset. 
* The pre-trained Glove features can be found on [glove website](https://nlp.stanford.edu/projects/glove/).


## Preprocessing
1. Preprocess grid-based image features: preprocess the image feature, including extracting pre-trained image faetures.
	```
	python preprocess/preprocess-images.py
	```
1. Preprocess the vocabulary: filtering top 3000 answers.
	```
	python preprocess/preprocess-vocab.py
	```
1. Preprocess question type: counting answers under each question type.
	```
	python preprocess/preprocess-qt.py
	```

## Model Training
```
python main.py --name=vqa-prior --gpu=0
```
## Model Test only
```
python main.py --test --name=vqa-prior --gpu=0
```

## Citation
If you plan to use this code as part of your published research, we'd appreciate it if you could cite our paper:
```
@Inproceedings{prior,
  author    = {Yangyang Guo, Zhiyong Cheng, Liqiang Nie, Yibing Liu, Yinglong Wang and Mohan Kankanhalli},
  title     = {Quantifying and Alleviating the Language Prior Problem in Visual Question Answering},
  booktitle = {SIGIR},
  year      = {2019},
}
```
