# GCI
The data and code of paper Zero-shot Micro-video Classification with Image Reconstruction in Graph Completion Network.

The dataset used in paper is [KuaiRec](https://kuairec.com/), which can be downloaded by the link.

# 1. Reconstruction of images
```bash
$ cd preprocess
```


## 1.1 Prompt design

Design the prompts to reconstruct the missing images from image modal.
Here are two kinds of prompts, one is a long form of "a micro-vido, keywords are:" and another is tags list directly.
Details in `prompts_long_small.txt`, `prompts_short_small.txt`, `prompts_long_big.txt`, `prompts_short_big.txt`.


## 1.2 Generate images by prompts

We generate images by model [Vis-CPM](https://github.com/OpenBMB/VisCPM?tab=readme-ov-file), or you can choose a suitable model to generate images.


## 1.3 Encode images

Set the vairables of your path of the generated images `images_lp_path` and `images_sp_path` in `encode_images.py`. The generator we use is chinese-clip-vit-base-patch16.
```bash
$ python encode_images.py
```
Or you can use the encodings we preprocess: `image_small_lp_features.txt`, `image_big_lp_features.txt`, `image_small_sp_features.txt`, `image_small_sp_features.txt`, whose images ares generated by different prompts.

## 1.4 Encode CSD
Then get the CSD vectors.
```bash
$ python encode_csd.py
```

# 2. Construct graph with images similarity
```bash
$ cd preprocess
$ python construct_graph.py
```
THe thresholds of similarities are 0.2, 0.4, 0.6 and 0.8. And the graphs can be seen in `preprocess/img_adj/`.


# 3. Training
Set your model name like 'test'.
```bash
$ python train.py --id test
```

Then you can will the results print on the terminal, and the predicted results in `npy_for_pre_recall_f1/` and model weights in `results/`.

More optional arguments are visible in `train.py`.

# Reference

This work has received assistance from the following. Consider citing their works if you find this repo useful.
```
@article{viscpm,
    title={Large Multilingual Models Pivot Zero-Shot Multimodal Learning across Languages}, 
    author={Jinyi Hu and Yuan Yao and Chongyi Wang and Shan Wang and Yinxu Pan and Qianyu Chen and Tianyu Yu and Hanghao Wu and Yue Zhao and Haoye Zhang and Xu Han and Yankai Lin and Jiao Xue and Dahai Li and Zhiyuan Liu and Maosong Sun},
    journal={arXiv preprint arXiv:2308.12038},
    year={2023}
}
```
```
@inproceedings{gao2022kuairec,
  author = {Gao, Chongming and Li, Shijun and Lei, Wenqiang and Chen, Jiawei and Li, Biao and Jiang, Peng and He, Xiangnan and Mao, Jiaxin and Chua, Tat-Seng},
  title = {KuaiRec: A Fully-Observed Dataset and Insights for Evaluating Recommender Systems},
  booktitle = {Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  series = {CIKM '22},
  location = {Atlanta, GA, USA},
  url = {https://doi.org/10.1145/3511808.3557220},
  doi = {10.1145/3511808.3557220},
  numpages = {11},
  year = {2022},
  pages = {540–550}
}
```
