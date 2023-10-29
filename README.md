# COOLANT
Code for paper "Cross-modal Contrastive Learning for Multimodal Fake News Detection".

# Dependency
+ python 3.5+
+ pytorch 1.0+
+ transformers 4.28.0

## Dataset
We conduct experiments on two benchmark datasets Twitter and Weibo. In experiments, we keep the same data split scheme as the benchmark. Specifically, for the Twitter dataset, we followed the work of ([Chen et al., 2022](https://github.com/cyxanna/CAFE)), and for the Weibo dataset, we followed the work of ([Wang et al., 2022](https://github.com/yaqingwang/EANN-KDD18)).


## Training
To train the COOLANT:
```shell script
python weibo/weibo.py 
python twitter/twitter.py 
```

## Citation
If you use source codes included in this toolkit in your work, please cite the following paper. The bibtex are listed below:
```shell script
@inproceedings{10.1145/3581783.3613850,
author = {Wang, Longzheng and Zhang, Chuang and Xu, Hongbo and Xu, Yongxiu and Xu, Xiaohan and Wang, Siqi},
title = {Cross-Modal Contrastive Learning for Multimodal Fake News Detection},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3613850},
doi = {10.1145/3581783.3613850},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {5696–5704},
numpages = {9},
keywords = {social media, multimodal fusion, fake news detection, contrastive learning},
location = {Ottawa ON, Canada},
series = {MM '23}
}
```