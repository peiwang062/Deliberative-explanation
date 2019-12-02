# Deliberative-explanation

PyTorch implementation of [Deliberative Explanations: visualizing network insecurities](https://papers.nips.cc/paper/8418-deliberative-explanations-visualizing-network-insecurities) (NeurIPS 2019). The implementation is partly based on the PyTorch implementation of [GradCAM](https://github.com/jacobgil/pytorch-grad-cam), thanks.

## Requirements

1. The project was implemented and tested in Python 3.5 and Pytorch 0.4. The higher versions should work after minor modification.
2. Other common modules like numpy, pandas and seaborn for visualization.
3. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.


## Datasets

[CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [ADE20K](http://sceneparsing.csail.mit.edu/) are used. Please organize them as below after download,


```
cub200
|_ CUB_200_2011
  |_ attributes
  |_ images
  |_ parts
  |_ train_test_split.txt
  |_ ...
```

```
ade
|_ ADEChallengeData2016
  |_ annotations
  |_ images
  |_ objectInfo 150.txt
  |_ sceneCategories.txt
```

## Implementation details

### data preparation

build train/validation/test sets as list of below. Refer to CUB200_gt_te.txt for format details

```
image_path class index
```

compute similarity for parts on CUB200 and objects on ADE20K,

```
create_similarityMatrix_cls_part_cub.py
create_similarityMatrix_cls_object_ade.py
```

prepare attribute location data on CUB200

```
get_gt_partLocs.py
```

### training

Two types of models need to be trained, the standard CNN classifier and [Hardness predictor](http://openaccess.thecvf.com/content_ECCV_2018/html/Pei_Wang_Towards_Realistic_Predictors_ECCV_2018_paper.html). Three most popular architectures were tested. For reproducing each result individually, we separately wrote the code for each experiment. For the classifier,
```
./cub200/train_cub_alexnet.py
./cub200/train_cub_vgg.py
./cub200/train_cub_res.py
./ade/train_ade_alexnet.py
./ade/train_ade_vgg.py
./ade/train_ade_res.py
```
for the hardness predictor,
```
./cub200/train_hp_cub_alexnet.py
./cub200/train_hp_cub_vgg.py
./cub200/train_hp_cub_res.py
./ade/train_hp_ade_alexnet.py
./ade/train_hp_ade_vgg.py
./ade/train_hp_ade_res.py
```

### visualization

Three types of attribution methods are compared, baseline [gradient based](https://arxiv.org/pdf/1312.6034.pdf), state-of-the-art [integrated gradient (IG) based](https://dl.acm.org/citation.cfm?id=3306024) and ours (gradient-Hessian(2ndG)).


1. for comparison of different scores,
```
insecurity_cs_cub_vgg.py
insecurity_entropy_cub_vgg.py
insecurity_hp_cub_vgg.py
insecurity_cs_ade_vgg.py
insecurity_entropy_ade_vgg.py
insecurity_hp_ade_vgg.py
```

2. for comparison of different attribution maps,
```
insecurity_hp_cub_vgg.py
insecurity_hp_cub_vgg_IG.py
insecurity_hp_cub_vgg_2ndG.py
insecurity_hp_ade_vgg.py
insecurity_hp_ade_vgg_IG.py
insecurity_hp_ade_vgg_2ndG.py
```

3. for comparison of different architectures,

```
insecurity_hp_cub_alex.py
insecurity_hp_cub_vgg.py
insecurity_hp_cub_res.py
insecurity_hp_ade_alex.py
insecurity_hp_ade_vgg.py
insecurity_hp_ade_res.py
```


### pretrained models

The pre-trained models for all experiments are availiable. [Site](https://drive.google.com/drive/folders/1GoTyEP5EGS_gkkGTFn_7ooI0ZdCgyPGx?usp=sharing).



## References

[1] Simonyan, K., Vedaldi, A. and Zisserman, A., 2013. Deep inside convolutional networks: Visualising image classification models and saliency maps. arXiv preprint arXiv:1312.6034.

[2] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra.  Grad-cam:  Visual explanations from deep networks via gradient-based localization.  In Proceedings of the IEEE International Conference on Computer Vision, pages 618–626, 2017.

[3] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 3319–3328. JMLR. org,4662017.

[4] Pei Wang and Nuno Vasconcelos. Towards realistic predictors. In The European Conference on Computer Vision, 2018.

[5] Welinder, P., Branson, S., Mita, T., Wah, C., Schroff, F., Belongie, S. and Perona, P., 2010. Caltech-UCSD birds 200.

[6] Bolei  Zhou,  Hang  Zhao,  Xavier  Puig,  Sanja  Fidler,  Adela  Barriuso,  and  Antonio  Torralba.   Scene parsing through ade20k dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017.
