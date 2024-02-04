# [Unsupervised Hybrid Network of Transformer and CNN for Blind Hyperspectral and Multispectral Image Fusion](https://ieeexplore.ieee.org/abstract/document/10415455)
This is a transformer-based super-resolution algorithm that fuse hyperspectral and multispectral image in a unsupervised manner in coping with unknown degradation in spectral and spatial domain. 

FeafusFormer performs strong ablity in blind fusion task. The performance comparisons are tested on CAVE data set  with 32x scale factor. 
![Introduce](https://github.com/Caoxuheng/imgs/blob/main/%E5%9B%BE%E7%89%8715.png)
# Flowchart
![Flowchart](https://github.com/Caoxuheng/imgs/blob/main/HIFtool/flowchart_Feafusformer.png)
# Result presentation  
Nonblind fusion results on Pavia, Chikusei and Xiongan datasets.  
![Result](https://github.com/Caoxuheng/imgs/blob/main/HIFtool/result_feafusformer.png)
The reconstructed results can be downloaded from [`here`](https://aistudio.baidu.com/aistudio/datasetdetail/173277).
# Guidance
**None**
# Requirements
## Environment
`Python3.8`  
`torch 1.12`,`torchvision 0.13.0`  
`Numpy`,`Scipy`  
## Datasets
[`CAVE dataset`](https://www1.cs.columbia.edu/CAVE/databases/multispectral/), 
 [`Preprocessed CAVE dataset`](https://aistudio.baidu.com/aistudio/datasetdetail/147509).
# Note
For any questions, feel free to email me at caoxuheng@tongji.edu.com.  
If you find our work useful in your research, please cite our paper ^.^
