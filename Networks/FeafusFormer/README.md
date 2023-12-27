# Unsupervised Hybrid Network of Transformer and CNN for Blind Hyperspectral and RGB image Fusion
This is a transformer-based super-resolution algorithm that fuse hyperspectral and multispectral image in a unsupervised manner in coping with unknown degradation in spectral and spatial domain. The uHNTC fully exploits the long-range information and multi-level feature of the both MSI and HSI. The HR texture is transfered to the LR-HSI by fusing the residual texture information of different level feature extracted from HR-RGB. For an HSI with size of 16x16x31, the uFT can achieve 32x spatial improvement with very high accuracy incoping with blind degradations. The quantitative results outperform exisiting SOTA algorithms.  
***The paper has been submitted to a journal***  
FeafusFormer performs strong ablity in blind fusion task. The performance comparisons are tested on CAVE data set  with 32x scale factor. 
![Introduce](https://github.com/Caoxuheng/imgs/blob/main/%E5%9B%BE%E7%89%8715.png)
# Flowchart
**None**
# Result presentation
The reconstructed results can be downloaded from [`here`](https://aistudio.baidu.com/aistudio/datasetdetail/173277).
# Guidance
**None**
# Requirements
## Environment
`Python3.8`  
`torch 1.12`,`torchvision 0.13.0`  
`Numpy`,`Scipy`  
*Also, we will create a Paddle version that implements FeafusFormer in AI Studio online for free!*
## Datasets
[`CAVE dataset`](https://www1.cs.columbia.edu/CAVE/databases/multispectral/), 
 [`Preprocessed CAVE dataset`](https://aistudio.baidu.com/aistudio/datasetdetail/147509).
# Note
For any questions, feel free to email me at caoxuheng@tongji.edu.com.  
If you find our work useful in your research, please cite our paper ^.^
