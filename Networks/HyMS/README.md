![TITLE](https://github.com/Caoxuheng/imgs/blob/main/OL.png)
# [Hyperspectral Image Super-resolution via Multi-stage Scheme without Employing Spatial Degradation](https://doi.org/10.1364/OL.473020)
This is a Model-based algorithm for MSI-HSI fusion without employing spatial degradation. HyMS achieves fusion super-resolution in a short time with SOTA performance.  
An online version based on AI-Studio is released: >>>> Online Version[*(CAVE*](https://aistudio.baidu.com/aistudio/projectdetail/4418051)/ [*Chikusei)*](https://aistudio.baidu.com/aistudio/projectdetail/5884199)<<<<
# Note
GPU-based HyMS may makes mistakes result from a wrong package config. The CPU-based one is the stable version.
# Flowchart
![Flowchart](https://github.com/Caoxuheng/imgs/blob/main/ol2.png)  
# Abstract
Recently, it has become popular to obtain a high spatial resolution hyperspectral image (HR-HSI) by fusing a low spatial resolution hyperspectral image (LR-HSI) with a high spatial resolution RGB image (HR-RGB). Existing HSI super-resolution methods are designed based on a known spatial degeneration. In practice, it is difficult to obtain correct spatial degradation, which restricts the performance of existing methods. Therefore, we propose a multi-stage scheme without employing the spatial degradation model. The multi-stage scheme consists of three stages: initialization, modification, and refinement. According to the angle similarity between the HR-RGB pixel and LR-HSI spectra, we first initialize a spectrum for each HR-RGB pixel. Then, we propose a polynomial function to modify the initialized spectrum so that the RGB color values of the modified spectrum are the same as the HR-RGB. Finally, the modified HR-HSI is refined by a proposed optimization model, in which a novel, to the best of our knowledge, spectral-spatial total variation (SSTV) regularizer is investigated to keep the spectral and spatial structure of the reconstructed HR-HSI. The experimental results on two public datasets and our real-world images demonstrate our method outperforms eight state-of-the-art existing methods in terms of both reconstruction accuracy and computational efficiency.  
# Result presentation  
![Simulate](https://github.com/Caoxuheng/imgs/blob/main/ol23.png)
![Real](https://github.com/Caoxuheng/imgs/blob/main/ol2real.png)
# Guidance  
Add your dataset path in `config.py`  
Run `main.py`
## Running interface  
![Introduce](https://github.com/Caoxuheng/imgs/blob/main/Rins.png)
# Requirements  
## Environment  
`Python 3.8`  
`Numba`, `Cupy`  
`Numpy`, `Scipy`, `Opencv`
## Datasets
[`CAVE dataset`](https://www1.cs.columbia.edu/CAVE/databases/multispectral/), 
 [`Preprocessed CAVE dataset`](https://aistudio.baidu.com/aistudio/datasetdetail/147509).
# Contact
For any questions, feel free to email Xuheng Cao(📧caoxuhengcn@gmail.com).  
If you find our work useful in your research, please cite our paper 🙂
```python  
@article{article,
author = {Cao, Xuheng and Lian, Yusheng and Liu, Zilong and Zhou, Han and Bin, Wang and Zhang, Wan and Huang, Beiqing},
year = {2022},
month = {09},
pages = {5184-5187},
title = {Hyperspectral Image Super-resolution via Multi-stage Scheme without Employing Spatial Degradation},
journal = {Optics Letters},
doi = {10.1364/OL.473020}
}
