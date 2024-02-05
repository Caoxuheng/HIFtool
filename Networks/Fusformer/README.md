# Fusformer
Code for the paper: "Fusformer: A Transformer-based Fusion Approach for Hyperspectral Image Super-resolution"

**Plateform** 

Python 3.8.5 + Pytorch 1.7.1 

----------------------------------------------------------------------------
**Data**
* demo_cave.h5: A testing image from the CAVE dataset, containing "GT" ( 512x512x31), "RGB" (128x128x3) and "LRHSI" (128x128x31).
* demo_cave_patches_h5: If your GPU memory is too small to run with the whole testing image, we cut the testing image into 64 patches with the size 64x64x31 of GT, 63x64x3 of RGB, and 16x16x31 of LRHSI, respectively.

Demo data download link: https://github.com/J-FHu/Fusformer/releases/tag/Pytorch

Complete training and test data download link: 1.https://drive.google.com/drive/folders/11EsA_qLtf306RxEReFSA_iKltdpn-IJT?usp=sharing  **or** 2.https://pan.baidu.com/s/1QOClak9iHHhC8Mo1mpVcRQ Password：kjra  
**Citation**
@ARTICLE{9841513,
  author={Hu, Jin-Fan and Huang, Ting-Zhu and Deng, Liang-Jian and Dou, Hong-Xia and Hong, Danfeng and Vivone, Gemine},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Fusformer: A Transformer-Based Fusion Network for Hyperspectral Image Super-Resolution}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  keywords={Transformers;Hyperspectral imaging;Task analysis;Computer architecture;Training;Superresolution;Feature extraction;Hyperspectral image super-resolution (HISR);image fusion;remote sensing;Transformer},
  doi={10.1109/LGRS.2022.3194257}}
