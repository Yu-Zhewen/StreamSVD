# StreamSVD
## Introduction
StreamSVD is an end-to-end toolflow that compresses CNNs with low-rank approximation and deploys the networks onto the FPGA device. The framework considers the SVD low-rank approximation algorithm and the accelerator's architecture simultaneously with a focus on a streaming accelerator architecture suitable for throughput maximisation. 
## Citation
@inproceedings{yu2021streamsvd,
  title={StreamSVD: Low-rank Approximation and Streaming Accelerator Co-design},
  author={Yu, Zhewen and Bouganis, Christos-Savvas},
  booktitle={2021 International Conference on Field-Programmable Technology (ICFPT)},
  pages={1--9},
  year={2021},
  organization={IEEE}
}
## Release
* v0_1
Non-hardware-aware decomposition scheme and decomposition rank selection. SVD iterative quantisation approach.
## Evalution
Download the pretrained models from https://drive.google.com/drive/folders/1O6N5deCjwdcTmHnZxRBCqDSWVDij5z5L?usp=sharing

Run the script
```
python vgg16_svd_optimize.py
```
