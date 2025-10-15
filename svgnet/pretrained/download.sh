"""
Source: https://github.com/HRNet/HRNet-Semantic-Segmentation.git

4.Performance on the COCO-Stuff dataset. The models are trained and tested with the input size of 520x520. 
If multi-scale testing is used, we adopt scales: 0.5,0.75,1.0,1.25,1.5,1.75,2.0 (the same as EncNet, DANet etc.).

model	            OHEM	Multi-scale	Flip	mIoU	

HRNetV2-W48 + OCR    yes    yes         yes     40.6

downlink: https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cocostuff_3965_torch04.pth
"""

#!/bin/bash

# Download HRNet-OCR pretrained model
wget https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cocostuff_3965_torch04.pth

