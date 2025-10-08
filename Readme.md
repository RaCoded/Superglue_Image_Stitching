# Superglue Image Stitching

This project is an automated image stitching pipeline using SuperGlue for feature matching and homography-based blending in Python. It is inspired by the original work Image-Stitching-OpenCV by linrl3 (), which used SIFT, KNN, and RANSAC. Here, the process has been fully automated to remove manual offsets and improve usability.


The goal is to stitch two or more overlapping images into a seamless panorama. SuperGlue extracts and matches local features between images, computes the homography matrix for alignment, and blends the images using an automatic mask for smooth transitions

## Features
- Fully automated feature matching with SuperGlue.
- Homography estimation using RANSAC.
- Automatic mask generation for smooth blending.
- RGB blending for natural transitions.
- No manual offset needed.

## Dependencies
- Python 3.8+
- OpenCV 4.x
- PyTorch
- Matplotlib
- SuperGlue pretrained model files (included in utils/models/SuperGluePretrainedNetwork)

## Usage
python main.py

## Matching 
![matching](https://github.com/linrl3/Image-Stitching-OpenCV/blob/master/images/matching.jpg)

## Output image
![pano](https://github.com/linrl3/Image-Stitching-OpenCV/blob/master/images/panorama.jpg)