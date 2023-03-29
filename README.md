# Tutorials on Computer Vision with PyTorch

This repository gives you a walkthrough of training and testing image classification and semantic segmentation algorithms on custom datasets. Finally, it shows how 
to build and run a simple web interface so that anyone can use it!

## Contents

1. Train image classification model to recognize dogs and cats
2. Train semantic segmentation model to segment skin lesions from dermoscopic images
3. Build a web interface using the image classification model

## Setup

Create a fresh conda environment.

```bash
# clone
git clone https://github.com/hasibzunair/cv-pytorch-tutorials
cd cv-pytorch-tutorials
# create fresh conda environment
conda create -n cvp python=3.8
conda activate cvp
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install gradio
```

## How to run?

You can simply run the notebooks in order. Datasets are available [here](https://github.com/hasibzunair/cv-pytorch-tutorials/releases/tag/v1). For running the 
web interface, you need to copy your image classification model named `model.pth` in the `web_interface` directory. Then simply run `python app.py`

## Related materials

List of related tutorials that could be useful to look at.

* Intro to Deep Learning - https://github.com/hasibzunair/neural-nets-for-babies
* Intro to Python - https://github.com/hasibzunair/ieee18-cv
* Intro to TensorFlow - https://github.com/hasibzunair/ericsson-upskill-tutorials
* Intro to Image Processing and Computer Vision - https://github.com/hasibzunair/ieee18-cv
* Intro to Image Classification with Python and Keras - https://github.com/hasibzunair/whats-image-classifcation-really
* Build 3D image classification models from CT scans https://keras.io/examples/vision/3D_image_classification/

This repo is built using code examples from [hasibzunair/learn-pytorch](https://github.com/hasibzunair/learn-pytorch).

<img src="./media/meme.jpeg" width="300">