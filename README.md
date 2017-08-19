# Hard Negatives and Selective Dropout for Adversarial Images

Or my first venture into understanding adversarial learning.

Building off of code developed by Tabacof & Valle in [ArXiv link](http://arxiv.org/abs/1510.05328)

Download the novice paper here ![Paper](written_final_report.pdf).

Or view below:

![Page 0001](images/0001.jpg)
![Page 0002](images/0002.jpg)
![Page 0003](images/0003.jpg)
![Page 0004](images/0004.jpg)
![Page 0005](images/0005.jpg)
![Page 0006](images/0006.jpg)
![Page 0007](images/0007.jpg)
![Page 0008](images/0008.jpg)
![Page 0009](images/0009.jpg)
![Page 0010](images/0010.jpg)
![Page 0011](images/0011.jpg)
![Page 0012](images/0012.jpg)
![Page 0013](images/0013.jpg)
![Page 0014](images/0014.jpg)
![Page 0015](images/0015.jpg)
![Page 0016](images/0016.jpg)
![Page 0017](images/0017.jpg)
![Page 0018](images/0018.jpg)
![Page 0019](images/0019.jpg)
![Page 0020](images/0020.jpg)
![Page 0021](images/0021.jpg)
![Page 0022](images/0022.jpg)
![Page 0023](images/0023.jpg)
![Page 0024](images/0024.jpg)
![Page 0025](images/0025.jpg)



## Requires

[Torch7](https://github.com/torch/torch7)

GFortran with BLAS

iTorch

## L-BFGS-B

The adversarial image optimization problem requires the box-constraints so that the distortions won't make the image go outside the pixel space (RGB = [0, 255]).

For this we use the Fortran library L-BFGS-B written by Nocedal, the author of the algorithm. To compile the library do the following:
```bash
cd lbfgsb
make lib
```
This library is as fast the Torch7 Optim's LBFGS (wihout bound constraints).

## MNIST

For MNIST, the code will train the classifier from scratch. A logistic regression should achieve about 7.5% error, and a standard convolutional network 1%. You need to download the dataset:

```bash
cd mnist
th download.lua
```

## Adversarial images

Now you can create adversarial images using:
```bash
th adversarial.lua -i image.png
```

Options:
```
-i: image file
-cuda: use GPU support (must have CUDA installed on your computer - test this with require 'cutorch')
-gpu: GPU device number
-ub: unbounded optimization (allow the distortion to go outside the pixel space)
-mc: probe the space around the adversarial image using white noise (default is Gaussian)
-hist: use nonparametric noise instead of Gaussian ("histogram")
-orig: probe the space around the original image instead
-numbermc: number of probes
-mnist: use MNIST instead of ImageNet dataset
-conv: use convolutional network with MNIST (instead of logistic regression)
-itorch: iTorch plotting
-seed: random seed
```

The resulting images and the distortions will be created on the same folder of the image.

## Selective Dropout

An extremely naive version of selective dropout is implemented along with a number of visualization techniques are implemented in the ipynb

A huge thanks to Professor Dobkin at Princeton University for his advice throughout the project.
