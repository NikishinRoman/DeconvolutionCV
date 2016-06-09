# DeconvolutionCV
Deconvolution algorithms toolbox based on **OpenCV**.  

The main goal of this tool is to implement some well-known deconvolution algorithms using **OpenCV*  
Algorithms available :  
- Wiener deconvolution  
- Richardson-Lucy deconvolution
- Tikhonov-Regularized Richardson-Lucy deconvolution

Soon available :  
- FISTA
- Total Variation regularized Richardson-Lucy deconvolution
- Additive versions of Richardson-Lucy deconvolution
- Partial Differential Equations based iterative deconvolution
- ADMM

## How to use

The "main.cpp" is given as an example of use.

## Limitations
Tested on grayscale square images with gaussian blurring kernels and gaussian noise.

### Performances
The algorithms are implemented using OpenCV functionnalities. Some operations are carried out "pixel by pixel"  
The implementations currently focus on readability  

## Dependencies

- [OpenCV](http://opencv.org/) (2.4.13 recommended)

## References and Publications
