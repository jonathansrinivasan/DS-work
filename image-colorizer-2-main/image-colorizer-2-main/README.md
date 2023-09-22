# Image (Re)Colorization v2: Transformer Edition
By Jonathan Srinivasan and Dakota Wilson                                    
DS-5899: Special Topics in Data Science - Transformers in Theory and Practice  

## Overview

**Problem:** 

Color photography was first created in the 1890's however was very expensive and difficult to achieve. It was not until the 1970s, 80 years later, when the accessability and use of color photography became widespread. This leaves us with huge amounts of photographs that are stuck in black and white, but by using modern machine learning methods, we hope to be able to add color to these photos that we once thought would be stuck in black and white forever.

We know that this has been done with CNNs in the past, however with the advent of transformer models, this is becoming easier and more accurate than ever. In this presentation, we hope to compare and contrast CNNs and transformers in their ability to colorize old photos.

What performs better in colorizing black and white photos? CNNs or Google ColTran? 

*Note: How do we calculate image coloring performace? Loss function? Eye test? Speed? Clarity?*

Let's use a CNN that we previously constructed for another course and compare it to Google's color transformer (ColTran)

## Data

- Dataset of sports photos (408.5 MB)
- Includes photos of over 100 sports, with 10,000+ entries
- https://www.kaggle.com/datasets/gpiosenka/sports-classification
- Our original hope was to fine-tune our model using these sports photos in attempts to make a model that was decent at predicting humans (skin color, clothing etc.).

## Methodology

### Convolutional Neural Network
- Built a model that 1) decomposes a color picture and recolorizes it and 2) colorizes black and white photos
- Improved upon a pretrained model (ex: switched loss function from MSE to LPIPS)
- 18 layers that reduce and convolute the photo, and then deconvolutes it back to a full size image

![Capture1](https://user-images.githubusercontent.com/48261978/231642125-66ce5354-792b-4f5d-a11e-699f3e4beacd.PNG)

- The A channel represents the color from green to red, with negative values indicating green and positive values indicating red. The B channel represents the color from blue to yellow, with negative values indicating blue and positive values indicating yellow. The main advantage of using AB channels is that it helps separates the color information from the brightness information.

- LPIPS: Learned Perceptual Image Patch Similarity - a metric created to try and replicate human vision. This loss function takes in two images and tries to define how different they are in the sense of human vision, as opposed to solely relying on pixel-wide differences. The LPIPS loss is computed as the cosine distance between the feature representations of the source and target images at each layer, normalized by the magnitude of the feature vectors. In our CNN project, we had taken a pre-existing model and outfitted it with this new loss to improve upon original results. This is the metric that we will be using in this project for evaluating the accuracy of our outputs against each other. 

### Google ColTran

- Why use a transformer for this when a CNN architecture has proven to be a powerful learner that is able to do a decent job of recolorizing an image?

Attention can prove to be incredibly helpful in the completion of this task; by increasing the receptive field that the model is able to "see" at each step, we expect our model to be able to provide more accurate results that take in the context of the whole photo. This context is something that a CNN is simply not able to find by taking traditional strides across an image.

![ColTran Architecture](https://github.com/dakotalw/image-colorizer-2/blob/main/coltran_architecture.png)

ColTran is comprised of three parts: an autoregressive colorizer that is based on a conditional variant of the Axial Transformer for low resolution coarse colorization. This outputs colors for the input grayscale into a 64 x 64 image, but does not combine them together. A color upsampler, which takes the color output from the first part, adds the grayscale parts (scaled down to 64x64) from the original image, and tries to improve the color based on the new complete image, all while still working with a 64x64 image. Finally, the third part takes the full color 64 x 64 image and proceeds to scale it back up to a 256 x 256 image.

Following the work of Axial Attention in Multidimensional Transformers, (Ho et al., 2019b) a standard qkv self-attention was employed within rows and columns of an image. By alternating row- and column self-attention they effectively allowed global exchange of information between all pixel positions. Row/column attention layers are the core components of the modeland they are used in the autoregressive colorizer, the spatial upsampler and the color upsampler.

ColTran was trained using the ImageNet database photos. Each of the three parts (colorizer, color upsampler, image upsampler) were trained with a batch-size of 224, 768 and 32 for 600K, 450K and 300K steps respectively. 4 axial attention blocks were used in each component of the architecture, with a hidden size of 512 and 4 heads.

ColTran has also been optimized to work as efficiently as possible with computational resouces; for instance the first step is only predicitng a single channel of "course" color, which takes less effort to predict as opposed to 2 channels that are traditionally seen in the L\*a\*b\* color space. In addition to this, downsampling the image to 64 x 64 also decreases the computational requirements for both step 1 and 2 of the model. Comparatively, in our CNN, the smallest layer that an image is downsampled to is 128 x 128.


### Testing

![outputs](https://github.com/dakotalw/image-colorizer-2/blob/main/grid_of_images.png)

In our testing, the CNN output images actually had a slightly lower LPIPS score than ColTran. (0.2543 for ColTran, 0.2317 for CNN) There could be several reasons for this, but it is more than likely due to the precision in the color of ColTran. We can observe that although the colors may not be as accurate, ColTran does a much better job of being consistent with its color choices, compared to the patchy look that our CNN gives. If an object is supposed to be blue, and a model predicts it as half blue and half red, the LPIPS would be lower for that prediction than if it had predicted completely red. However, when viewing one of these images, the all red image would look more natural to the human eye. This demonstrates both the weaknesss of LPIPS as well as how difficult it can be to create a loss function that accurately represents human vision.

### Limitations

Unfortunately, a common issue experienced across all computer vision tasks regardless of architecture is the high computational cost of processing images. This issue only worsens with higher quality images, which are becoming more and more prevalent as time goes on. For both our model and ColTran, images needed to be resized to 256 x 256 pixels for computational efficiency. This increase in time and computation experienced with larger images may be manageable when we are predicting across images, but training a model with larger images will quickly add orders of magnitude to computation time and costs.

This is not the worst problem to have, however! Modern transformer architectures have been developed to upscale images from lower pixel dimensions into higher quality images. This is even applied in ColTran; in the third and final step of predicting, the model takes a 64 x 64 image and scales it all the way up to 256 x 256.

To revisit the loss function - as mentioned, LPIPS is not perfect and although it may be a better estimator of image similarity than something like MSE, it still has its issues. There have been new loss functions developed, both at a per-pixel level and at a perceptual level to try and alleviate the difficulty in estimating image similarity numerically, that may be better fit for estimating the accuracy of these models. This is entirely up to the discretion of the engineer training a model, and depending on the CV task at hand, loss functions may differ. In the case of ColTran, a per-pixel log-liklihoood function was used to estimate color similarity. 

## Inferencing Using Black and White Photos

![bw-outputs](https://github.com/dakotalw/image-colorizer-2/blob/main/bw-output.png)

## Next Steps

The objective of this project is to show the advancements in image colorization. We initally improved our CNN model by replacing the MSE loss function in our with the Learned Perceptual Image Patch Similarity (LPIPS) loss function. This change resulted in a significant boost in model performance. However, more recently, other loss functions have emerged that offer even greater accuracy than LPIPS. In addition to all of this, the increase in computational avaliability and efficiency will allow us in the future to be able to create similar models that work at much higher resolutions than 256 x 256.

In addition to this, in terms of transformers, stable diffusion has taken over with anything to do with images. Stable diffusion allows us to colorize and modify images based on specific prompts. This technique enables us to create prompts that allow us to colorize black and white photos in a precise and efficient manner.  

### Stable Diffusion

- Neural Network called ControlNet
- **Canny Edge Detection Algorithm:** applied to the image to extract edges and convert it into a binary image. The process involves smoothing the image using a Gaussian filter, computing the gradient magnitude and direction at each pixel, suppressing non-maximum edges, and hysteresis thresholding to remove weak edges.        
- **Memory-Efficient Attention:** uses local attention, where the input sequence is split into chunks and self attention is applied to the chunks within a certain distance of the current position. This reduces the number of computations and memory required to compute the self-attention scores, and allows the model to handle longer input sequences.  

```python
enable_xformers_memory_efficient_attention()
```
Original:              
![kideating](https://user-images.githubusercontent.com/48261978/231781731-9383cffa-efea-4fd6-98cf-80bb63160746.jpg)

After Canny Edge Detection:                                           
![kid2](https://user-images.githubusercontent.com/48261978/231782443-bdd38f06-dde0-488b-b7be-4fbef1ca6cde.png)


Prompt: ["add color to this image, portrait, high quality"]                                              
Negative Prompt: ["black and white, cartoon, low quality"]                                

![kid1](https://user-images.githubusercontent.com/48261978/231782496-4379b696-81be-411c-b4b4-3ece884eea3c.png)

Using stable diffusion on sports photos:                                        

![sports](https://user-images.githubusercontent.com/48261978/231811822-2562b3fe-3097-4dd8-8772-eef246af53e2.jpg)

Prompt: ["add color to this image, portrait, high quality, two soccer opponents shaking hands, they are in a soccer field, Pelé & Giacinto Facchetti shaking hands, a lot of fans in the background"]                      


![soccer](https://user-images.githubusercontent.com/48261978/231812199-6b503fa3-f5cf-4591-8394-2a89aea90e75.png)


## Code Demo (ColTran):

https://colab.research.google.com/drive/1VVc3AFAo2HRHFfrnam2v-hEFKut9oniW#scrollTo=toLwtBQGwCgY

## Resources
CNN:
- https://github.com/richzhang/colorization
- https://www.kaggle.com/code/theblackmamba31/autoencoder-grayscale-to-color-image
- https://lukemelas.github.io/image-colorization.html (Baseline model)

ColTran:
- https://arxiv.org/pdf/2102.04432.pdf (Colorization Transformer - Google Brain Team)
- https://arxiv.org/abs/1912.12180 (Axial Attention in Multidimensional Transformers - Ho et al. 2019)
- https://github.com/google-research/google-research/tree/master/coltran

Stable Diffusion:                        
- https://huggingface.co/lllyasviel/sd-controlnet-canny
