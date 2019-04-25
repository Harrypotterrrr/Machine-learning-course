# Image Stylization

**Note:** Please redirect to [stylization repo](https://github.com/Harrypotterrrr/Image-stylization)

Image stylization is to combine a real image as a base content with the perceptual context of a style image to a final stylized image.
This project is an implementations of [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576). 

### Introduction

The core idea of it is to interpret the gram matrix as style loss to express the perceptual and cognitive mode of style image while leverage the tradeoff between the loss from the origin content image.

The trainable parameters contains only the pixel value of final output image.

### Result

<table>
    <tr>
        <td><img src=https://github.com/Harrypotterrrr/Image-stylization/blob/master/image/myplot1.png border=0></td>
        <td><img src=https://github.com/Harrypotterrrr/Image-stylization/blob/master/image/myplot2.png border=0></td>
        <td><img src=https://github.com/Harrypotterrrr/Image-stylization/blob/master/image/myplot3.png border=0></td>
    </tr>
</table>

### Prerequisites

| **Package** | **version** |
|-------------|-------------|
| python      | 3.5.2       |
| torch       | 1.0.1       |
| torchvision | 0.2.2       |

### Getting Started

#### Train

`python3 train.py`

argument parser is coming soon.. 

### Reference

- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [Image Style Transfer Using Convolutional Neural Networks](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
- [Instance Normalization, The Missing Ingredient for Fast Stylization](https://arxiv.org/pdf/1607.08022)
- [Meta Networks for Neural Style Transfer](http://openaccess.thecvf.com/content_cvpr_2018/papers/Shen_Neural_Style_Transfer_CVPR_2018_paper.pdf)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)


