import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

norm_mean = [0.485, 0.456, 0.406] # convention
norm_std = [0.229, 0.224, 0.225]

# reference pytorch example: https://github.com/pytorch/examples/blob/0.4/fast_neural_style/neural_style/utils.py#L21-L26
def gram_matrix(x):
    """
    :param x: torch tensor
    :return: the gram matrix of x
    """
    (b, c, h, w) = x.size()
    phi = x.view(b, c, h * w)
    phi_T = phi.transpose(1, 2)
    return phi.bmm(phi_T) / (c * h * w) # use batch matrix(vector) inner product

# reference: https://en.wikipedia.org/wiki/Total_variation_denoising
def total_variance(img):
    """
    get the L1 loss of total variance loss
    :param img: tensor of shape(B,C,H,W)
    :return: a scalar, total variance loss
    """
    return torch.sum(torch.abs(img[:,:,1:,:] - img[:,:,:-1,:])) + torch.sum(torch.abs(img[:,:,:,1:] - img[:,:,:,:-1]))

class ImageProcess():

    @staticmethod
    def common_transforms(h=None, w=None):
        """
        an function of common process
        :param h: height
        :param w: width
        :return: transformation function
        """
        norm = transforms.Normalize(mean=norm_mean, std=norm_std)

        t = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.CenterCrop((h, w)),
            transforms.ToTensor(), # squash PIL image in range[0,255] of shape(H,W,C) to a FloatTensor in range[0,1] of shape(C,H,W)
            norm,
        ])

        return t

    @staticmethod
    def preprocess_image(img, h = None, w = None):
        """
        preprocess an image
        :param img: PIL image
        :param h: height
        :param w: width
        :return: FloatTensor with 4 dimension(B, C, H, W)
        """
        norm = transforms.Normalize(mean=norm_mean, std=norm_std)

        if h and w:
            t = ImageProcess.common_transforms(h, w)
        else:
            t = transforms.Compose([
                transforms.ToTensor(),
                norm,
            ])
        return t(img).unsqueeze(0)

    @staticmethod
    def read_image(path, h = None, w = None):
        """
        open an image
        :param path: the path of image
        :return: torch tensor of an image
        """
        img = Image.open(path)
        tensor_img = ImageProcess.preprocess_image(img, h, w)
        return tensor_img

    @staticmethod
    def torchTensor2Image(tensor):
        """
        convert torch tensor range in [0,1] of shape(B,C,H,W) to numpy array range in [0,255] of shape(B,H,W,C)
        :param tensor: torch tensor
        :return: numpy array of the first image in batch
        """
        image = tensor.detach().cpu().numpy()
        image = image * np.array(norm_std).reshape((1, 3, 1, 1)) + np.array(norm_mean).reshape((1, 3, 1, 1))
        image = image.transpose(0, 2, 3, 1) * 255.
        # numpy clip and change type to integer
        image = image.clip(0, 255).astype(np.uint8) # in pytorch clip is clamp
        return image[0]

    @staticmethod
    def paint_image(tensor, title=None):
        """
        paint the image after recover it to numpy array
        :param ts: torch tensor of image
        :return: NULL
        """
        image = ImageProcess.torchTensor2Image(tensor)
        plt.axis('off')
        plt.imshow(image)
        plt.show()
        if title is not None:
            plt.title(title)
        plt.close()

    @staticmethod
    def save_image(tensor, path):
        """
        save the image converted from a tensor format
        :param tensor: torch tensor of the image
        :param path: oriented path to save the image
        """
        img = ImageProcess.torchTensor2Image(tensor)
        Image.fromarray(img).save(path)
        print("Successfully save the final stylized image to:", path)

    @staticmethod
    def save_paint_plot(st, ct, ot, path):
        """
        save three images in one plot
        :param st: style_image in torch tensor
        :param ct: content_image in torch tensor
        :param ot: output_image in torch tensor
        :param path: oriented path to save the image
        """

        st_img = ImageProcess.torchTensor2Image(st)
        ct_img = ImageProcess.torchTensor2Image(ct[0])
        ot_img = ImageProcess.torchTensor2Image(ot[0])
        print(type(st_img))

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=100, figsize=(10, 4))
        ax1.axis('off')
        ax1.set_title('style_image')
        ax1.imshow(st_img)
        ax2.axis('off')
        ax2.set_title('content_image')
        ax2.imshow(ct_img)
        ax3.axis('off')
        ax3.set_title('output_image')
        ax3.imshow(ot_img)
        fig.show()
        fig.savefig(path)
        plt.close()


if __name__ == "__main__":
    from common import output_img_path
    test_img = ImageProcess.read_image("./image/style_img.jpg")
    print("test image size:", test_img.size())
    ImageProcess.paint_image(test_img)
    ImageProcess.save_paint_plot(test_img, test_img, test_img, output_img_path+"test.jpg")