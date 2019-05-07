import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as tv_models

import model
from utility import gram_matrix, ImageProcess

debug = True
iter_times = 1000
uniform_h = 200
uniform_w = 200
output_img_path = "./image/output_img.jpg"
style_img_path = "./image/style_img_v1.jpg"
content_img_path = "./image/content_img.jpg"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# load image to torchTensor
style_img = ImageProcess.read_image(style_img_path).to(device)
print("style image shape:", style_img.shape)

content_img = ImageProcess.read_image(content_img_path).to(device)
print("content image shape:", content_img.shape)

# paint images
ImageProcess.paint_image(style_img,"style_image")
ImageProcess.paint_image(content_img,"content_image")

# build feature model
vgg16 = tv_models.vgg16(pretrained=True)
# convert into self feature extraction model
vgg16 = model.VGG(vgg16.features[:23]).to(device).eval() # notify all layers in eval mode
if debug:
    print(vgg16)

# get features
style_features = vgg16(style_img)
content_features = vgg16(content_img)
if debug:
    print("style feature:")
    print([i.shape for i in style_features])
    print("content feature:")
    print([i.shape for i in content_features])

# calculate Gram matrix according to the extracted feature
style_gram = [gram_matrix(i) for i in style_features]
if debug:
    print("style Gram matrix:")
    print([i.shape for i in style_gram])

# the stage of train the model
initial_white_noise = True
if initial_white_noise:
    # start with whiting noise
    output_img = torch.randn(content_img.size()).to(device)
    output_img.requires_grad_(True)
    style_weight = 1e5

else:
    ## get the copy of input image as input and set its parameters able to be trained
    output_img = content_img.clone().requires_grad_(True)
    style_weight = 1e7

optimizer = optim.LBFGS([output_img])

## set hyperparameter
content_weight = 0.5

## build an list item to be a counter of the closure
it = [0]

## train the model
while it[0] < iter_times:

    def closure():
        optimizer.zero_grad() # TODO
        output_features = vgg16(output_img)

        # summarize the loss between output_img and style_img, content_img
        content_loss = F.mse_loss(input=output_features[2], target=content_features[2])
        style_loss = 0
        output_gram = [gram_matrix(i) for i in output_features]
        for og, sg in zip(output_gram, style_gram):
            style_loss += F.mse_loss(input=og, target=sg)
        # factors of the tradeoff between style_loss and content_loss is hyperparameters
        loss = style_loss * style_weight + content_loss * content_weight

        if it[0] % 20 == 0:
            print("Step %d: style_loss: %.5f content_loss: %.5f" % (it[0], style_loss, content_loss))
        if it[0] % 100 == 0:
            ImageProcess.paint_image(output_img, title='Output Image')

        # calculate gradient through backward
        loss.backward()
        it[0] += 1

        return loss

    # LBFGS optimizer to update parameters needs a closure that reevaluates the model and returns the loss
    optimizer.step(closure)

ImageProcess.paint_image(output_img, title='Output Image')
ImageProcess.save_image(output_img, output_img_path)

