import torch
import torch.nn.functional as F
import torch.optim as optim

import torchvision.models as tv_models

import numpy as np
from tqdm import tqdm

from model import VGG, TransformNet
from utility import gram_matrix, total_variance, ImageProcess
from data_loader import data_loader

# path setting
output_img_path = "./image/output/"
style_img_path = "./image/style_img_v2.jpg"
model_save_path = "./model.pth"

# verbose setting
debug = False
verbose_print = False
verbose_batch = 100

# test setting
test_num = 30

# image crop setting
uniform_h = 256
uniform_w = 256

# hyperparmaeter setting
iter_times = 1
dataset_size = 8000
batch_size = 4

style_weight = 1e5
content_weight = 1
totalVariation_weight = 1e-6
learning_rate = 1e-3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# load image to torchTensor
style_img = ImageProcess.read_image(style_img_path, uniform_h, uniform_w).to(device)

# paint images
if verbose_print:
    ImageProcess.paint_image(style_img,"style_image")

# build feature model
vgg16 = tv_models.vgg16(pretrained=True)
# convert into self feature extraction model
vgg16 = VGG(vgg16.features[:23]).to(device).eval() # notify all layers in eval mode
if verbose_print:
    print(vgg16)

# get style image features
style_features = vgg16(style_img)
if verbose_print:
    print("style feature:")
    print([i.shape for i in style_features])

# calculate Gram matrix according to the extracted feature
style_gram = [gram_matrix(i) for i in style_features]
if verbose_print:
    print("style Gram matrix:")
    print([i.shape for i in style_gram])

# instancialize the transformNet
transform_net = TransformNet(32).to(device)

# initilize the optimizer
optimizer = optim.Adam(transform_net.parameters(), lr=learning_rate)
# learning rate scheduler to adjust training
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

transform_net.train() ## Sets the module parameter in training mode.

# try to use multi-GPUs
# if torch.cuda.device_count() > 1:
#   print("Use", torch.cuda.device_count(), "GPUs!")
#   transform_net = torch.nn.DataParallel(transform_net)


# train the model

for epoch in range(iter_times):
    print("Epoch %d" % epoch)

    with tqdm(enumerate(data_loader), total=dataset_size, ncols=40) as pbar: # len(data_loader)
        for batch, (content_imgs, _) in pbar:

            if batch > dataset_size:
                break
            
            scheduler.step()
            optimizer.zero_grad()

            content_imgs = content_imgs.to(device)
            output_imgs = transform_net(content_imgs)
            output_imgs = output_imgs.clamp(-3, 3) # TODO

            if batch == 0 and debug:
                print("style image size:", style_img.size())
                print("content imagess size:", content_imgs.size())
                print("out images size:", output_imgs.size())
            # style_img.size()    == [         1, channels, height, width]
            # content_imgs.size() == [batch_size, channels, height, width]
            # output_imgs.size()  == [batch_size, channels, height, width]

            # extract the feature through vggNet
            content_features = vgg16(content_imgs)
            output_features = vgg16(output_imgs)

            if batch == 0 and debug:
                print("style features size:", style_features[0].size())
                print("content features size:", content_features[0].size())
                print("out features size:", output_features[0].size())
            # style_features.size()     == [4,          1, channels, height, width] **not a Tenor nor a list**
            # content_features.size()   == [4, batch_size, channels, height, width] **not a Tenor nor a list**
            # output_features.size()    == [4, batch_size, channels, height, width] **not a Tenor nor a list**


            # content loss
            content_loss = F.mse_loss(output_features[1], content_features[1]) # TODO [1]

            # total variation loss
            ## reference
            totalVariation_loss = total_variance(output_imgs)

            # style loss
            style_loss = 0
            output_gram = [gram_matrix(x) for x in output_features]
            if batch == 0 and debug:
                print("output gram size:", output_gram[0].size())
            # output_gram.size() == [batch_size, channels, channels]
            # style_gram.size()  == [         1, channels, channels]
            for og, sg in zip(output_gram, style_gram):
                style_loss += F.mse_loss(og, sg.expand_as(og)) # expand the sg to the same size of og which doesn't allocate any memory, since zip will remain the shorter iterator

            # total loss
            loss = content_weight * content_loss + style_weight * style_loss + totalVariation_weight * totalVariation_loss

            loss.backward()
            optimizer.step()

            description = "\nStep %d: style_loss: %.5f content_loss: %.5f totalVariation_loss: %.5f" % (batch, style_weight * style_loss, content_weight * content_loss, totalVariation_weight * totalVariation_loss)
            if batch % verbose_batch == 0:
                # print(description)
                ImageProcess.save_paint_plot(style_img, content_imgs, output_imgs, "%s%d.jpg" % (output_img_path, batch))

            pbar.set_description(description) # there is sth wrong with Pycharm interactive terminal

# save weights of the model
torch.save(transform_net.state_dict(), model_save_path)