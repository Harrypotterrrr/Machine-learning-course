import torch
import torchvision

from utility import ImageProcess
from common import *


# load transformation function
data_transform = ImageProcess.common_transforms(uniform_h, uniform_w)


dataset = torchvision.datasets.ImageFolder('/your/path/to/coco/dataset/', transform=data_transform)
# dataset = torch.utils.data.Subset(dataset, [i for i in range(100)])
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

if verbose_print:
    print("dataset description:\n", dataset)

if __name__ == "__main__":
    print(type(dataset))
