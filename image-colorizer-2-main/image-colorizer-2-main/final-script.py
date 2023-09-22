import numpy as np
import matplotlib.pyplot as plt

# For conversion
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.color as convert
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# For utilities
import os, shutil, time
from PIL import Image
import sys

#define model
class ColorizationNet(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet18(num_classes=365) 
    # Change first conv layer to accept single-channel (grayscale) input
    #resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    midlevel_features = self.midlevel_resnet(input)

    # Upsample to get colors
    output = self.upsample(midlevel_features)
    return output

#PATH
path = str(sys.argv[1])

#HELPER FUNCTIONS
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()

#MODEL
model = ColorizationNet()
model.load_state_dict(torch.load('./checkpoints/model-epoch-186-losses-0.124.pth', map_location=torch.device('cpu')))
model.train()

#LOAD IMAGE
image = Image.open(path)
size = (224*2,224*2)
resized = image.resize(size)

testtens = to_tensor(resized)
testtens = testtens[0:3]
rgb = convert.rgb_to_lab(testtens)
img_ab = rgb[1:3,:,:]

img_gray = rgb[0,:,:].unsqueeze(0)
img_gray = torch.stack([img_gray, torch.zeros(img_gray.shape), torch.zeros(img_gray.shape)],dim = 1)


model_output = model(img_gray)

output = model(img_gray)
output_l = img_gray[:,0,:,:]
output_a = output[:,0,:,:]
output_b = output[:,1,:,:]
output_ab = torch.cat([output_l,output_a,output_b],dim=0)
output_ab = convert.lab_to_rgb(output_ab)


output_grey = convert.lab_to_rgb(img_gray.squeeze(0))

final_img = to_pil(output_ab)
final_grey = to_pil(output_grey)

new_size = (224, 224)  # Replace with the desired size
resized_image = final_img.resize(new_size)

resized_image.save('evaluated-photos/out_cnn/output.jpg')
# plt.imshow(final_img)
# plt.show()

f, axarr = plt.subplots(1,3)
axarr[0].imshow(resized)
axarr[0].set_title('Original')
axarr[1].imshow(final_grey)
axarr[1].set_title('Greyscale')
axarr[2].imshow(final_img)
axarr[2].set_title('Colorized')
plt.show()
