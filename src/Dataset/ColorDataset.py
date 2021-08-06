import os
import glob
from skimage import io, transform
from skimage.color import rgb2lab,rgb2gray,lab2rgb
import numpy as np
import torch
from skimage.transform import resize
from torchaudio import datasets
from tqdm import tqdm
from torchvision.datasets import ImageFolder


#

class LabImageDataset():
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        print("Start to load data!")
        self.load()
        print("Loaded!")

    def load(self):
        self.dataset = ImageFolder(self.path, transform= self.transform)


    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        #Get PIL image
        image = self.dataset[idx]
        #PIL to numpy
        image_original = np.array(image[0])

        #RGB to LAB
        image_lab=(rgb2lab(image_original)+128)/255
        image_ab=image_lab[:,:,1:3]
        image_ab=torch.from_numpy(image_ab.transpose((2,0,1))).float()
        
        #RGB to Gray
        image_original=rgb2gray(image_original)
        image_original=torch.from_numpy(image_original).unsqueeze(0)     



        data=image_original
        label=image_ab
        
        return data, label

    def reconstruct_rgb(self, grey_scale, ab_img):
        color_image = torch.cat((grey_scale, ab_img), 0).numpy() # combine channels
        color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
        color_image = lab2rgb(color_image.astype(np.float64))
        grey_scale = grey_scale.squeeze().numpy()
        return color_image


    