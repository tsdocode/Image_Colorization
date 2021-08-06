from Dataset.ColorDataset import LabImageDataset as LID
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms, utils
import numpy as np

transforms = transforms.Compose([
        transforms.Resize((600, 600)),
])

test = LID("../data", transform=transforms)

d , l = test[0]

img = test.reconstruct_rgb(d,l)


cv2.imshow("" ,img)
cv2.waitKey(500)


