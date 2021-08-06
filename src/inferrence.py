from Model.autoencoder import AutoEncoder
import torch
from skimage import io, transform
from skimage.transform import resize
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from Dataset.ColorDataset import LabImageDataset
from torchvision import transforms, utils



transform = transforms.Compose([
        transforms.Resize((256,256)),
])


dataset = LabImageDataset("../data", transform)

test_image , truth= dataset[0]

model = AutoEncoder().cuda()
model.load_state_dict(torch.load("./model.pth"))
model.eval()


for param in model.parameters():
    print(param)

# print(test_image.shape)

test_image = test_image.unsqueeze(0).cuda()

output1 = model(test_image)


output1 = output1.squeeze(0).permute(1,2,0).cpu().detach().numpy()
print(output1)


n = np.where(output1 < 1)
print(n)

# # output1 = output1*128
# # result = np.zeros((256, 256, 3))

# # result[:,:,0] = _color[0][:,:,0]
# # result[:,:,1:] = output1[0]

# # io.imsave("result.png", lab2rgb(result))