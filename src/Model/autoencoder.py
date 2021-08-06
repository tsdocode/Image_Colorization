from typing import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch 

#check
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(OrderedDict(
            [
               ("conv_1" , nn.Conv2d(1, 64, 3, stride= 2, padding=1)),
               ("relu_1" , nn.ReLU()),
               ("conv_2" , nn.Conv2d(64, 128, 3,  padding=1)),
               ("relu_2" , nn.ReLU()),
               ("conv_3" , nn.Conv2d(128, 128, 3, stride = 2,  padding=1)),
               ("relu_3" , nn.ReLU()),
               ("conv_4" , nn.Conv2d(128, 256, 3,  padding=1)),
               ("relu_4" , nn.ReLU()),
               ("conv_5" , nn.Conv2d(256, 256, 3, stride = 2 , padding=1)),
               ("relu_5" , nn.ReLU()),
               ("conv_6" , nn.Conv2d(256, 512, 3,  padding=1)),
               ("relu_6" , nn.ReLU()),
               ("conv_7" , nn.Conv2d(512, 512, 3,  padding=1)),
               ("relu_7" , nn.ReLU()),
               ("conv_8" , nn.Conv2d(512, 256, 3,  padding=1)),
               ("relu_8" , nn.ReLU()),
            ]
        ))
        self.decoder = nn.Sequential(OrderedDict(
            [
                ("conv_9" , nn.Conv2d(256, 128, 3,  padding=1)),
                ("relu_9" , nn.ReLU()),
                ("upsample" , nn.Upsample(scale_factor=2)),
                ("conv_10" , nn.Conv2d(128, 64, 3,  padding=1)),
                ("relu_10" , nn.ReLU()),
                ("upsample_1" , nn.Upsample(scale_factor=2)),
                ("conv_11" , nn.Conv2d(64, 32, 3,  padding=1)),
                ("relu_11" , nn.ReLU()),
                ("conv_12" , nn.Conv2d(32, 16, 3,  padding=1)),
                ("relu_12" , nn.ReLU()),
                ("conv_13" , nn.Conv2d(16, 2, 3,padding=1)),
                ("tanh" , nn.Tanh()),
                ("upsample_2" , nn.Upsample(scale_factor=2)),
            ]
        ))

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    test = torch.randn(1, 1, 256, 256)
    model = AutoEncoder()
    output = model(test)
    # print(output.shape)