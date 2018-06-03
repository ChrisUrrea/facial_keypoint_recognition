## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as weight_init



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # input tensor:  (1, 227, 227)
        # output tensor: (n + 2p - f)/S + 1 = 95 -> (32, 220, 220)
        # after pooling: (32, 110, 110)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.5)
        )      
        # input tensor:  (32, 110, 110)
        # output tensor: (n + 2p - f)/S + 1 -> (64, 110, 110)
        # after pooling: (64, 50, 50)    
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4)
        )
        
        # input tensor:  (64, 50, 50)
        # output tensor: (n + 2p - f)/S + 1 -> (128, 50, 50)
        # after pooling: (128, 25, 25)   
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4)
        )
        
        # input tensor:  (128, 25, 25)
        # output tensor: (n + 2p - f)/S + 1 -> (256, 25, 25)
        # after global pooling: (256, 12, 12)
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.4)
        )
        
        # input tensor:  (256, 12, 12)
        # output tensor: (n + 2p - f)/S + 1 -> (512, 12, 12)
        # after global pooling: (512, 12, 12)
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(p=0.4)
        )
        

        ### Flatten after Global Avg Pooling for Linear layer ###
        self.fc_layer6 = nn.Sequential(
            nn.Linear(512, 700),
            nn.ReLU(),
            nn.BatchNorm1d(700),
            nn.Dropout(p=0.33)
        )
        
        
        self.fc_layer7 = nn.Sequential(
            nn.Linear(700,136)
        )
        
        for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                            weight_init.xavier_uniform_(m.weight)
                            weight_init.constant_(m.bias, 0)
                    elif isinstance(m, nn.Linear):
                            weight_init.xavier_normal_(m.weight)
                            weight_init.uniform_(m.bias, -0.1,0.1)
   
        
         
    
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer6(x)
        x = self.fc_layer7(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
