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
        
        # input tensor:  (1, 224, 224)
        # output tensor: (n + 2p - f)/S + 1  -> (32, 224, 224)
        # after pooling: (32, 112, 112)
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.3)
        )      
        # input tensor:  (32, 112, 112)
        # output tensor: (n + 2p - f)/S + 1 -> (64, 112, 112)
        # after pooling: (64, 56, 56)    
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.35)
        )
        
        # input tensor:  (64, 56, 56)
        # output tensor: (n + 2p - f)/S + 1 -> (128, 56, 56)
        # after max pooling: (128, 28, 28)   
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.35)
        )
        
        # input tensor:  (128, 28, 28)
        # output tensor: (n + 2p - f)/S + 1 -> (256, 28, 28)
        # after max pooling: (256, 14, 14)
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.35)
        )
        # input tensor:  (256, 14, 14)
        # output tensor: (n + 2p - f)/S + 1 -> (512, 14, 14)
        # after max pooling: (512, 7, 7)
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.35)
        )
        
        # input tensor:  (512, 7, 7)
        # output tensor: (n + 2p - f)/S + 1 -> (1024, 7, 7)
        # after max pooling: (1024, 1, 1)
        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            # nn.Dropout2d(p=0.35)
        )

        ### Flatten after Global Avg Pooling for Linear layer (512 feature maps) ###
        
        self.fc_layer7 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.4)
        )
        
        self.fc_layer8 = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=0.4)
        )
       
       # switch to 9 if adding extra FC layer
        self.fc_layer9 = nn.Sequential(
            nn.Linear(1000,136)
        )
        
      
        ## initialize weights
        for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                            weight_init.kaiming_uniform_(m.weight)
                    elif isinstance(m, nn.Linear):
                            weight_init.xavier_normal_(m.weight)
                            weight_init.uniform_(m.bias)
   
        
         
    
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        dtype = torch.cuda.FloatTensor
        x = torch.autograd.Variable(x.type(dtype))
        
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer7(x)
        x = self.fc_layer8(x)
        x = self.fc_layer9(x)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
