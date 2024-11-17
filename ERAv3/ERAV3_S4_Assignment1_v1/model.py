import torch
import torch.nn as nn

class MNIST_CNN(nn.Module):
    def __init__(self, kernel_config=None):
        super(MNIST_CNN, self).__init__()
        
        # Default configuration if none provided
        if kernel_config is None:
            kernel_config = [32, 64, 128, 128]
        
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, kernel_config[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv layer
            nn.Conv2d(kernel_config[0], kernel_config[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv layer
            nn.Conv2d(kernel_config[1], kernel_config[2], kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Fourth conv layer
            nn.Conv2d(kernel_config[2], kernel_config[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(kernel_config[3] * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x 