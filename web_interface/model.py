import torch.nn as nn

#######################################
### PRE-TRAINED MODELS AVAILABLE HERE
## https://pytorch.org/vision/stable/models.html
from torchvision import models
#######################################

def resnet18(num_classes):
        """Transfer learning with VGG16 pretrained on ImageNet dataset.

        Args:
            num_classes (int): number of classes

        Returns:
            _type_: model
        """
        
        # get pretrained model, VGG16 pre-trained on ImageNet
        model = models.vgg16(pretrained=True)

        # freeze whole model
        for param in model.parameters():
                param.requires_grad = False
                
        # replace last fc layer with new fc layer(s)
        model.classifier[-1] = nn.Sequential(
                        nn.Linear(4096, 512), 
                        nn.ReLU(), 
                        nn.Dropout(0.5),
                        nn.Linear(512, num_classes))
        return model