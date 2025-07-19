import torch
import torchvision
import torch.nn as nn
from torchvision.models import resnet18

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

""" Added Flow Fusion Encoder """
class ResNet18FlowFusion(nn.Module):
    def __init__(self, flow_channels=2):
        super().__init__()
        # Load base resnet without pretrained weights
        base = resnet18(weights=None)

        # Save reference to components
        self.img_conv1 = base.conv1  # original: in_channels=3, out_channels=64
        self.flow_proj = nn.Sequential(
            nn.Conv2d(flow_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Keep all layers as in original
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.out_dim = 512  # keep track for external use

        # Remove fc layer (we want raw features)
        self.fc = nn.Identity()

    def forward(self, img, flow):
        """
        img: [B, 3, H, W]
        flow: [B, 2 or 3, H, W]
        """
        x_img = self.img_conv1(img)         # [B, 64, H/2, W/2]
        x_flow = self.flow_proj(flow)       # [B, 64, H/2, W/2]
        x = x_img + x_flow                  # fuse

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        return x



def get_resnet18_flow_fusion():
    return ResNet18FlowFusion(flow_channels=2)
    
