import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math


class ResnetExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(ResnetExtractor, self).__init__()
        self.features = torch.hub.load("pytorch/vision:v0.8.0", "resnet34", pretrained=pretrained)

    def forward(self, x):
        x = self.features.bn1(self.features.conv1(x))
        x = self.features.maxpool(self.features.relu(x))
        
        out = []
        x = self.features.layer1(x) # 64, s/4, s/4
        out.append(x)
        x = self.features.layer2(x) # 128, s/8, s/8
        out.append(x)
        x = self.features.layer3(x) # 256, s/16, s/16
        out.append(x)
        x = self.features.layer4(x) # 512, s/32, s/32
        out.append(x)
        
        return out


class Merger(nn.Module):
    def __init__(self):
        super(Merger, self).__init__()

        self.conv1 = nn.Conv2d(768, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(256, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 32, 1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[2]), 1)
        # 512 + 256
        y = self.relu1(self.bn1(self.conv1(y)))        
        y = self.relu2(self.bn2(self.conv2(y)))
        
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[1]), 1)
        # 128 + 128
        y = self.relu3(self.bn3(self.conv3(y)))        
        y = self.relu4(self.bn4(self.conv4(y)))
        
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[0]), 1)
        # 64 + 64
        y = self.relu5(self.bn5(self.conv5(y)))        
        y = self.relu6(self.bn6(self.conv6(y)))
        
        y = self.relu7(self.bn7(self.conv7(y)))
        return y

class Outputer(nn.Module):
    def __init__(self):
        super(Outputer, self).__init__()
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, scope=512):
        score = self.sigmoid1(self.conv1(x))
        loc   = self.sigmoid2(self.conv2(x)) * scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
        geo   = torch.cat((loc, angle), 1) 
        return score, geo
        
    
class EAST(nn.Module):
    def __init__(self, pretrained=True):
        super(EAST, self).__init__()
        self.extractor = ResnetExtractor(pretrained)
        self.merge = Merger()
        self.output = Outputer()
    
    def forward(self, x, scope=512):
        return self.output(self.merge(self.extractor(x)), scope)
        

if __name__ == '__main__':
    m = EAST()
    x = torch.randn(1, 3, 256, 256)
    score, geo = m(x)
    print(score.shape)
    print(geo.shape)
