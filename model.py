import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math


class DensenetExtractor(nn.Module):
    def __init__(self, pretrained):
        super(DensenetExtractor, self).__init__()
        self.features = torch.hub.load("pytorch/vision:v0.8.0", "densenet201", pretrained=pretrained).features
        
        self.conv1 = nn.Conv2d(256, 128, 1)
        self.conv2 = nn.Conv2d(512, 256, 1)
        self.conv3 = nn.Conv2d(1792, 512, 1)
        self.conv4 = nn.Conv2d(1920, 512, 1)

    def forward(self, x):
        out = []
        
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        
        x = self.features.denseblock1(x)
        out.append(self.conv1(x))
        x = self.features.transition1(x)
        
        x = self.features.denseblock2(x)
        out.append(self.conv2(x))
        x = self.features.transition2(x)
        
        x = self.features.denseblock3(x)
        out.append(self.conv3(x))
        x = self.features.transition3(x)
        
        x = self.features.denseblock4(x)
        out.append(self.conv4(x))
        
        return out


class Merger(nn.Module):
    def __init__(self):
        super(Merger, self).__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(384, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(192, 32, 1)
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
        y = self.relu1(self.bn1(self.conv1(y)))        
        y = self.relu2(self.bn2(self.conv2(y)))
        
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[1]), 1)
        y = self.relu3(self.bn3(self.conv3(y)))        
        y = self.relu4(self.bn4(self.conv4(y)))
        
        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[0]), 1)
        y = self.relu5(self.bn5(self.conv5(y)))        
        y = self.relu6(self.bn6(self.conv6(y)))
        
        y = self.relu7(self.bn7(self.conv7(y)))
        return y

class Outputer(nn.Module):
    def __init__(self, scope=512):
        super(Outputer, self).__init__()
        self.conv1 = nn.Conv2d(32, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(32, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = 512
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        score = self.sigmoid1(self.conv1(x))
        loc   = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
        geo   = torch.cat((loc, angle), 1) 
        return score, geo
        
    
class EAST(nn.Module):
    def __init__(self, pretrained=True):
        super(EAST, self).__init__()
        self.extractor = DensenetExtractor(pretrained)
        self.merge = Merger()
        self.output = Outputer()
    
    def forward(self, x):
        return self.output(self.merge(self.extractor(x)))
        

if __name__ == '__main__':
    m = EAST()
    x = torch.randn(1, 3, 256, 256)
    score, geo = m(x)
    print(score.shape)
    print(geo.shape)
