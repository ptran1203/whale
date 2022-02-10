import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class ArcModule(nn.Module):
    def __init__(self, in_features, out_features, s=10, m=0.3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = torch.tensor(math.cos(math.pi - m))
        self.mm = torch.tensor(math.sin(math.pi - m) * m)

    def forward(self, inputs, labels):
        cos_th = F.linear(inputs, F.normalize(self.weight))
        cos_th = cos_th.clamp(-1, 1)
        sin_th = torch.sqrt(1.0 - torch.pow(cos_th, 2))
        cos_th_m = cos_th * self.cos_m - sin_th * self.sin_m
        # print(cos_th.dtype, self.th.dtype, cos_th_m.dtype, self.mm.dtype)
        cos_th_m = torch.where(cos_th > self.th, cos_th_m, cos_th - self.mm)

        cond_v = cos_th - self.th
        cond = cond_v <= 0
        cos_th_m[cond] = (cos_th - self.mm)[cond]

        if labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        onehot = torch.zeros(cos_th.size()).cuda()
        labels = labels.type(torch.LongTensor).cuda()
        onehot.scatter_(1, labels, 1.0)
        outputs = onehot * cos_th_m + (1.0 - onehot) * cos_th
        outputs = outputs * self.s
        return outputs

class Net(nn.Module):

    def __init__(self, backbone, n_classes, channel_size=512, dropout=0.3, pretrained=False):
        super(Net, self).__init__()
        self.name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.channel_size = channel_size
        self.out_feature = n_classes
        self.in_features = self.backbone.classifier.in_features
        self.margin = ArcModule(in_features=self.channel_size, out_features=self.out_feature)
        self.dropout = nn.Dropout2d(dropout, inplace=True)
        self.fc = nn.Linear(self.in_features, self.channel_size)
        self.bn = nn.BatchNorm1d(self.channel_size)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x, labels=None, p=3):
        batch_size = x.shape[0]
        features = self.backbone.forward_features(x)
        features = gem(features, p=p).view(batch_size, -1)
        # features = self.pooling(features).view(batch_size, -1)
        
        # features = self.dropout(features)
        features = self.fc(features)
        features = self.bn(features)
        features = F.normalize(features)
        if labels is not None:
            return self.margin(features, labels)
        else:
            return features

if __name__ == '__main__':
    model = Net('tf_efficientnet_b0', 2)
    img = torch.zeros((2, 3, 224, 224))
    feat = model(img)
    print(feat.shape)