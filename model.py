import math
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def freeze_bn(model): 
    c = 0
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            c += 1
            module.eval()

    print(f"Freeze {c} batchnorm layers")

def init_weights(m):
    if isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1.0)
        m.bias.data.fill_(0.0)
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
   
    def __repr__(self):
        return f'GeM(p={self.p})'


class AdaCos(nn.Module):
    def __init__(self, in_features, out_features, m=0.50, ls_eps=0, theta_zero=math.pi/4):
        super(AdaCos, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta_zero = theta_zero
        self.s = math.log(out_features - 1) / math.cos(theta_zero)
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.weight)
        # dot product
        logits = F.linear(x, W)
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta)
            self.s = torch.log(B_avg) / torch.cos(torch.min(self.theta_zero * torch.ones_like(theta_med), theta_med))
        output *= self.s

        return output


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False, ls_eps=0.9, device='cuda'):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.device = device

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)).float()
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output

class Net(nn.Module):

    def __init__(self, backbone, n_classes, cfg, channel_size=512, pretrained=False):
        super(Net, self).__init__()
        neck = cfg.neck
        pool = cfg.pool
        self.name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained)
        self.channel_size = channel_size
        self.out_feature = n_classes

        if not isinstance(cfg.device, str):
            self.device = cfg.device
        else:
            self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))

        if cfg.freeze_bn:
            freeze_bn(self.backbone)
            
        if 'efficientnet' in backbone:
            self.in_features = self.backbone.classifier.in_features
        elif 'resne' in backbone: # Resnet family
            self.in_features = self.backbone.fc.in_features
        elif 'senet' in backbone:
            self.in_features = self.last_linear.in_features
        else:
            raise ValueError(backbone)

        if neck == "D":
            self.neck = nn.Sequential(
                nn.Linear(self.in_features, self.channel_size, bias=True),
                nn.BatchNorm1d(self.channel_size),
                torch.nn.PReLU()
            )
        elif neck == "F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.in_features, self.channel_size, bias=True),
                nn.BatchNorm1d(self.channel_size),
                torch.nn.PReLU()
            )
        else:
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.in_features, self.channel_size),
            )

        self.neck.apply(init_weights)
        # print("weight init: DONE")

        if cfg.head == "arcface":
            self.head = ArcMarginProduct(in_features=self.channel_size, out_features=self.out_feature,
                                        ls_eps=cfg.ls_eps, m=cfg.m, device=self.device)
        elif cfg.head == "adacos":
            self.head = AdaCos(self.channel_size, self.out_feature, m=cfg.m, ls_eps=cfg.ls_eps)

        if pool == 'gem':
            self.pooling = GeM(p=3)
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)


    def forward(self, x, labels=None, p=3):
        batch_size = x.shape[0]
        features = self.backbone.forward_features(x)
        features = self.pooling(features)
        features = features.view(batch_size, -1)

        features = self.neck(features)
        if labels is not None:
            return features, self.head(features, labels)
        else:
            return features

if __name__ == '__main__':
    model = Net('tf_efficientnet_b0', 2)
    img = torch.zeros((2, 3, 224, 224))
    feat = model(img)
    print(feat.shape)