import torch
from torch import nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights
from torchvision.models._utils import IntermediateLayerGetter as ILG



def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class SiLU(nn.Module):  
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        return_layer = {"layer1": "1","layer2": "2","layer3": "3","layer4": "4"}
        self.body = ILG(backbone, return_layers=return_layer)
    def forward(self, x):
        outputs = self.body(x)
        x4, x8, x16, x32 = outputs['1'], outputs['2'], outputs['3'], outputs['4']
        return x4, x8, x16, x32


class Resnet34(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1) 
        return_layer = {"layer1": "1","layer2": "2","layer3": "3","layer4": "4"}
        self.body = ILG(backbone, return_layers=return_layer)
    def forward(self, x):
        outputs = self.body(x)
        x4, x8, x16, x32 = outputs['1'], outputs['2'], outputs['3'], outputs['4']
        return x4, x8, x16, x32



class SignalConv(nn.Module):
    """(convolution => [BN] => ReLU)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.signal_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # (c2, eps=0.001, momentum=0.03)
            nn.ReLU(inplace=True))    # nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.signal_conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)
    

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels, maxpool, Down_ratio=2):
        super().__init__()
        if maxpool:
            self.down = nn.MaxPool2d(Down_ratio)
        else:    
            self.down = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3,stride=2,padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True))
        self.conv = SignalConv(in_channels, out_channels)
        #self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x):
        x_out = self.down(x)
        return self.conv(x_out)



class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, chanel, Up_ratio=2, Up_modol='bilinear'):
        super().__init__()
        self.Up_ratio = Up_ratio
        self.mode     = Up_modol
        self.up   = nn.ConvTranspose2d(chanel, chanel, kernel_size=Up_ratio, stride=Up_ratio)
        self.conv = SignalConv(chanel,chanel)
    def forward(self, x):
        if self.mode == 'bilinear':
            x = F.interpolate(x, scale_factor=self.Up_ratio, mode='bilinear', align_corners=True)
        elif self.mode == 'TransConv':
            x = self.up(x)
        return self.conv(x)
    


class Up_cat(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, Up_modol='bilinear'):
        super().__init__()
        up_ratio = 2
        if Up_modol == 'bilinear':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
        elif Up_modol == 'TransConv':
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))   
        elif Up_modol == 'Pixel_shuffle':
            self.up = nn.Sequential(
                nn.Conv2d(in_channels, up_ratio * in_channels, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(upscale_factor=up_ratio),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True))
        self.conv = DoubleConv(out_channels*2, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Down_add(nn.Module):
    """Downscaling then double conv"""
    def __init__(self, in_channels, out_channels, maxpool):
        super().__init__()
        if maxpool:
            self.down = nn.MaxPool2d(2)
        else:    
            self.down = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True))
        self.conv1 = SignalConv(in_channels,out_channels)
        self.conv2 = SignalConv(out_channels,out_channels)
    def forward(self, x, y):
        x_ = self.down(x)
        self.conv2(self.conv1(x_)+y)
        return self.conv2(self.conv1(x_)+y)



class Pre_MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_layer=nn.GELU):  #num-layers=3
        super().__init__()
        self.num_layers = num_layers
        self.act_layer = act_layer
        h = [hidden_dim] * (num_layers - 2)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [hidden_dim]))
        self.output = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        out = self.output(x)
        return out


class Pre_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d,):
        super(Pre_Conv, self).__init__()
        inter_channels1 = in_channels//2
        inter_channels2 = in_channels//4
        self.block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=inter_channels1,kernel_size=3, padding=1, bias=False),
        norm_layer(inter_channels1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=inter_channels1, out_channels=inter_channels2,kernel_size=3, padding=1, bias=False),
        norm_layer(inter_channels2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1))
        self.output = nn.Conv2d(in_channels=inter_channels2, out_channels=out_channels,kernel_size=1)
    def forward(self, x):
        x = self.block(x)
        return self.output(x)



class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=SiLU(), deploy=False):
        super(RepConv, self).__init__()
        self.deploy         = deploy
        self.groups         = g
        self.in_channels    = c1
        self.out_channels   = c2
        
        assert k == 3
        assert autopad(k, p) == 1

        padding_11  = autopad(k, p) - k // 2
        self.act    = nn.LeakyReLU(0.1, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam    = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)
        else:
            self.rbr_identity   = (nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if c2 == c1 and s == 1 else None)
            self.rbr_dense      = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )
            self.rbr_1x1        = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2, eps=0.001, momentum=0.03),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3  = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1  = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid    = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel      = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma       = branch[1].weight
            beta        = branch[1].bias
            eps         = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel      = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma       = branch.weight
            beta        = branch.bias
            eps         = branch.eps
        std = (running_var + eps).sqrt()
        t   = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):
        std     = (bn.running_var + bn.eps).sqrt()
        bias    = bn.bias - bn.running_mean * bn.weight / std

        t       = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn      = nn.Identity()
        conv    = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias   = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense  = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1    = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias    = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1           = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded      = identity_conv_1x1.bias
            weight_identity_expanded    = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            bias_identity_expanded      = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded    = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        
        self.rbr_dense.weight   = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias     = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
        self.rbr_reparam    = self.rbr_dense
        self.deploy         = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None






if __name__ == '__main__':
    
    input = torch.rand(4,3,256,256)
