
from torch import nn
import torch.nn.functional as F
from nets.net_parts import  SignalConv, Down_add, Up
from nets.attention_parts import PSSA

    
class Up_add(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, chanel, Up_ratio=2, Up_modol='bilinear'):
        super().__init__()
        self.Up_ratio = Up_ratio
        self.mode     = Up_modol
        self.up = nn.ConvTranspose2d(chanel, chanel, kernel_size=Up_ratio, stride=Up_ratio)
        self.conv1 = SignalConv(chanel,chanel)
        self.conv2 = SignalConv(chanel,chanel)
    def forward(self, x, y):
        if self.mode == 'bilinear':
            out = self.conv1(F.interpolate(x, scale_factor=self.Up_ratio, mode='bilinear', align_corners=True))+y
        elif self.mode == 'TransConv':
            out = self.conv1(self.up(x))+y
        return self.conv2(out)




class Res_SA_UP_DOWN(nn.Module):
    def __init__(self, backbone, anchors_mask, num_classes, chanel=64, down_num=5, 
                 Up_modol='TransConv', pre_model='mlp',
                 prior=0.01, depth=[2,2,4,4,4,4], num_heads=[4,4,4,4,4,4],
                 mlp_ratio=[4,4,4,4,4,4], sr_ratio=[[2,2,2,2,2,1]], cr_ratio=[[16,8,4,2,1,1]], deep_supervise=False):
        super().__init__()#[[2],[2],[4],[1,1,1,1],[2,2,2],[3,3]]
        self.backbone   = backbone #Up_ratio=[8,4,2,1],
        # Top layer
        self.toplayer  = nn.Conv2d(512, chanel, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer4 = nn.Conv2d(256, chanel, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, chanel, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 64, chanel, kernel_size=1, stride=1, padding=0)
        # up_add + Smooth layers
        self.up_add4   = Up_add(chanel, Up_ratio=2, Up_modol=Up_modol)  #up+smooth
        self.up_add3   = Up_add(chanel, Up_ratio=2, Up_modol=Up_modol)
        self.up_add2   = Up_add(chanel, Up_ratio=2, Up_modol=Up_modol) 

        # SR self-attention
        self.PSA2 = PSSA(embed_dim=chanel, depth=depth[2], num_heads=num_heads[2], 
                        mlp_ratio=mlp_ratio[2], sr_ratio=sr_ratio[2],cr_ratio=cr_ratio[2])
        self.PSA3 = PSSA(embed_dim=chanel, depth=depth[3], num_heads=num_heads[3], 
                        mlp_ratio=mlp_ratio[3], sr_ratio=sr_ratio[3],cr_ratio=cr_ratio[3])
        self.PSA4 = PSSA(embed_dim=chanel, depth=depth[4], num_heads=num_heads[4], 
                        mlp_ratio=mlp_ratio[4], sr_ratio=sr_ratio[4],cr_ratio=cr_ratio[4])
        self.PSA5 = PSSA(embed_dim=chanel, depth=depth[5], num_heads=num_heads[5], 
                        mlp_ratio=mlp_ratio[5], sr_ratio=sr_ratio[5],cr_ratio=cr_ratio[5])
        # down
        self.down_add5 = Down_add(chanel, chanel, maxpool=False)
        self.down_add4 = Down_add(chanel, chanel, maxpool=False)
        self.down_add3 = Down_add(chanel, chanel, maxpool=False)
        # chanel_ratio2
        self.rep_conv_3 = SignalConv(chanel, chanel*2)  # * 4
        self.rep_conv_4 = SignalConv(chanel, chanel*2)  # * 8
        self.rep_conv_5 = SignalConv(chanel, chanel*2)  # * 16
        # head
        self.yolo_head_s3 = nn.Conv2d(chanel*2, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_s4 = nn.Conv2d(chanel*2, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_s5 = nn.Conv2d(chanel*2, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, x):
        ## Res_SA_UP_DOWN
        ## backbone
        x4, x8, x16, x32 = self.backbone(x)
        ## reduce channel
        x32 = self.toplayer (x32)
        x16 = self.latlayer4(x16)
        x8  = self.latlayer3( x8)
        x4  = self.latlayer2( x4)
        ## SA
        t32 = self.PSA5(x32)
        t16 = self.PSA4(x16)
        t8  = self.PSA3( x8)
        t4  = self.PSA2( x4)
        ## UP_add
        p16 = self.up_add4(t32,t16)
        p8  = self.up_add3(p16, t8)
        p4  = self.up_add2( p8, t4)
        ## Down_add
        s3 = self.down_add3(p4,p8)
        s4 = self.down_add4(s3,p16)
        s5 = self.down_add5(s4,t32)
        ## Rep_conv
        s3_ = self.rep_conv_3(s3)
        s4_ = self.rep_conv_4(s4)
        s5_ = self.rep_conv_5(s5)
        ## head
        out2 = self.yolo_head_s3(s3_)
        out1 = self.yolo_head_s4(s4_)
        out0 = self.yolo_head_s5(s5_)

        return [out0, out1, out2]




if __name__ == '__main__':
    import torch
    input = torch.rand(4,3,640,640)

