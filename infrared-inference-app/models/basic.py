import torch
import torch.nn as nn

def down_block(in_c, out_c, use_bn=True):
    layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)

def up_block(in_c, out_c, use_bn=True, final=False):
    layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=not use_bn)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_c))
    if final:
        layers.append(nn.Tanh())
    else:
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)

class BasicUNetGenerator(nn.Module):
    # FIXED: Default base_f set to 256 to match your checkpoint size
    def __init__(self, input_nc=3, output_nc=1, base_f=256):
        super().__init__()
        f = base_f

        # Encoder
        self.down1 = down_block(input_nc, f, use_bn=False)  
        self.down2 = down_block(f, f*2)                     
        self.down3 = down_block(f*2, f*4)                   
        self.down4 = down_block(f*4, f*4)                   
        self.down5 = down_block(f*4, f*4) # Bottleneck

        # Decoder (Expects concatenation)
        self.up1 = up_block(f*4, f*4)
        self.up2 = up_block(f*8, f*4)
        self.up3 = up_block(f*8, f*2)
        self.up4 = up_block(f*4, f)
        self.up5 = up_block(f*2, output_nc, use_bn=False, final=True)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Skip connections (Concatenation)
        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        u5 = self.up5(torch.cat([u4, d1], dim=1))
        return u5