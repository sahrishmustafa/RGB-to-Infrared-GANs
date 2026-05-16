import torch
import torch.nn as nn

def conv_block(in_ch, out_ch, kernel=3, stride=1, padding=1, use_bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)]
    if use_bn:
        layers.append(nn.InstanceNorm2d(out_ch, affine=False))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def deconv_block(in_c, out_c, kernel=2, stride=2, padding=0, use_bn=True):
    layers = [nn.ConvTranspose2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding)]
    if use_bn:
        layers.append(nn.InstanceNorm2d(out_c, affine=False))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

class ClawGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        # Encoder
        self.enc1 = nn.Sequential(conv_block(in_channels, f), conv_block(f, f))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(conv_block(f, f*2), conv_block(f*2, f*2))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(conv_block(f*2, f*4), conv_block(f*4, f*4))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(conv_block(f*4, f*8), conv_block(f*8, f*8)) # bottleneck

        # Decoder A
        self.up3_a = deconv_block(f*8, f*4)
        self.dec3_a = nn.Sequential(conv_block(f*8, f*4), conv_block(f*4, f*4))
        self.up2_a = deconv_block(f*4, f*2)
        self.dec2_a = nn.Sequential(conv_block(f*4, f*2), conv_block(f*2, f*2))
        self.up1_a = deconv_block(f*2, f)
        self.dec1_a = nn.Sequential(conv_block(f*2, f), conv_block(f, f))

        # Decoder B (Claw)
        self.up3_b = deconv_block(f*8, f*4)
        self.dec3_b = nn.Sequential(conv_block(f*8, f*4), conv_block(f*4, f*4))
        self.up2_b = deconv_block(f*4, f*2)
        self.dec2_b = nn.Sequential(conv_block(f*4, f*2), conv_block(f*2, f*2))
        self.up1_b = deconv_block(f*2, f)
        self.dec1_b = nn.Sequential(conv_block(f*2, f), conv_block(f, f))

        self.final_conv = nn.Conv2d(f, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        c4 = self.enc4(p3)

        # Decoder A
        u3a = self.up3_a(c4)
        cat3a = torch.cat([u3a, c3], dim=1)
        out3a = self.dec3_a(cat3a)

        u2a = self.up2_a(out3a)
        cat2a = torch.cat([u2a, c2], dim=1)
        out2a = self.dec2_a(cat2a)

        u1a = self.up1_a(out2a)
        cat1a = torch.cat([u1a, c1], dim=1)
        out1a = self.dec1_a(cat1a)

        # Decoder B
        u3b = self.up3_b(c4)
        cat3b = torch.cat([u3b, out3a], dim=1)
        out3b = self.dec3_b(cat3b)

        u2b = self.up2_b(out3b)
        cat2b = torch.cat([u2b, out2a], dim=1)
        out2b = self.dec2_b(cat2b)

        u1b = self.up1_b(out2b)
        cat1b = torch.cat([u1b, out1a], dim=1)
        out1b = self.dec1_b(cat1b)

        merged = 0.5 * (out1a + out1b)
        out = self.final_conv(merged)
        return self.tanh(out)