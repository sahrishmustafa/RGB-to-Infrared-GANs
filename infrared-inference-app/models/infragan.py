import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, submodule=None, outermost=False, innermost=False):
        super().__init__()

        # Standard down-sampling block
        down = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, True)
        )

        if innermost:
            # Innermost: just down and up, no submodule
            up = nn.Sequential(
                nn.ConvTranspose2d(out_c, in_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(True)
            )
            self.model = nn.Sequential(down, up)
        elif outermost:
            # Outermost: down, submodule, up (to 1 channel IR output), Tanh
            self.model = nn.Sequential(
                down,
                submodule,
                nn.ConvTranspose2d(out_c, 1, 4, 2, 1), # Output is 1 channel (Infrared)
                nn.Tanh()
            )
        else:
            # Intermediate: down, submodule, up
            up = nn.Sequential(
                nn.ConvTranspose2d(out_c, in_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(in_c),
                nn.ReLU(True)
            )
            self.model = nn.Sequential(down, submodule, up)

    def forward(self, x):
        # Recursive execution
        for layer in self.model:
            x = layer(x)
        return x

def build_unet_from_file(input_nc=3):
    # This matches the structure in your infraGan.py
    # Structure: 3 -> 64 -> 128 -> 256 -> 512 -> 512 (Innermost)
    
    innermost = UNetBlock(512, 512, None, innermost=True)
    block = UNetBlock(256, 512, innermost)
    block = UNetBlock(128, 256, block)
    block = UNetBlock(64, 128, block)
    
    # Outermost takes input_nc (3) and maps to 64, wrapping the whole block
    model = UNetBlock(input_nc, 64, block, outermost=True)
    return model

# FIXED: This is now a function that returns the model directly.
# This removes the extra "model.model" nesting that caused your error.
def InfraGANGenerator(input_nc=3, output_nc=1):
    return build_unet_from_file(input_nc)