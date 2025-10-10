import torch, torch.nn as nn

def conv_bn_relu(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class UNetTiny(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base=16):
        super().__init__()
        self.enc1 = conv_bn_relu(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_bn_relu(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.bott = conv_bn_relu(base*2, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = conv_bn_relu(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = conv_bn_relu(base*2, base)
        self.out  = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bott(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        logits = self.out(d1)
        return logits
