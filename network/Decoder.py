from torch import nn


class Decoder(nn.Module):
    """
    Image to Patch Embedding
    copy Decoder_NN
    feature_channel = embed_dim
    """
    def __init__(self, feature_channel):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel, feature_channel // 2, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel // 2, feature_channel // 2, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel // 2, feature_channel // 2, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel // 2, feature_channel // 2, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel // 2, feature_channel // 4, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel // 4, feature_channel // 4, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel // 4, feature_channel // 8, (3, 3)),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel // 8, feature_channel // 8, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(feature_channel // 8, 3, (3, 3)),
        )

    def forward(self, x):
        # B, C, H, W = x.shape
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Decoder(feature_channel=64).to(device)

    img = torch.randn((1, 64, 16, 16)).to(device)
    out = net.forward(img)
    print(out.shape)