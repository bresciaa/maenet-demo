import torch.nn as nn
import torch

import torch.nn as nn

'''
    Model configuration definition
'''
class Config:
    def __init__(self):
        # Data configs
        self.image_size = 256
        self.train_split = 0.8
        self.val_split = 0.2  # Not being used
        self.manual_seed = 42  # Can be modified
        self.allow_flip = False  # Will add randomized flip
        self.flip_probability = 0.5

        # Training configs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 60
        self.batch_size = 8
        self.learning_rate = 0.001

        # Loader configs
        self.allow_subset = False  # Optional subset
        self.subset = 2  # Subset size

        assert (
            0 < self.flip_probability <= 1
        ), "Flip probability must be between 0 and 1"

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1,
    ):
        super().__init__()
        self.expansion = 4
        self.stride = stride

        # ResNet
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x
    
# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # conv1
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet 152
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        # Multihead Attention
        self.query_proj = nn.Linear(256, 256)
        self.key_proj = nn.Linear(256, 256)
        self.value_proj = nn.Linear(256, 256)
        self.attention1 = nn.MultiheadAttention(embed_dim=256, num_heads=8)

    def forward(self, x):
        # Input size: 3, 512 x 512
        layer_outputs = []

        x = self.conv1(x)
        layer_outputs.append(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        layer_outputs.append(x)

        # Multihead Attention
        x_reshaped = x.flatten(2).transpose(1, 2)
        query = key = value = x_reshaped
        attn_output, attn_output_weights = self.attention1(query, key, value)
        attn_output = attn_output.transpose(1, 2).reshape(x.shape) # Reshape back
        max_attn_output = attn_output.max(dim=1, keepdim=True)[0]

        x = x + attn_output

        x = self.layer2(x)
        layer_outputs.append(x)
        x = self.layer3(x)
        layer_outputs.append(x)
        x = self.layer4(x)

        return x, layer_outputs

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(ResidualBlock, [3, 8, 36, 3], img_channel, num_classes)

class MAENet(nn.Module):
    def __init__(self, num_classes=2):
        super(MAENet, self).__init__()

        # ResNet 152 Encoder
        self.resnet = ResNet152(img_channel=3, num_classes=num_classes)

        # Decoder
        self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder1 = DecoderBlock(2048, 1024)

        self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder2 = DecoderBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DecoderBlock(512, 256)

        self.up4 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder4 = DecoderBlock(128, 64)

        self.up5 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(num_classes, num_classes, kernel_size=1)

    def forward(self, x):
        x, resnet_layer = self.resnet(x)
        # Input shape: [1, 2048, 16, 16]

        x = self.up1(x)
        x = torch.cat((x, resnet_layer[3]), dim=1)
        x = self.decoder1(x)

        x = self.up2(x)
        x = torch.cat((x, resnet_layer[2]), dim=1)
        x = self.decoder2(x)

        x = self.up3(x)
        x = torch.cat((x, resnet_layer[1]), dim=1)
        x = self.decoder3(x)

        x = self.up4(x)
        x = torch.cat((x, resnet_layer[0]), dim=1)
        x = self.decoder4(x)

        x = self.up5(x)
        x = self.final_conv(x)

        # Activation function
        # Note: This activation is commented out while Focal Loss is being
        #       used as the loss function. If Using BCE Loss, use the sigmoid
        #       activation function instead.
        # x = torch.sigmoid(x)
        x = x[:, 1, :, :]
        x = x.unsqueeze(1)
        return x


'''
    Return the MA-E-Net
'''
def load_model(path="app/k3_maenet.pth"):
    model = MAENet(num_classes=2)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model