import torch
from network import build_network

def test_resnets():
    resnet_types = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    in_channels = 1
    num_classes = 4
    input_shape = (1, in_channels, 32, 32, 16)
    x = torch.randn(input_shape)
    for resnet in resnet_types:
        model = build_network(resnet, in_channels, num_classes)
        out = model(x)
        print(f"{resnet} output shape: {out.shape}")

def main():
    test_resnets()

if __name__ == "__main__":
    main() 