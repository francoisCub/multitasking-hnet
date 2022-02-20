from collections import OrderedDict

from torch import nn, unique

from torchvision.models import resnet18


class ConvMNIST(nn.Module):
    def __init__(self, dim=10, nonlinear=True):
        super().__init__()
        self.dim = dim
        activation = nn.PReLU if nonlinear else nn.Identity
        self.convs = nn.Sequential(
                                nn.Conv2d(1, 1, 5, bias=False), activation(),
                                nn.Conv2d(1, 1, 5, bias=False), activation(),
                                nn.Conv2d(1, 1, 5, bias=False), activation(),
                                nn.Conv2d(1, 1, 5, bias=False), activation(),
                                nn.Conv2d(1, 1, 5, bias=False), activation(),
                                nn.Conv2d(1, 1, 5, bias=False), activation())
        self.out = nn.Linear(1 * 4 * 4, dim)

    def forward(self, x):
        x = self.convs(x)
        y = self.out(x.view(x.shape[0], -1))
        return y


class ConvTaskEnsembleMNIST(nn.Module):
    def __init__(self, dim=2, nbr_task=5, nonlinear=True):
        super().__init__()
        self.task_models = nn.ModuleList(
            [ConvMNIST(dim=dim, nonlinear=nonlinear) for _ in range(nbr_task)])

    def forward(self, x, task=None):
        task = unique(task)
        if len(task) > 1:
            raise ValueError()
        y = self.task_models[task](x)
        return y


class ConvTaskEnsembleCIFAR(nn.Module):
    def __init__(self, resnet_base, in_channels, num_classes, n=1, nbr_task=10):
        super().__init__()
        self.task_models = nn.ModuleList(
            [resnet_base(in_channels=in_channels, num_classes=num_classes, n=n) for _ in range(nbr_task)])

    def forward(self, x, task=None):
        task = unique(task)
        if len(task) > 1:
            raise ValueError()
        y = self.task_models[task](x)
        return y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=kernel_size, stride=1, padding=padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU()
        if stride > 1 or self.in_channels != self.out_channels:
            self.conv1x1 = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.stride == 1 and self.in_channels == self.out_channels:
            # ReLU can be applied before or after adding the input
            return self.relu2(out) + x
        else:
            return self.relu2(out) + self.conv1x1(x)


class ExampleCifar(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, n=1):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(
            in_channels*32*32, num_classes))

    def forward(self, x):
        return self.net(x)


class TinyResNet(nn.Module):
    def __init__(self, in_channels, num_classes, n=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1, bias=False), nn.ReLU())
        self.res1 = nn.Sequential(
            *[ResidualBlock(3, 3, stride=1) for _ in range(n-1)])
        self.res2 = ResidualBlock(3, 5, stride=2)
        self.res3 = nn.Sequential(
            *[ResidualBlock(5, 5, stride=1) for _ in range(n-1)])
        self.res4 = ResidualBlock(5, 10, stride=2)
        self.res5 = ResidualBlock(10, 10)

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                        nn.Flatten(),
                                        nn.Linear(10, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.classifier(out)
        return out


class SmallResNet(nn.Module):
    def __init__(self, in_channels, num_classes, n=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False), nn.ReLU())
        self.res1 = nn.Sequential(
            *[ResidualBlock(16, 16, stride=1) for _ in range(n-1)])
        self.res2 = ResidualBlock(16, 32, stride=2)
        self.res3 = nn.Sequential(
            *[ResidualBlock(32, 32, stride=1) for _ in range(n-1)])
        self.res4 = ResidualBlock(32, 64, stride=2)
        self.res5 = ResidualBlock(64, 64)

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                        nn.Flatten(),
                                        nn.Linear(64, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.classifier(out)
        return out


class ResNet32x32(nn.Module):
    def __init__(self, in_channels, num_classes, n=1):
        super().__init__()
        self.convto16 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=True), nn.ReLU())
        self.res16 = nn.Sequential(
            *[ResidualBlock(16, 16, stride=1) for _ in range(n-1)])
        self.res16to32 = ResidualBlock(16, 32, stride=2)
        self.res32 = nn.Sequential(
            *[ResidualBlock(32, 32, stride=1) for _ in range(n-1)])
        self.res32to64 = ResidualBlock(32, 64, stride=2)
        self.res64 = nn.Sequential(
            *[ResidualBlock(64, 64, stride=1) for _ in range(n-1)])

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                        nn.Flatten(),
                                        nn.Linear(64, num_classes))

    def forward(self, x):
        out = self.convto16(x)
        out = self.res16(out)
        out = self.res16to32(out)
        out = self.res32(out)
        out = self.res32to64(out)
        out = self.res64(out)
        out = self.classifier(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, n=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=1, bias=False), nn.ReLU())
        self.res1 = nn.Sequential(
            *[ResidualBlock(64, 64, stride=1) for _ in range(n-1)])
        self.res2 = ResidualBlock(64, 128, stride=2)
        self.res3 = nn.Sequential(
            *[ResidualBlock(128, 128, stride=1) for _ in range(n-1)])
        self.res4 = ResidualBlock(128, 256, stride=2)
        self.res5 = nn.Sequential(
            *[ResidualBlock(256, 256, stride=1) for _ in range(n-1)])
        self.res6 = ResidualBlock(256, 512, stride=2)
        self.res7 = ResidualBlock(512, 512, stride=1)

        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res7(out)
        out = self.classifier(out)
        return out

# Resnet and conv_block from https://www.kaggle.com/kmldas/cifar10-resnet-90-accuracy-less-than-5-min


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()
        if pool:
            self.pooling = nn.AveragePool2d(2)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        if self.pool:
            out = self.pooling(out)
        return out


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              #   nn.BatchNorm2d(out_channels), # we don't do that here
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, n=1):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

def replace_bn(m, replace_conv=False, remove_bn=False):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if isinstance(target_attr, nn.BatchNorm2d):
            if remove_bn:
                setattr(m, attr_str, nn.Identity())
            else:
                setattr(m, attr_str, nn.BatchNorm2d(num_features=target_attr.num_features, eps=target_attr.eps, momentum=1.0, affine=target_attr.affine, track_running_stats=False))
        elif replace_conv and isinstance(target_attr, nn.Conv2d):
            setattr(m, attr_str, nn.Conv2d(in_channels=target_attr.in_channels, out_channels=target_attr.out_channels,
            kernel_size=target_attr.kernel_size, padding=target_attr.padding, bias=True, stride=target_attr.stride,
            groups=target_attr.groups, padding_mode=target_attr.padding_mode, dilation=target_attr.dilation))
        elif isinstance(target_attr, nn.MaxPool2d):
            setattr(m, attr_str, nn.Identity())
    if isinstance(m, nn.Sequential):
        for i in range(len(m)):
            if isinstance(m[i], nn.BatchNorm2d):
                m[i] = nn.Identity()
    for n, ch in m.named_children():
        replace_bn(ch, replace_conv, remove_bn)

# https://discuss.pytorch.org/t/how-to-replace-a-layer-with-own-custom-variant/43586
def get_resnet18(in_channels, num_classes, n=None):
    model = resnet18(pretrained=False)
    replace_bn(model,replace_conv=False, remove_bn=False)
    model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(in_features=512, out_features=num_classes)
    return model
