
class CNN1(nn.Module):
  def __init__(self, l1, l2, l3):
    super().__init__()
    self.con_layer1 = nn.Conv2d(in_channels = 3, out_channels = l1, kernel_size=5, padding=1)
    self.con_layer2 = nn.Conv2d(in_channels = l1, out_channels = l2, kernel_size=3, padding=1)
    self.con_layer3 = nn.Conv2d(in_channels = l2, out_channels = l3, kernel_size = 5, padding = 1)
    self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.flat1 = nn.LazyLinear(10)

    self.model = nn.Sequential(
        self.con_layer1, nn.ReLU(), self.max_pool,
        self.con_layer2, nn.ReLU(), self.max_pool,
        self.con_layer3, nn.ReLU(), self.max_pool,
        nn.Flatten(),
        self.flat1

        )

  def forward(self, x):
    return self.model(x)


# RESNET18


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetMNIST, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)  # 28x28
        self.layer2 = self._make_layer(128, 2, stride=2) # 14x14
        self.layer3 = self._make_layer(256, 2, stride=2) # 7x7
        self.layer4 = self._make_layer(512, 2, stride=2) # 4x4
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

