import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class Baseline_Net(nn.Module):
    def __init__(self):
        super(Baseline_Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(9216, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 9216)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def get_outputs(self, images):
        outputs = []
        for i in range(images.size()[0]):
            image = images[i][None, :, :, :]
            output = self.forward(image)
            outputs.append(output)
        mean = torch.mean(torch.stack(outputs), axis=0)
        return mean


class ResNet_without_residual(nn.Module):
    def __init__(self):
        super(ResNet_without_residual, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)

        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.bn5 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(12544, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool2(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))

        x = x.view(-1, 12544)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc1(x))
        return x

    def get_outputs(self, images):
        outputs = []
        for i in range(images.size()[0]):
            image = images[i][None, :, :, :]
            output = self.forward(image)
            outputs.append(output)
        mean = torch.mean(torch.stack(outputs), axis=0)
        return mean


class ResNet(nn.Module):
    def __init__(self, block, K=32):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, K, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(K)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = block(K, K, stride=1)
        self.layer2 = block(K, 2 * K, stride=2)
        self.layer3 = block(2 * K, 4 * K, stride=2)
        self.layer4 = block(4 * K, 8 * K, stride=2)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(8 * 7 * 7 * K, 1)
        self.out_shape = 8 * 7 * 7 * K

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(self.bn1(x))
        x = self.pool2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, self.out_shape)
        x = self.dropout(x)
        return x

    def get_outputs(self, images):
        outputs = []
        for i in range(images.size()[0]):
            image = images[i][None, :, :, :]
            output = self.forward(image)
            output = torch.sigmoid(self.fc1(output))
            outputs.append(output)
        mean = torch.mean(torch.stack(outputs), axis=0)
        return mean

    def get_features(self, images):
        outputs = []
        for i in range(images.size()[0]):
            image = images[i][None, :, :, :]
            output = self.forward(image)
            outputs.append(output)
        return torch.mean(torch.stack(outputs), axis=0)


class ResNet_with_clinical_attributes(nn.Module):
    def __init__(self, block, K=32):
        super(ResNet_with_clinical_attributes, self).__init__()
        self.conv1 = nn.Conv2d(3, K, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(K)
        self.pool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = block(K, K, stride=1)
        self.layer2 = block(K, 2 * K, stride=2)
        self.layer3 = block(2 * K, 4 * K, stride=2)
        self.layer4 = block(4 * K, 8 * K, stride=2)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(8 * 7 * 7 * K + 2, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 1)
        self.out_shape = 8 * 7 * 7 * K

    def first_layer(self, x):
        x = self.conv1(x)
        x = torch.relu(self.bn1(x))
        x = self.pool2(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, self.out_shape)
        x = self.dropout(x)
        return x

    def forward(self, images, data):
        outputs = []
        for i in range(images.size()[0]):
            image = images[i][None, :, :, :]
            output = self.first_layer(image)
            output = torch.cat((output.flatten(), data), 0)
            output = F.relu(self.fc1(output))
            output = self.dropout(output)
            output = F.relu(self.fc2(output))
            output = self.dropout(output)
            output = F.relu(self.fc3(output))
            output = self.dropout(output)
            output = torch.sigmoid(self.fc4(output))
            outputs.append(output)
        mean = torch.mean(torch.stack(outputs), axis=0)
        return mean

    def get_features(self, images):
        outputs = []
        for i in range(images.size()[0]):
            image = images[i][None, :, :, :]
            output = self.forward(image)
            outputs.append(output)
        return torch.mean(torch.stack(outputs), axis=0)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class GatingNetwork(nn.Module):
    """
    A gating network that takes an aggregated feature map for a set of smears, and patient attributes.
    It outputs the probability that the final prediction should favour the blood smears.
    """

    def __init__(self, cnn_outshape):
        super(GatingNetwork, self).__init__()

        self.fc1 = nn.Linear(cnn_outshape, 1)
        self.fc2 = nn.Linear(3, 2)
        self.tanh = torch.tanh
        self.cnn_outshape = cnn_outshape

    def forward(self, fmap, attr):
        x = self.tanh(self.fc1(fmap.view(-1, self.cnn_outshape)))
        x = torch.cat((x[0], attr), dim=-1)
        probs = F.softmax(self.fc2(self.x))
        return probs[0]


class MOE(nn.Module):
    def __init__(self, K=32):
        super(MOE, self).__init__()
        self.cnn = ResNet(BasicBlock, K)
        self.mlp = MLP()
        self.gated_net = GatingNetwork(self.cnn.out_shape)

    def forward(self, x, images):
        out_mlp = self.mlp(x)
        out_cnn = self.cnn.get_outputs(images)
        features_cnn = self.cnn.get_features(images)
        prob = self.gated_net(features_cnn, x)
        y_pred = prob * out_cnn + (1 - prob) * out_mlp
        y_pred = torch.sigmoid(y_pred)
        return y_pred
