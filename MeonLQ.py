import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Gdn import Gdn


class MyView(nn.Module):
    # only for estimating size
    def __init__(self, size):
        super(MyView, self).__init__()
        self.size = size
    def forward(self, x):
        return x.view(self.size)

class MyCat(nn.Module):
    # only for estimating size
    def __init__(self, dim):
        super(MyCat, self).__init__()
        self.dim = dim
    def forward(self, x, y):
        return torch.cat((x, y), self.dim)

class MeonNoDT(nn.Module):
    # only score
    def __init__(self):
        super(MeonNoDT, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.gdn1 = Gdn(8)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.gdn2 = Gdn(16)
        self.conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.gdn3 = Gdn(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.gdn4 = Gdn(64)

        self.st2_fc1 = nn.Conv2d(64, 256, 1, stride=1, padding=0)
        self.st2_gdn1 = Gdn(256)
        self.st2_fc2 = nn.Conv2d(256, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]

        x = F.max_pool2d(self.gdn1(self.conv1(x)), (2, 2))
        x = F.max_pool2d(self.gdn2(self.conv2(x)), (2, 2))
        x = F.max_pool2d(self.gdn3(self.conv3(x)), (2, 2))
        x = F.max_pool2d(self.gdn4(self.conv4(x)), (2, 2))

        y2 = self.st2_gdn1(self.st2_fc1(x))
        s = self.st2_fc2(y2)
        s = s.view(batch_size, -1)

        return s

class MeonNoDTSize(nn.Module):
    # only score
    def __init__(self):
        super(MeonNoDTSize, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.gdn1 = Gdn(8)
        self.max_pool2d1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.gdn2 = Gdn(16)
        self.max_pool2d2 = nn.MaxPool2d((2, 2))
        self.conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.gdn3 = Gdn(32)
        self.max_pool2d3 = nn.MaxPool2d((2, 2))
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.gdn4 = Gdn(64)
        self.max_pool2d4 = nn.MaxPool2d((2, 2))

        self.st2_fc1 = nn.Conv2d(64, 256, 1, stride=1, padding=0)
        self.st2_gdn1 = Gdn(256)
        self.st2_fc2 = nn.Conv2d(256, 1, 1, stride=1, padding=0)
        self.view = MyView(size=(25, -1))

class MeonNoDTWide(nn.Module):
    # only score
    def __init__(self):
        super(MeonNoDTWide, self).__init__()

        self.conv1 = nn.Conv2d(3, 2048, 7, stride=4, padding=4)
        self.gdn1 = Gdn(2048)
        self.fc1 = nn.Conv2d(2048 * 10 * 10, 128, 1, stride=1, padding=0)
        self.gdn2 = Gdn(128)
        self.fc2 = nn.Conv2d(128, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]

        x = self.conv1(x)
        x = self.gdn1(x)
        x = F.max_pool2d(x, (6, 6))
        x = x.view(batch_size, -1, 1, 1)
        x = self.fc1(x)
        y = self.gdn2(x)
        s = self.fc2(y)
        s = s.view(batch_size, -1)

        return s

class MeonNoDTCornia(nn.Module):
    # only score
    def __init__(self):
        super(MeonNoDTCornia, self).__init__()

        self.conv = nn.Conv2d(3, 10000, 7, stride=5, padding=1)    # 47x47
        self.fc = nn.Conv2d(20000, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]

        x = self.conv(x)
        x1 = F.max_pool2d(x, (47, 47))      # maxpooling
        x2 = -F.max_pool2d(-x, (47, 47))    # minpooling
        x = torch.cat((x1, x2), 1)
        x = self.fc(x)
        x = x.view(batch_size, -1)

        return x

class MeonNoDTCorniaSize(nn.Module):
    # only score
    def __init__(self):
        super(MeonNoDTCorniaSize, self).__init__()

        self.conv = nn.Conv2d(3, 10000, 7, stride=5, padding=1)    # 47x47
        self.max_pool2d = nn.MaxPool2d((47, 47))
        self.fc = nn.Conv2d(10000, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

class MeonNoDTWideSize(nn.Module):

    # only score
    def __init__(self):
        super(MeonNoDTWideSize, self).__init__()

        self.conv1 = nn.Conv2d(3, 2048, 7, stride=4, padding=4)
        self.gdn1 = Gdn(2048)
        self.max_pool2d = nn.MaxPool2d((6, 6))
        self.view1 = MyView(size=(25, -1, 1, 1))
        self.fc1 = nn.Conv2d(2048 * 10 * 10, 128, 1, stride=1, padding=0)
        self.gdn2 = Gdn(128)
        self.fc2 = nn.Conv2d(128, 1, 1, stride=1, padding=0)
        self.view2 = MyView(size=(25, -1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.max_pool2d(x)
        x = self.view1(x)
        x = self.fc1(x)
        y = self.gdn2(x)
        s = self.fc2(y)
        s = self.view1(s)

        return s

class Meon(nn.Module):
    # score + dist
    def __init__(self, output_channel):
        super(Meon, self).__init__()

        self.output_channel = output_channel

        # shared layer parameters
        self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.gdn1 = Gdn(8)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.gdn2 = Gdn(16)
        self.conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
       # self.batch3 = nn.BatchNorm2d(32)
        self.gdn3 = Gdn(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.batch4 = nn.BatchNorm2d(64)
        self.gdn4 = Gdn(64)


        # subtask 1 parameters
        self.st1_fc1 = nn.Conv2d(384, 256, 1, stride=1, padding=0) #384=6*64
        # self.st1_fc1 = F.dropout(self.st1_fc1, p=drop_rate, training=self.trainning)
        self.st1_gdn1 = Gdn(256)
        self.st1_fc2 = nn.Conv2d(256, self.output_channel, 1, stride=1, padding=0)

#        # subtask 2 parameters on 08252020 backup
        self.st2_fc1 = nn.Conv2d(384, 32, 1, stride=1, padding=0)#384=6*64
        self.st2_gdn1 = Gdn(32)
        self.st2_fc2 = nn.Conv2d(32, self.output_channel, 1, stride=1, padding=0)


        # init both subtasks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]
        feacon = []
        for i, proj_img in enumerate(x.chunk(6, 1)):  # divide x[12,6,3,235,235] to 6 [12,3,235,235]

            # share layer
            x = F.max_pool2d(self.gdn1(self.conv1(torch.squeeze(proj_img, 1).float())), (2, 2))
            x = F.max_pool2d(self.gdn2(self.conv2(x)), (2, 2))
            x = F.max_pool2d(self.gdn3(self.conv3(x)), (2, 2))
            x = F.max_pool2d(self.gdn4(self.batch4(self.conv4(x))), (2, 2))
            feacon.append(x)
           # feacon.append(x.unsqueeze(dim=1)) ##Max pool
        # subtask 1
        #feature_c = torch.cat(feacon, dim=1) ##Max pool
       # feature_max = torch.max(feature_c, dim=1)[0] ##Max pool
        y = self.st1_fc2(nn.functional.dropout(self.st1_gdn1(self.st1_fc1(torch.cat(feacon, dim=1))), p=0.5, training=True))
        #y = self.st1_fc2(nn.functional.dropout(self.st1_gdn1(self.st1_fc1(feature_max)), p=0.5, training=True))  ##Max pool
        y = y.view(batch_size, -1)

        # subtask 2
        p = F.softmax(y)
        p = p.view(batch_size, -1)
        s = self.st2_fc2(nn.functional.dropout(self.st2_gdn1(self.st2_fc1(torch.cat(feacon, dim=1))), p=0.5, training=True))
       # s = self.st2_fc2(nn.functional.dropout(self.st2_gdn1(self.st2_fc1(feature_max)), p=0.5, training=True))  ##Max pool
        s = s.view(batch_size, -1)


        g = torch.sum(p * s, dim=1)


        return y, g #fout


class MeonDT(nn.Module):
    # only dist
    def __init__(self, output_channel, clamp=False):
        super(MeonDT, self).__init__()

        # shared layer parameters
        self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.gdn1 = Gdn(8)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.gdn2 = Gdn(16)
        self.conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        # self.batch3 = nn.BatchNorm2d(32)
        self.gdn3 = Gdn(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.batch4 = nn.BatchNorm2d(64)
        self.gdn4 = Gdn(64)


        # subtask 1 parameters
        self.st1_fc1 = nn.Conv2d(384, 256, 1, stride=1, padding=0)##64*6=384
        self.st1_gdn1 = Gdn(256)
        self.st1_fc2 = nn.Conv2d(256, output_channel, 1, stride=1, padding=0)

        if clamp is True:
            print("===== Clamping the dist level between 0 and 17 =====")
            self.clamp = torch.clamp
        else:
            self.clamp = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]
        feacon = []
        for i, proj_img in enumerate(x.chunk(6,1)): # divide x[12,6,3,235,235] to 6 [12,3,235,235]
            # share layer
            x = F.max_pool2d(nn.functional.dropout(self.gdn1(self.conv1(torch.squeeze(proj_img, 1).float())), p=0.5, training=True), (2, 2))
            x = F.max_pool2d(nn.functional.dropout(self.gdn2(self.conv2(x)), p=0.5, training=True), (2, 2))
            x = F.max_pool2d(nn.functional.dropout(self.gdn3(self.conv3(x)), p=0.5, training=True), (2, 2))
            x = F.max_pool2d(nn.functional.dropout(self.gdn4(self.conv4(x)), p=0.5, training=True), (2, 2))
            feacon.append(x)
            #feacon.append(x.unsqueeze(dim=1))   ##Max pool
        # subtask 1
       # feature_c = torch.cat(feacon, dim=1) ##Max pool
       # feature_max = torch.max(feature_c, dim=1)[0] ##Max pool
        y = self.st1_fc2(nn.functional.dropout(self.st1_gdn1(self.st1_fc1(torch.cat(feacon, dim=1))), p=0.5, training=True))
       #y = self.st1_fc2(nn.functional.dropout(self.st1_gdn1(self.st1_fc1(feature_max)), p=0.5, training=True))  ##Max pool
        y = y.view(batch_size, -1)

        if self.clamp is not None:
            y = self.clamp(y, min=0, max=17)

        return y


class MeonSplit(nn.Module):
    # score + dist
    def __init__(self, output_channel):
        super(MeonSplit, self).__init__()

        self.output_channel = output_channel
        # no shared layer
        # subtask 1: dist
        with torch.no_grad():
            self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
            self.gdn1 = Gdn(8)
            self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
            self.gdn2 = Gdn(16)
            self.conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
            self.gdn3 = Gdn(32)
            self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
            self.gdn4 = Gdn(64)

            self.st1_fc1 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
            self.st1_gdn1 = Gdn(128)
            self.st1_fc2 = nn.Conv2d(
                128, self.output_channel, 1, stride=1, padding=0)

            self.st1_net = [self.conv1, self.gdn1, self.conv2, self.gdn2, self.conv3, self.gdn3,
                            self.conv4, self.gdn4, self.st1_fc1, self.st1_gdn1, self.st1_fc2]

            for layer in self.st1_net:
                for param in layer.parameters():
                    param.requires_grad = False

        # subtask 2: scsore
        self.st2_conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.st2_gdn1 = Gdn(8)
        self.st2_conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.st2_gdn2 = Gdn(16)
        self.st2_conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.st2_gdn3 = Gdn(32)
        self.st2_conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.st2_gdn4 = Gdn(64)

        self.st2_fc1 = nn.Conv2d(64, 256, 1, stride=1, padding=0)
        self.st2_gdn5 = Gdn(256)
        self.st2_fc2 = nn.Conv2d(
            256, self.output_channel, 1, stride=1, padding=0)

        # init both subtasks, subtask1 will be overide by model loading
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]

        # subtask 1: dist
        x1 = F.max_pool2d(self.gdn1(self.conv1(x)), (2, 2))
        x1 = F.max_pool2d(self.gdn2(self.conv2(x1)), (2, 2))
        x1 = F.max_pool2d(self.gdn3(self.conv3(x1)), (2, 2))
        x1 = F.max_pool2d(self.gdn4(self.conv4(x1)), (2, 2))

        y = self.st1_fc2(self.st1_gdn1(self.st1_fc1(x1)))
        y = y.view(batch_size, -1)

        # subtask 2: score
        x2 = F.max_pool2d(self.st2_gdn1(self.st2_conv1(x)), (2, 2))
        x2 = F.max_pool2d(self.st2_gdn2(self.st2_conv2(x2)), (2, 2))
        x2 = F.max_pool2d(self.st2_gdn3(self.st2_conv3(x2)), (2, 2))
        x2 = F.max_pool2d(self.st2_gdn4(self.st2_conv4(x2)), (2, 2))

        # fuse
        p = F.softmax(y)
        p = p.view(batch_size, -1)

        s = self.st2_fc2(self.st2_gdn5(self.st2_fc1(x2)))
        s = s.view(batch_size, -1)

        g = torch.sum(p * s, dim=1)

        return y, g


class MeonSplitRegression(nn.Module):
    # score + dist
    def __init__(self, clamp=False):
        super(MeonSplitRegression, self).__init__()

        # no shared layer

        # subtask 1: dist
        with torch.no_grad():
            self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
            self.gdn1 = Gdn(8)
            self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
            self.gdn2 = Gdn(16)
            self.conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
            self.gdn3 = Gdn(32)
            self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
            self.gdn4 = Gdn(64)

            self.st1_fc1 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
            self.st1_gdn1 = Gdn(128)
            self.st1_fc2 = nn.Conv2d(128, 4, 1, stride=1, padding=0)

            if clamp is True:
                print("===== Clamping the dist level between 0 and 17 =====")
                self.clamp = torch.clamp
            else:
                self.clamp = None

            self.st1_net = [self.conv1, self.gdn1, self.conv2, self.gdn2, self.conv3,
                            self.gdn3, self.conv4, self.gdn4, self.st1_fc1, self.st1_gdn1, self.st1_fc2]

            for layer in self.st1_net:
                for param in layer.parameters():
                    param.requires_grad = False

        # subtask 2: score
        self.st2_conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.st2_gdn1 = Gdn(8)
        self.st2_conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.st2_gdn2 = Gdn(16)
        self.st2_conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.st2_gdn3 = Gdn(32)
        self.st2_conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.st2_gdn4 = Gdn(64)

        self.st2_fc1 = nn.Conv2d(64, 256, 1, stride=1, padding=0)
        self.st2_gdn5 = Gdn(256)
        self.st2_fc2 = nn.Conv2d(256, 4, 1, stride=1, padding=0)

        # fuse layer
        self.st3_fc1 = nn.Conv2d(8, 32, 1, stride=1, padding=0)
        self.st3_gdn1 = Gdn(32)
        self.st3_fc2 = nn.Conv2d(32, 1, 1, stride=1, padding=0)

        # init both subtasks, subtask1 will be overide by model loading
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]

        # subtask 1: dist
        x1 = F.max_pool2d(self.gdn1(self.conv1(x)), (2, 2))
        x1 = F.max_pool2d(self.gdn2(self.conv2(x1)), (2, 2))
        x1 = F.max_pool2d(self.gdn3(self.conv3(x1)), (2, 2))
        x1 = F.max_pool2d(self.gdn4(self.conv4(x1)), (2, 2))

        y = self.st1_fc2(self.st1_gdn1(self.st1_fc1(x1)))
        # y = y.view(batch_size, -1)
        if self.clamp is not None:
            y = self.clamp(y, min=0, max=17)

        # subtask 2: score
        x2 = F.max_pool2d(self.st2_gdn1(self.st2_conv1(x)), (2, 2))
        x2 = F.max_pool2d(self.st2_gdn2(self.st2_conv2(x2)), (2, 2))
        x2 = F.max_pool2d(self.st2_gdn3(self.st2_conv3(x2)), (2, 2))
        x2 = F.max_pool2d(self.st2_gdn4(self.st2_conv4(x2)), (2, 2))
        s = self.st2_fc2(self.st2_gdn5(self.st2_fc1(x2)))
        # s = s.view(batch_size, -1)

        # fuse
        x3 = self.st3_fc1(torch.cat((s, y), 1))
        x3 = self.st3_gdn1(x3)
        g = self.st3_fc2(x3)

        # fuse
        # p = F.softmax(y)
        # p = p.view(batch_size, -1)

        # s = self.st2_fc2(self.st2_gdn5(self.st2_fc1(x2)))
        # s = s.view(batch_size, -1)

        # g = torch.sum(p * s, dim=1)

        return y, g


class MeonSplitRegressionUpdate(nn.Module):
    # score + dist
    def __init__(self, clamp=False):
        super(MeonSplitRegressionUpdate, self).__init__()
        super(MeonSplitRegressionUpdate, self).__init__()

        # no shared layer

        # subtask 1: dist
        self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.gdn1 = Gdn(8)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.gdn2 = Gdn(16)
        self.conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.gdn3 = Gdn(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.gdn4 = Gdn(64)

        self.st1_fc1 = nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.st1_gdn1 = Gdn(128)
        self.st1_fc2 = nn.Conv2d(128, 4, 1, stride=1, padding=0)

        if clamp is True:
            print("===== Clamping the dist level between 0 and 17 =====")
            self.clamp = torch.clamp
        else:
            self.clamp = None

        # subtask 2: score
        self.st2_conv1 = nn.Conv2d(18, 8, 5, stride=2, padding=2)
        self.st2_gdn1 = Gdn(8)
        self.st2_conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.st2_gdn2 = Gdn(16)
        self.st2_conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.st2_gdn3 = Gdn(32)
        self.st2_conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.st2_gdn4 = Gdn(64)

        self.st2_fc1 = nn.Conv2d(64, 256, 1, stride=1, padding=0)
        self.st2_gdn5 = Gdn(256)
        self.st2_fc2 = nn.Conv2d(256, 4, 1, stride=1, padding=0)

        # fuse layer
        self.st3_fc1 = nn.Conv2d(8, 32, 1, stride=1, padding=0)
        self.st3_gdn1 = Gdn(32)
        self.st3_fc2 = nn.Conv2d(32, 1, 1, stride=1, padding=0)

        # init both subtasks, subtask1 will be overide by model loading
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]

        # subtask 1: dist
        x1 = F.max_pool2d(self.gdn1(self.conv1(x)), (2, 2))
        x1 = F.max_pool2d(self.gdn2(self.conv2(x1)), (2, 2))
        x1 = F.max_pool2d(self.gdn3(self.conv3(x1)), (2, 2))
        x1 = F.max_pool2d(self.gdn4(self.conv4(x1)), (2, 2))

        y = self.st1_fc2(self.st1_gdn1(self.st1_fc1(x1)))
        # y = y.view(batch_size, -1)
        if self.clamp is not None:
            y = self.clamp(y, min=0, max=17)

        # subtask 2: score
        x2 = F.max_pool2d(self.st2_gdn1(self.st2_conv1(x)), (2, 2))
        x2 = F.max_pool2d(self.st2_gdn2(self.st2_conv2(x2)), (2, 2))
        x2 = F.max_pool2d(self.st2_gdn3(self.st2_conv3(x2)), (2, 2))
        x2 = F.max_pool2d(self.st2_gdn4(self.st2_conv4(x2)), (2, 2))
        s = self.st2_fc2(self.st2_gdn5(self.st2_fc1(x2)))
        # s = s.view(batch_size, -1)

        # fuse
        x3 = self.st3_fc1(torch.cat((s, y), 1))
        x3 = self.st3_gdn1(x3)
        g = self.st3_fc2(x3)

        # fuse
        # p = F.softmax(y)
        # p = p.view(batch_size, -1)

        # s = self.st2_fc2(self.st2_gdn5(self.st2_fc1(x2)))
        # s = s.view(batch_size, -1)

        # g = torch.sum(p * s, dim=1)

        return y, g
