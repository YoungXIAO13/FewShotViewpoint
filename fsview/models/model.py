import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import resnet


class FC(nn.Module):
    '''
        A wrapper for a 2D pytorch conv layer.
    '''
    def __init__(self, in_channels, out_channels, bn=False, dropout=False):
        super(FC, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.dropout = nn.Dropout() if dropout else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


# ============================================================================ #
#                             Proposed network                                 #
# ============================================================================ #
class ShapeEncoderMV(nn.Module):
    """Shape Encoder using rendering images under multiple views

        Arguments:
        feature_dim: output feature dimension for each rendering image
        channels: 3 for normal rendering image, 4 for normal map with depth map, and 3*12 channels for concatenating

        Return:
        A tensor of size NxC, where N is the batch size and C is the feature_dim
    """
    def __init__(self, feature_dim=256, channels=3):
        super(ShapeEncoderMV, self).__init__()
        self.render_encoder = resnet.resnet18(input_channel=channels, num_classes=feature_dim)

    def forward(self, renders):
        # reshape render images from dimension N*K*C*H*W to (N*K)*C*H*W
        N, K, C, H, W = renders.size()
        renders = renders.view(N*K, C, H, W)

        # pass the encoder and reshape render features from dimension (N*K)*D1 to N*(K*D1)
        render_feature = self.render_encoder(renders)
        render_feature = render_feature.view(N, -1)
        return render_feature


class ShapeEncoderPC(nn.Module):
    """Shape Encoder using point cloud
        Arguments:
            feature_dim: output feature dimension for the point cloud
        Return:
            A tensor of size NxC, where N is the batch size and C is the feature_dim
    """

    def __init__(self, feature_dim=1024):
        super(ShapeEncoderPC, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, feature_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(feature_dim)

    def forward(self, shapes):
        x = F.relu(self.bn1(self.conv1(shapes)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(shapes.size(0), -1)
        return x


class PoseEstimator(nn.Module):
    """Pose estimator using image feature with shape feature

        Arguments:
        img_feature_dim: output feature dimension for image
        shape_feature_dim: output feature dimension for shape
        shape: shape representation in PointCloud or MultiView
        channels: channel number for multi-view encoder

        Return:
        Three angle bin classification probability with a delta value regression for each bin
    """
    def __init__(self, view_num=12, img_feature_dim=1024, shape_feature_dim=256, channels=3,
                 azi_classes=24, ele_classes=12, inp_classes=24, shape=None):
        super(PoseEstimator, self).__init__()

        # RGB image encoder
        self.img_encoder = resnet.resnet18(num_classes=img_feature_dim)

        # 3D shape encoder
        if shape == 'pointcloud':
            self.shape_encoder = ShapeEncoderPC(feature_dim=shape_feature_dim)
        else:
            self.shape_encoder = ShapeEncoderMV(feature_dim=shape_feature_dim, channels=channels)

        shape_feature_dim = shape_feature_dim * view_num if shape != 'pointcloud' else shape_feature_dim
        self.shape_fc = nn.Sequential(nn.Linear(shape_feature_dim, img_feature_dim),
                                      nn.BatchNorm1d(img_feature_dim), nn.ReLU(inplace=True))

        self.diff_fc = nn.Sequential(nn.Linear(img_feature_dim, int(img_feature_dim / 2)),
                                     nn.BatchNorm1d(int(img_feature_dim / 2)), nn.ReLU(inplace=True))
        self.corr_fc = nn.Sequential(nn.Linear(img_feature_dim, int(img_feature_dim / 2)),
                                     nn.BatchNorm1d(int(img_feature_dim / 2)), nn.ReLU(inplace=True))

        self.compress = nn.Sequential(nn.Linear(img_feature_dim * 2, 800), nn.BatchNorm1d(800), nn.ReLU(inplace=True),
                                      nn.Linear(800, 400), nn.BatchNorm1d(400), nn.ReLU(inplace=True),
                                      nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU(inplace=True))

        # classifier branch
        self.fc_cls_azi = nn.Linear(200, azi_classes)
        self.fc_cls_ele = nn.Linear(200, ele_classes)
        self.fc_cls_inp = nn.Linear(200, inp_classes)

        # regression branch
        self.fc_reg_azi = nn.Linear(200, azi_classes)
        self.fc_reg_ele = nn.Linear(200, ele_classes)
        self.fc_reg_inp = nn.Linear(200, inp_classes)

    def forward(self, im, shape, get_mean_att=False, mean_class_att=None):
        # pass the image through image encoder
        img_feature = self.img_encoder(im)  # B*C

        if mean_class_att is not None:
            shape_feature = mean_class_att.view(img_feature.shape[0], -1)
        else:
            # pass the shape through shape encoder
            shape_feature = self.shape_encoder(shape)
            shape_feature = self.shape_fc(shape_feature)  # B*C

        if get_mean_att:
            return shape_feature

        # combine the 2D img feature with 3D model feature using TDID
        global_feature = torch.cat(
            (self.corr_fc(img_feature * shape_feature),
             self.diff_fc(img_feature - shape_feature),
             img_feature), dim=1)
        x = self.compress(global_feature)

        cls_azi = self.fc_cls_azi(x)
        cls_ele = self.fc_cls_ele(x)
        cls_inp = self.fc_cls_inp(x)

        reg_azi = self.fc_reg_azi(x)
        reg_ele = self.fc_reg_ele(x)
        reg_inp = self.fc_reg_inp(x)
        return [cls_azi, cls_ele, cls_inp, reg_azi, reg_ele, reg_inp]


if __name__ == '__main__':
    print('test model')

    sim_im = Variable(torch.rand(4, 3, 224, 224)).cuda()
    sim_shape = Variable(torch.rand(4, 3, 2500)).cuda()

    model = PoseEstimator(shape='pointcloud')
    model.cuda()

    cls_azi, cls_ele, cls_inp, reg_azi, reg_ele, reg_inp = model(sim_im, sim_shape)
    print(cls_azi.size(), cls_ele.size(), cls_inp.size(), reg_azi.size(), reg_ele.size(), reg_inp.size())
