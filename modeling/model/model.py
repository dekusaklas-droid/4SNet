import torch
import torch.nn as nn
from torch.nn import init

from AFF import  AFF
from resnet import resnet50, resnet18
import torch.nn.functional as F



class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)

        self.visible = model_v
        original_conv = self.visible.conv1

        self.visible.conv1 = AFF(
            3, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
            kernel_num=2,
            use_ksm_local=True,
            ksm_only_kernel_att=False,
            param_reduction=1.0,
            use_fbm_if_k_in=[7],
            fbm_cfg={
                'k_list': [2, 4],
                'lowfreq_att': False,
                'act': 'sigmoid',
                'spatial_group': 8
            }
        )
        with torch.no_grad():
            self.visible.conv1.weight.copy_(original_conv.weight)

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)

        self.thermal = model_t
        original_conv = self.thermal.conv1

        self.thermal.conv1 = AFF(
            3, 64,
            kernel_size=7, stride=2, padding=3, bias=False,
            kernel_num=2,
            use_ksm_local=True,
            ksm_only_kernel_att=False,
            param_reduction=1.0,
            use_fbm_if_k_in=[7],
            fbm_cfg={
                'k_list': [2, 4],
                'lowfreq_att': True,
                'act': 'sigmoid',
                'spatial_group': 8
            }
        )
        with torch.no_grad():
            self.thermal.conv1.weight.copy_(original_conv.weight)

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x



class CNL(nn.Module):
    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(high_dim), )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                                   nn.BatchNorm2d(self.high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h

        return z


class PNL(nn.Module):
    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim), )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z


class MFA_block(nn.Module):
    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z


def pairwise_distance(query_features, gallery_features):
    x = query_features
    y = gallery_features
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


class SSM(nn.Module):
    def __init__(self, in_channels, freq_bands=[0.1, 0.3, 0.5, 0.7]):
        super(SSM, self).__init__()
        self.freq_bands = freq_bands
        self.in_channels = in_channels
        self.num_bands = len(freq_bands)

        self.conv_processors = nn.ModuleList()
        for i in range(self.num_bands):
            self.conv_processors.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels, 1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.fusion_network = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.band_weights = nn.Parameter(torch.ones(self.num_bands) / self.num_bands)
        self.dropout = nn.Dropout(p=0.01)

    def create_frequency_bands(self, x_fft, device):
        B, C, H, W = x_fft.shape

        freq_h = torch.fft.fftfreq(H, device=device)
        freq_w = torch.fft.fftfreq(W, device=device)
        freq_grid_h, freq_grid_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
        freq_magnitude = torch.sqrt(freq_grid_h ** 2 + freq_grid_w ** 2)

        masks = []
        freq_min = 0.0

        for freq_max in self.freq_bands:
            mask = (freq_magnitude >= freq_min) & (freq_magnitude < freq_max)
            mask = mask.float()
            mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
            masks.append(mask)
            freq_min = freq_max

        return masks

    def apply_frequency_masks(self, x_fft, masks):
        spatial_features = []

        for i, mask in enumerate(masks):
            masked_fft = x_fft * mask
            spatial_feature = torch.fft.ifft2(masked_fft, dim=(-2, -1)).real
            spatial_features.append(spatial_feature)

        return spatial_features

    def process_spatial_features(self, spatial_features):
        processed_features = []

        for i, spatial_feat in enumerate(spatial_features):
            processed_feat = self.conv_processors[i](spatial_feat)
            processed_features.append(processed_feat)

        return processed_features

    def apply_fusion_network(self, batch_concat, original_batch_size):
        fused_features = self.fusion_network(batch_concat)

        C, H, W = fused_features.shape[1], fused_features.shape[2], fused_features.shape[3]
        reshaped_features = fused_features.view(self.num_bands, original_batch_size, C, H, W)
        reshaped_features = reshaped_features.permute(1, 0, 2, 3, 4)

        weights = F.softmax(self.band_weights, dim=0)

        output = torch.zeros(original_batch_size, C, H, W, device=fused_features.device)
        for i in range(self.num_bands):
            output += weights[i] * reshaped_features[:, i, :, :, :]

        return output

    def forward(self, x, return_freq_features=False):
        B, C, H, W = x.shape
        device = x.device

        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        masks = self.create_frequency_bands(x_fft, device)
        spatial_features = self.apply_frequency_masks(x_fft, masks)
        processed_features = self.process_spatial_features(spatial_features)

        freq_features_dict = {}
        if return_freq_features:
            freq_names = ['ultra_low', 'low', 'mid', 'high']
            for i, feat in enumerate(processed_features):
                compressed_feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                freq_features_dict[freq_names[i]] = compressed_feat

        batch_concat = torch.cat((x, processed_features[0], processed_features[1]), 0)
        output = self.dropout(batch_concat)

        return output


class ChannelUpsampler(nn.Module):
    def __init__(self, in_channels=64, out_channels=512, out_size=None):
        super(ChannelUpsampler, self).__init__()
        self.out_size = out_size
        self.upsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.upsample.apply(weights_init_kaiming)

    def forward(self, x):
        if self.out_size is not None:
            x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        x = self.upsample(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, dataset, arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        self.dataset = dataset
        if self.dataset == 'regdb':
            pool_dim = 1024
            self.ssm = SSM(512)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = MFA_block(512, 256, 1)
        else:
            pool_dim = 2048
            self.ssm = SSM(1024)
            self.MFA1 = MFA_block(256, 64, 0)
            self.MFA2 = MFA_block(512, 256, 1)
            self.MFA3 = MFA_block(1024, 512, 1)

        
        self.cu1 = ChannelUpsampler(64, 512, (48, 18))
        self.cu2 = ChannelUpsampler(256, 512, (48, 18))
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.l2norm = Normalize(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bie_alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)

        elif modal == 1:
            x = self.visible_module(x1)

        elif modal == 2:
            x = self.thermal_module(x2)

        x_ = x
        x = self.base_resnet.base.layer1(x_)
        x_ = self.MFA1(x, x_)
        x = self.base_resnet.base.layer2(x_)
        x_ = self.MFA2(x, x_)

        if self.dataset == 'regdb':
            x_ = self.ssm(x_)
            x = self.base_resnet.base.layer3(x_)
        else:
            x = self.base_resnet.base.layer3(x_)
            x_ = self.MFA3(x, x_)
            x_ = self.ssm(x_)
            x = self.base_resnet.base.layer4(x_)

        xp = self.avgpool(x)
        x_pool = xp.view(xp.size(0), xp.size(1))
        feat = self.bottleneck(x_pool)

        if self.training:
            xps = xp.view(xp.size(0), xp.size(1), xp.size(2)).permute(0, 2, 1)
            xp1, xp2, xp3 = torch.chunk(xps, 3, 0)
            xpss = torch.cat((xp2, xp3), 1)
            loss_ort = torch.triu(torch.bmm(xpss, xpss.permute(0, 2, 1)), diagonal=1).sum() / (xp.size(0))
            return x_pool, self.classifier(feat), loss_ort
        else:
            return self.l2norm(x_pool), self.l2norm(feat)
