import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import math

import torchvision.models as tvmodels
import torchvision.transforms.functional as TF

class Identify_Feature_Feeding_Block(nn.Module):
    def __init__(self, output_dim, z_id_size=512):
        super(Identify_Feature_Feeding_Block, self).__init__()
        self.output_dim = output_dim
        self.z_id_size = z_id_size
        self.fc = nn.Linear(self.z_id_size, self.output_dim)

    def forward(self, z_id):
        out = self.fc(z_id)
        out = torch.unsqueeze(out, 2)
        out = torch.unsqueeze(out, 3)
        first_1028 = out[:, 0:int(self.output_dim / 2), :, :]
        second_1024 = out[:, int(self.output_dim / 2):self.output_dim, :, :]
        return first_1028, second_1024


class Operation_Unit(nn.Module):
    def __init__(self, channel, identity_feature_out_dim, identity_feature_in_dim=512, act_out=True):
        super(Operation_Unit, self).__init__()
        self.channel = channel
        self.act_out = act_out
        self.id_feature_out_dim = identity_feature_out_dim
        self.id_feature_in_dim = identity_feature_in_dim
        self.Conv1 = nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=0)
        self.activation = nn.ReLU()
        self.IFF = Identify_Feature_Feeding_Block(self.id_feature_out_dim, z_id_size=self.id_feature_in_dim)

    def forward(self, input_feature, id_feature):
        idf0_0, idf0_1 = self.IFF(id_feature)
        x0 = F.pad(input_feature, (1, 1, 1, 1, 0, 0), mode='reflect')
        x0 = self.Conv1(x0)
        x0 = torch.sub(x0, torch.mean(x0, (2, 3), keepdim=True))

        x1 = torch.mul(x0, x0)
        x1 = torch.mean(x1, (2, 3), keepdim=True)
        x1 = torch.add(x1, 9.99999993922529e-9)
        x1 = torch.sqrt(x1)
        x1 = torch.div(1.0, x1)
        x2 = torch.mul(x0, x1)
        x2 = torch.mul(idf0_0, x2)
        x2 = torch.add(x2, idf0_1)
        if self.act_out == True:
            return self.activation(x2)
        else:
            return x2


class Feature_Fusion_Block(nn.Module):
    def __init__(self, channel, identity_feature_out_dim, identity_feature_in_dim):
        super(Feature_Fusion_Block, self).__init__()

        self.channel = channel
        self.identity_feature_out_dim = identity_feature_out_dim
        self.identity_feature_in_dim = identity_feature_in_dim

        self.OP1 = Operation_Unit(self.channel, self.identity_feature_out_dim,
                                  identity_feature_in_dim=self.identity_feature_in_dim)
        self.OP2 = Operation_Unit(self.channel, self.identity_feature_out_dim,
                                  identity_feature_in_dim=self.identity_feature_in_dim, act_out=False)

    def forward(self, input_feature, id_feature):
        x = self.OP1(input_feature, id_feature)
        x = self.OP2(x, id_feature)
        return input_feature + x

class Encoder_noBNIN(nn.Module):
    def __init__(self, id_feature_dim):
        super(Encoder_noBNIN, self).__init__()
        self.id_feature_dim = id_feature_dim
        self.Encoder_channel = [3, 128, 256, 512, 1024]
        self.Encoder_kernel_size = [7, 3, 3, 3]
        self.pading_scale = [0, 1, 1, 1]
        self.stride_scale = [1, 1, 2, 2]
        self.Encoder = nn.ModuleDict({f'layer_{i}': nn.Sequential(
            nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i + 1], kernel_size=self.Encoder_kernel_size[i],
                      stride=self.stride_scale[i], padding=self.pading_scale[i]),
            nn.LeakyReLU(0.2)
        ) for i in range(4)})

        # self.FFB1 = Feature_Fusion_Block(1024,2048)

        self.fusion_module = nn.ModuleDict(
            {f'fusion_layer_{i}': Feature_Fusion_Block(1024, 2048, 512)
             for i in range(6)})

    def forward(self, x, id_feature):
        x = F.pad(x, (3, 3, 3, 3, 0, 0), mode='reflect')
        for i in range(4):
            x = self.Encoder[f'layer_{i}'](x)
        # x = self.FFB1(input_feature=x,id_feature=id_feature)

        for i in range(6):
            x = self.fusion_module[f'fusion_layer_{i}'](x, id_feature)
        return x

class Encoder(nn.Module):
    def __init__(self, id_feature_dim):
        super(Encoder, self).__init__()
        self.id_feature_dim = id_feature_dim
        self.Encoder_channel = [3, 128, 256, 512, 1024]
        self.Encoder_kernel_size = [7, 3, 3, 3]
        self.pading_scale = [0, 1, 1, 1]
        self.stride_scale = [1, 1, 2, 2]
        self.Encoder = nn.ModuleDict({f'layer_{i}': nn.Sequential(
            nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i + 1], kernel_size=self.Encoder_kernel_size[i],
                      stride=self.stride_scale[i], padding=self.pading_scale[i]),
            nn.LeakyReLU(0.2)
        ) for i in range(4)})

        # self.FFB1 = Feature_Fusion_Block(1024,2048)

        self.fusion_module = nn.ModuleDict(
            {f'fusion_layer_{i}': Feature_Fusion_Block(1024, 2048, 512)
             for i in range(6)})

    def forward(self, x, id_feature):
        x = F.pad(x, (3, 3, 3, 3, 0, 0), mode='reflect')
        for i in range(4):
            x = self.Encoder[f'layer_{i}'](x)
        # x = self.FFB1(input_feature=x,id_feature=id_feature)

        for i in range(6):
            x = self.fusion_module[f'fusion_layer_{i}'](x, id_feature)
            if i < 5:  # No need to apply norm after the last fusion layer
                bn_layer = getattr(self, f'batch_norm_{i}')
                in_layer = getattr(self, f'instance_norm_{i}')
                x = bn_layer(x)
                x = in_layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=1023, out_channels=3):
        super(Decoder, self).__init__()

        self.in_channel = in_channels
        self.out_channel = out_channels
        # self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.Upsample = nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear')

        self.Conv1 = nn.Conv2d(self.in_channel, 512, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.Conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.Conv4 = nn.Conv2d(128, self.out_channel, kernel_size=7, stride=1, padding=0)

        self.Activation_LeakyRelu = nn.LeakyReLU(0.2)
        self.Activation_Tanh = nn.Tanh()

    def forward(self, visual_feature,get_latent=False):
        out_latent = []
        x = self.Upsample(visual_feature)
        out_latent.append(x)
        # pdb.set_trace()
        x = self.Activation_LeakyRelu(self.Conv1(x))
        out_latent.append(x)
        x = self.Upsample(x)
        x = self.Activation_LeakyRelu(self.Conv2(x))
        out_latent.append(x)
        x = self.Activation_LeakyRelu(self.Conv3(x))
        out_latent.append(x)
        x = F.pad(x, (3, 3, 3, 3, 0, 0), mode='reflect')
        x = self.Conv4(x)
        x = self.Activation_Tanh(x)
        x = torch.add(x, 1.0)
        x = torch.div(x, 2.0)
        if get_latent:
            return x, out_latent
        else:   
            return x


class Decoder_Enlarge(nn.Module):
    def __init__(self, in_channels=1023, out_channels=3):
        super(Decoder_Enlarge, self).__init__()

        self.in_channel = in_channels
        self.out_channel = out_channels
        # self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.Upsample = nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear')

        self.Conv1 = nn.Conv2d(self.in_channel, 512, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.Conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.Conv4_new = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.Conv5_new = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.Conv6_new = nn.Conv2d(128, self.out_channel, kernel_size=5, stride=1, padding=0)

        self.Activation_LeakyRelu = nn.LeakyReLU(0.2)
        self.Activation_Tanh = nn.Tanh()

    def forward(self, visual_feature,get_latent=False):
        out_latent = []
        x = self.Upsample(visual_feature)
        out_latent.append(x)
        # pdb.set_trace()
        x = self.Activation_LeakyRelu(self.Conv1(x))
        out_latent.append(x)
        x = self.Upsample(x)
        x = self.Activation_LeakyRelu(self.Conv2(x))
        out_latent.append(x)
        x = self.Activation_LeakyRelu(self.Conv3(x))
        out_latent.append(x)
        x = self.Activation_LeakyRelu(self.Conv4_new(x))
        out_latent.append(x)
        #x = F.pad(x, (3, 3, 3, 3, 0, 0), mode='reflect')
        x = self.Activation_LeakyRelu(self.Conv5_new(x))
        out_latent.append(x)
        x = F.pad(x, (3, 3, 3, 3, 0, 0), mode='reflect')
        #pdb.set_trace()
        x = self.Conv6_new(x)
        x = self.Activation_Tanh(x)
        
        x = torch.add(x, 1.0)
        x = torch.div(x, 2.0)
        #pdb.set_trace()
        if get_latent:
            return x, out_latent
        else:   
            return x


class Decoder_Enlarge_for_onnx(nn.Module):
    def __init__(self, in_channels=1023, out_channels=3):
        super(Decoder_Enlarge_for_onnx, self).__init__()

        self.in_channel = in_channels
        self.out_channel = out_channels
        # self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.Upsample = nn.Upsample(scale_factor=2, align_corners=False, mode='bilinear')

        self.Conv1 = nn.Conv2d(self.in_channel, 512, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.Conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.Conv4_new = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.Conv5_new = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.Conv6_new = nn.Conv2d(128, self.out_channel, kernel_size=5, stride=1, padding=0)

        self.Activation_LeakyRelu = nn.LeakyReLU(0.2)
        self.Activation_Tanh = nn.Tanh()

    def forward(self, visual_feature):
        x = self.Upsample(visual_feature)
        # pdb.set_trace()
        x = self.Activation_LeakyRelu(self.Conv1(x))
        x = self.Upsample(x)
        x = self.Activation_LeakyRelu(self.Conv2(x))
        x = self.Activation_LeakyRelu(self.Conv3(x))
        x = self.Activation_LeakyRelu(self.Conv4_new(x))
        #x = F.pad(x, (3, 3, 3, 3, 0, 0), mode='reflect')
        x = self.Activation_LeakyRelu(self.Conv5_new(x))
        x = F.pad(x, (3, 3, 3, 3, 0, 0), mode='reflect')
        #pdb.set_trace()
        x = self.Conv6_new(x)
        x = self.Activation_Tanh(x)
        
        x = torch.add(x, 1.0)
        x = torch.div(x, 2.0)
        #pdb.set_trace()
        return x


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class Discriminator(nn.Module):
    def __init__(self, img_size=384, max_conv_dim=1024):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        num_domains = 1
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]
        repeat_num = int(np.log2(img_size)) - 1
        for _l in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            if _l%2==0:
                blocks += [ResBlk(dim_in, dim_out, downsample=False)]
            else:
                blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out
        
        # repeat_num = int(np.log2(img_size)) - 2 # This is the original code
        # for _ in range(repeat_num):
        #     dim_out = min(dim_in * 2, max_conv_dim)
        #     blocks += [ResBlk(dim_in, dim_out, downsample=True)]
        #     dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        #blocks += [nn.Conv2d(dim_out, dim_out, 4, 1, 0)] # Original (Adv for single img)
        blocks += [nn.Conv2d(dim_out, dim_out, 1, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.main = nn.Sequential(*blocks)


    def get_features(self, x):
        feature_list = []
        for _depth,block in enumerate(self.main):
            x = block(x)
            if _depth > 7 and _depth < 10:
                feature_list.append(x)
        #out = x
        #out = out.view(out.size(0), -1)  # (batch, num_domains)
        # pdb.set_trace()
        return feature_list

    def forward(self, x):
        # out = self.main(x)
        # pdb.set_trace()
        for block in self.main:
            x = block(x)
        #pdb.set_trace()
        out = x
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out
    
    


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss based on fixed VGG-16 feature activations.

    Args
    ----
    layer_ids : tuple[int]
        Indices (0-based, inclusive) of VGG *conv* blocks whose outputs
        will be compared.  Defaults to (3, 8, 15, 22)   # conv1_2, conv2_2, conv3_3, conv4_3
    weights : tuple[float] | None
        Per-layer weights.  If None, equal weighting is used.
    criterion : str
        'l1' or 'l2' – distance metric for feature differences.
    resize : bool
        If True, input tensors are resized to 224×224 to match VGG training size.
    """

    def __init__(
        self,
        layer_ids=(3, 8, 15, 22),
        weights=None,
        criterion: str = "l1",
        resize: bool = True,
    ):
        super().__init__()

        # --- load VGG-16 pretrained on ImageNet ---
        vgg = tvmodels.vgg16(weights=tvmodels.VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()                # inference mode
        for p in vgg.parameters():  # freeze
            p.requires_grad = False
        self.vgg = vgg

        # --- layers & weights ---
        self.layer_ids = layer_ids
        self.weights = (
            torch.ones(len(layer_ids)) if weights is None else torch.tensor(weights)
        )

        if criterion == "l1":
            self.distance = nn.L1Loss(reduction="none")
        elif criterion == "l2":
            self.distance = nn.MSELoss(reduction="none")
        else:
            raise ValueError("criterion must be 'l1' or 'l2'")

        self.resize = resize

        # ImageNet channel-wise normalisation parameters
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    # -------------------------------------------------------------

    #@torch.no_grad()  # inference only
    def _extract_feats(self, x: torch.Tensor):
        feats, out = [], x
        if self.resize:
            out = nn.functional.interpolate(out, size=(224, 224), mode="bilinear", align_corners=False)
        out = (out - self.mean) / self.std  # normalise to ImageNet stats

        for i, layer in enumerate(self.vgg):
            out = layer(out)
            if i in self.layer_ids:
                feats.append(out)
            if i >= max(self.layer_ids):
                break
        return feats

    # -------------------------------------------------------------

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between `pred` and `target`.

        Both tensors must be in **range [0, 1]** and shape `(N, 3, H, W)`.
        """
        assert pred.shape == target.shape, "Input tensors must have the same shape."
        #pdb.set_trace()
        feats_pred = self._extract_feats(pred)
        feats_tgt = self._extract_feats(target)

        losses = []
        for f_p, f_t, w in zip(feats_pred, feats_tgt, self.weights):
            diff = self.distance(f_p, f_t).mean(dim=[1, 2, 3])  # per-image scalar
            losses.append(w * diff)

        # mean over layers, then batch
        loss = torch.stack(losses, dim=0).sum(dim=0).mean()
        return loss