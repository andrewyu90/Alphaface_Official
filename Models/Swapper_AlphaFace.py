import torch
import torch.nn.functional as F
import pdb

import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch import nn
from Models.Swapper_Units_AdIN import Encoder, Discriminator, Decoder, Encoder_noBNIN, VGGPerceptualLoss
from Models.arcface_resnet import resnet50
# from Model.MultiScaleDiscriminator import MultiscaleDiscriminator
from Models.iresnet import iresnet100
from backbones import get_model
import numpy as np

# from Model.loss import GANLoss, AEI_Loss
batch_size = 1


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        x = x - self.mean
        x = x / self.std
        return x

class ResNet_ID_encoder(pl.LightningModule):
    def __init__(self, ema_path):
        super(ResNet_ID_encoder, self).__init__()
        self.resnet = resnet50()
        #pdb.set_trace()

        self.ema = nn.Linear(512, 512)
        self.ema.weight.data = torch.nn.Parameter(torch.from_numpy(np.load(ema_path)).permute(1,0)).cuda()
        self.ema.bias.data.fill_(0.0) 
        #self.ema = torch.from_numpy(np.load(ema_path)).cuda()

    def processing_ema_only(self,latent):
        x = self.ema(latent)
        # x = torch.matmul(x,self.ema)
        out = torch.div(x, torch.linalg.norm(x, dim=1, keepdim=True))
        return out

    def forward(self, inputs):
        x = self.resnet(inputs)
        x = self.ema(x)
        #x = torch.matmul(x,self.ema)
        out = torch.div(x, torch.linalg.norm(x, dim=1, keepdim=True))
        return out

class Swapper(nn.Module): #class Swapper_Enlarge(pl.LightningModule):
    def __init__(self, source_dim,exist_BN=True):
        super(Swapper, self).__init__()
        self.source_dim = source_dim
        if exist_BN:
            self.E = Encoder(self.source_dim)
        else:
            self.E = Encoder_noBNIN(self.source_dim)
        self.G = Decoder(1024, 3)
        self.dis = None
        self.train_adv = None


    def forward(self, target, source,get_latent=False):
        # source = np.dot(source, self.ema)
        # source /= np.linalg.norm(source)
        output = self.E(target, source)
        if get_latent==True:
            output,latent = self.G(output,get_latent=get_latent)
            return output,latent 
        else:
            output = self.G(output)
        return output

    def forward(self, target, source,get_latent=False):
        # source = np.dot(source, self.ema)
        # source /= np.linalg.norm(source)
        output = self.E(target, source)
        if get_latent==True:
            output,latent = self.G(output,get_latent=get_latent)
            return output,latent 
        else:
            output = self.G(output)
        return output
    


class AlphaFace(pl.LightningModule):
    def __init__(self, Swapper, id_encoder,fine_tune=False):
        super(AlphaFace, self).__init__()
        self.Swapper = Swapper.cuda()
        #self.Id_encoder = None
        self.fine_tune = fine_tune
        self.Id_encoder = id_encoder.cuda()

        self.dis = None
        self.train_adv = False
        
        self.feats_extractor = None

    def Prepareing_adversrial_learning(self,img_size,max_conv_size):
        self.dis = Discriminator(img_size=img_size, max_conv_dim=max_conv_size).cuda()
        self.train_adv = True
        
    def Preparing_VGG_percet_loss(self):
        self.feats_extractor = VGGPerceptualLoss(layer_ids=[3, 8, 15, 22]).cuda()
        

    def set_grads(self):
        swapper_param_list = list(self.Swapper.parameters())

        for param in swapper_param_list:
            param.requires_grad = True

        
        idc_param_list = list(self.Id_encoder.parameters())
        for param in idc_param_list:
            param.requires_grad = False

        # if self.fine_tune==False:
        #     idc_param_list = list(self.Id_encoder.parameters())
        #     for param in idc_param_list:
        #         param.requires_grad = False

        if self.train_adv==True:
            adv_param_list = list(self.dis.parameters())
            for param in adv_param_list:
                param.requires_grad = True

    def save(self, fname, step):
        print('Saving checkpoint into %s...' % fname)
        PATH_Swapper = fname + '_swapper_%d' % (step) + '.pt'
        PATH_IDC = fname + '_idc_%d' % (step) + '.pt'
        PATH_DIS = fname + '_dis_%d' % (step) + '.pt'

        PATH_Swapper_last = fname + '_swapper_last' + '.pt'
        PATH_IDC_last = fname + '_idc_last' + '.pt'
        PATH_DIS_last = fname + '_dis_last' + '.pt'

        # Save the mask module
        torch.save(self.Swapper.state_dict(), PATH_Swapper)
        torch.save(self.Swapper.state_dict(), PATH_Swapper_last)
        torch.save(self.ID_encoder.state_dict(), PATH_IDC)
        torch.save(self.ID_encoder.state_dict(), PATH_IDC_last)

        # Save discriminator
        if self.train_adv==True:
            torch.save(self.discriminator.state_dict(), PATH_DIS)
            torch.save(self.discriminator.state_dict(), PATH_DIS_last)

        # Save the mask module


    def load(self, fname, step=None):
        print('Loading checkpoint from %s...' % fname)
        if step == None:
            PATH_Swapper = fname + '_swapper_last' + '.pt'
            PATH_IDC = fname + '_idc_last' + '.pt'
            PATH_DIS = fname + '_dis_last' + '.pt'
        else:
            PATH_Swapper = fname + '_swapper_%d' % (step) + '.pt'
            PATH_IDC = fname + '_idc_%d' % (step) + '.pt'
            PATH_DIS = fname + '_dis_%d' % (step) + '.pt'

        # Save the mask module
        self.Swapper.load_state_dict(torch.load(PATH_Swapper))
        # Save discriminator
        self.dis.load_state_dict(torch.load(PATH_DIS))
        if self.fine_tune==False:
            self.Id_encoder.load_state_dict(torch.load(PATH_IDC))

    def get_id_code(self,source):
        return self.Id_encoder(source).detach()


    def forward(self, target, source,get_latent=False):
        #if it's the fine-tunning process, only identity code (512D) will be provided, so, no need to use Id_encoder
        if self.fine_tune==False:
            source = self.Id_encoder(source)
        #Else
        output = self.Swapper(target, source,get_latent=get_latent)
        return output


    def add_batch_instant_norm2swapper(self):
        bn_layer = nn.BatchNorm2d(1024)  # BatchNorm after fusion
        in_layer = nn.InstanceNorm2d(1024, affine=True)  # InstanceNorm after fusion
        #name, module = self.Swapper.E.Encoder['layer_3'].items()
        self.Swapper.E.Encoder['layer_3']  = nn.Sequential(
                self.Swapper.E.Encoder['layer_3'] ,  # Original Feature_Fusion_Block
                bn_layer,  # BatchNorm
                in_layer,  # InstanceNorm
            )
        #pdb.set_trace()
        for i in range(5):  # Since we need norm layers between 6 fusion layers, we only need 5 sets of norms
            out_channels = 1024  # Assuming fusion layer output size is 2048
            # Directly add BatchNorm and InstanceNorm as attributes to the model
            setattr(self.Swapper.E, f'batch_norm_{i}', nn.BatchNorm2d(out_channels))
            setattr(self.Swapper.E, f'instance_norm_{i}', nn.InstanceNorm2d(out_channels, affine=True))



def build_AlphaFace(config=None,fine_tune=False,adv_train=True,new_id_model=False):
    #Swapper
    
    swapper = Swapper(512)
    print('Loading pre-trained model for the ID encoder')
    if new_id_model==False:
        ID_encoder = ResNet_ID_encoder(ema_path='./Models/emp.npy')
        ID_encoder.resnet.load_state_dict(torch.load('./Models/arcface_w600k_r50_pytorch.pt', map_location=torch.device('cuda')))
    else:
        weight = torch.load(config.id_network_path)
        ID_encoder = get_model(config.id_network, dropout=0, fp16=False).cuda()
        ID_encoder.load_state_dict(weight)
        
    #Framework
    DPG = AlphaFace(swapper, ID_encoder, fine_tune=fine_tune)
    if adv_train==True:
        DPG.Prepareing_adversrial_learning(img_size=256,max_conv_size=512)
        DPG.Preparing_VGG_percet_loss()
    return DPG

if __name__ == '__main__':
    build_models()

