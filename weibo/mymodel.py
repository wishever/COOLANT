from distutils.command.config import config
import pandas as pd 
import numpy as np 
import json, time 
from tqdm import tqdm 
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings
import re
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.utils.data import Dataset
import math
import random
from re import X
# from random import random
import torch
import torch.nn as nn
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from torchvision.models import resnet18
from SENet import Network

class Encoder(nn.Module):
    def __init__(self, z_dim=2):
        super(Encoder, self).__init__()
        self.z_dim = z_dim
        # Vanilla MLP
        self.net = nn.Sequential(
            nn.Linear(64, 64),
            #nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, z_dim * 2),
        )

    def forward(self, x):
        params = self.net(x) # N * 4
        mu, sigma = params[:, :self.z_dim], params[:, self.z_dim:] # N * 2 
        sigma = softplus(sigma) + 1e-7 
        return Independent(Normal(loc=mu, scale=sigma), 1)

class AmbiguityLearning(nn.Module):
    def __init__(self):
        super(AmbiguityLearning, self).__init__()
        # self.encoding = EncodingPart()
        self.encoder_text = Encoder()
        self.encoder_image = Encoder()

    def forward(self, text_encoding, image_encoding):
        p_z1_given_text = self.encoder_text(text_encoding) 
        p_z2_given_image = self.encoder_image(image_encoding)
        z1 = p_z1_given_text.rsample() 
        z2 = p_z2_given_image.rsample()
        kl_1_2 = p_z1_given_text.log_prob(z1) - p_z2_given_image.log_prob(z1)
        kl_2_1 = p_z2_given_image.log_prob(z2) - p_z1_given_text.log_prob(z2)
        skl = (kl_1_2 + kl_2_1)/ 2.
        skl = torch.sigmoid(skl)
        return skl

class UnimodalDetection(nn.Module):
    def __init__(self, shared_dim=128, prime_dim = 16):
        super(UnimodalDetection, self).__init__()
        self.text_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )
        self.image_uni = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, prime_dim),
            nn.BatchNorm1d(prime_dim),
            nn.ReLU()
        )

    def forward(self, text_encoding, image_encoding):
        text_prime = self.text_uni(text_encoding)
        image_prime = self.image_uni(image_encoding)
        return text_prime, image_prime

class CrossModule4Batch(nn.Module):
    def __init__(self, text_in_dim=64, image_in_dim=64, corre_out_dim=64):
        super(CrossModule4Batch, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.corre_dim = 64
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.c_specific_2 = nn.Sequential(
            nn.Linear(self.corre_dim, corre_out_dim),
            nn.BatchNorm1d(corre_out_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_in = text.unsqueeze(2) 
        image_in = image.unsqueeze(1) 
        corre_dim = text.shape[1]
        similarity = torch.matmul(text_in, image_in) / math.sqrt(corre_dim)
        correlation = self.softmax(similarity)
        correlation_p = self.pooling(correlation).squeeze()
        correlation_out = self.c_specific_2(correlation_p)
        return correlation_out 

class FastCNN(nn.Module):
    # a CNN-based altertative approach of bert for text encoding
    def __init__(self, channel=32, kernel_size=(1, 2, 4, 8)):
        super(FastCNN, self).__init__()
        self.fast_cnn = nn.ModuleList()
        for kernel in kernel_size:
            self.fast_cnn.append(
                nn.Sequential(
                    nn.Conv1d(768, channel, kernel_size=kernel),
                    nn.BatchNorm1d(channel),
                    nn.ReLU(),
                    nn.Dropout(),
                    nn.AdaptiveMaxPool1d(1) 
                )
            )

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x_out = []
        for module in self.fast_cnn:
            x_out.append(module(x).squeeze(-1)) 
        x_out = torch.cat(x_out, 1)
        return x_out 

class EncodingPart(nn.Module):
    def __init__(
        self,
        cnn_channel=32,
        cnn_kernel_size=(1, 2, 4, 8),
        shared_image_dim=128,
        shared_text_dim=128
    ):
        super(EncodingPart, self).__init__()
        self.shared_text_encoding = FastCNN(
            channel=cnn_channel,
            kernel_size=cnn_kernel_size
        )
        self.shared_text_linear = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, shared_text_dim),
            nn.BatchNorm1d(shared_text_dim),
            nn.ReLU()
        )
        self.shared_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_image_dim),
            nn.BatchNorm1d(shared_image_dim),
            nn.ReLU()
        )

    def forward(self, text, image):
        text_encoding = self.shared_text_encoding(text) 
        text_shared = self.shared_text_linear(text_encoding)
        image_shared = self.shared_image(image) 
        return text_shared, image_shared

class SimilarityModule(nn.Module):
    def __init__(self, bert_path, text_fea = 512,image_fea = 512, shared_dim=128, sim_dim=64):
        super(SimilarityModule, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False
        #self.fc = nn.Linear(self.config.hidden_size, text_fea)
        
        #res_34 = torchvision.models.resnet34(pretrained=True)
        #for param in res_34.parameters():
        #    param.requires_grad = False
        #fc_features = res_34.fc.out_features
        #self.res34 = res_34
        #self.image_fc2 =  nn.Linear(fc_features, image_fea)
        
        self.img_backbone = resnet18(pretrained=True)
        self.img_model = nn.ModuleList([
            self.img_backbone.conv1,
            self.img_backbone.bn1,
            self.img_backbone.relu,
            self.img_backbone.layer1,
            self.img_backbone.layer2,
            self.img_backbone.layer3,
            self.img_backbone.layer4
        ])
        self.img_model = nn.Sequential(*self.img_model)
        for param in self.img_model.parameters():
            param.requires_grad = False
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.img_fc = nn.Linear(self.img_backbone.inplanes, image_fea)

        self.encoding = EncodingPart()
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.sim_classifier_dim = sim_dim * 2
        self.sim_classifier = nn.Sequential(
            nn.BatchNorm1d(self.sim_classifier_dim),
            nn.Linear(self.sim_classifier_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, input_ids, attention_mask, token_type_ids, img):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)     
        text = outputs[0]  
        n_batch = img.size(0)
        img_out = self.img_model(img)
        img_out = self.avg_pool(img_out)
        img_out = img_out.view(n_batch, -1)
        img_out = self.img_fc(img_out)
        img_out = F.normalize(img_out, p=2, dim=-1)
      
        text_encoding, image_encoding = self.encoding(text, img_out)
        text_aligned = self.text_aligner(text_encoding) 
        image_aligned = self.image_aligner(image_encoding)
        sim_feature = torch.cat([text_aligned, image_aligned], 1)
        pred_similarity = self.sim_classifier(sim_feature) 
        return text_aligned, image_aligned, pred_similarity

class Multi_Model(nn.Module):
    def __init__(self, bert_path, feature_dim=64+16+16, h_dim=64, text_fea = 200, image_fea=512):
        super(Multi_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        #for param in self.bert.parameters():
        #    param.requires_grad = False    
        print("pre trained model")
        self.text_fc1 = nn.Linear(self.config.hidden_size, 128)
        
        self.cnn = nn.Sequential(
            nn.Conv2d( 3, 1,kernel_size=5,stride=2,padding=2),# 1 * 112*112
            nn.ReLU(),
            nn.MaxPool2d(2),#1*56*56
            nn.Conv2d(1,1,kernel_size=5,stride = 2 ,padding = 0),#1*26*26
            nn.ReLU(),
        )
        self.image_fc = nn.Sequential(
            nn.Linear(1*26*26, image_fea),
            nn.Linear(image_fea,128)
        )
    
        self.cross_module = CrossModule4Batch()
        self.encoding = EncodingPart()
        self.ambiguity_module = AmbiguityLearning()
        self.uni_repre = UnimodalDetection()
       
        self.uni_se = UnimodalDetection(prime_dim=64)
        self.senet = Network(64, 128, 24, 3)
        self.classifier_corre = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(h_dim, 2)
        )
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,img=None, text_aligned=None, image_aligned=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
       
        text = outputs[1]  
       
        text = self.text_fc1(text)
        
        image = self.cnn(img)
        img_out = self.image_fc(image.view(image.size(0),-1))
        
        #text_prime, image_prime = self.encoding(text, img_out)
        text_prime, image_prime = self.uni_repre(text, img_out)
        text_se, image_se = self.uni_se(text, img_out)
        correlation = self.cross_module(text_aligned, image_aligned)


        text_se, image_se, corre_se = text_se.unsqueeze(-1), image_se.unsqueeze(-1), correlation.unsqueeze(-1) 
        attention_score = self.senet(torch.cat([text_se, image_se, corre_se], -1)) 

        text_final = text_prime * attention_score[:,0].unsqueeze(1)
        img_final = image_prime * attention_score[:,1].unsqueeze(1) 
        corre_final = correlation * attention_score[:,2].unsqueeze(1)
        final_corre = torch.cat([text_final, img_final, corre_final], 1) 
        pre_label = self.classifier_corre(final_corre)

        skl = self.ambiguity_module(text_aligned, image_aligned)
        weight_uni = (1-skl).unsqueeze(1)
        weight_corre = skl.unsqueeze(1) 
        skl_score = torch.cat([weight_uni, weight_uni, weight_corre], 1)
        
        return pre_label, attention_score, skl_score,final_corre