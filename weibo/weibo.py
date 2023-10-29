from cgitb import text
from email.mime import image
from pydoc import cli
from string import digits
from mymodel import Multi_Model, SimilarityModule
#import matplotlib.pyplot as plt
import pickle 
from PIL import Image
import re
import os 
import copy
from torchvision.transforms.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import BertModel, BertConfig, BertTokenizer, AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import warnings
from torch import LongTensor
import torch 
import time 
from sklearn.metrics import accuracy_score, classification_report,recall_score
from clip import CLIP
import math
warnings.filterwarnings('ignore')

seed = 3407
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 

class Config():
    def __init__(self):
        self.batch_size = 64
        self.epochs = 50
        self.bert_path = "./bert_ch_model"
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.lr = 1e-3
        self.l2 = 1e-5

class FakeNewsDataset(Dataset):
    def __init__(self,input_three,event,image,label) :
        self.event = LongTensor(list(event)) 
        self.image = torch.FloatTensor([np.array(i) for i in image]) 
        self.label = LongTensor(list(label))
        self.input_three = list()
        self.input_three.append( LongTensor(input_three[0]))
        self.input_three.append(LongTensor(input_three[1]))
        self.input_three.append(LongTensor(input_three[2]))
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self,idx):
        return self.input_three[0][idx],self.input_three[1][idx],self.input_three[2][idx],self.image[idx],self.event[idx],self.label[idx]

def cleanSST(string):
    string = re.sub(u"[，。 :,.；|-“”——_/nbsp+&;@、《》～（）())#O！：【】]", "", string)
    return string.strip().lower()

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 'Total parameters: {}, Trainable parameters: {}'.format(total_num, trainable_num)

def evaluate(clip_module, multi_model,vali_dataloader,device):
    clip_module.eval()
    multi_model.eval()
    val_true, val_pred = [], []
    with torch.no_grad():
        for index,(batch_text0,batch_text1,batch_text2,batch_image,batch_event,batch_label) in enumerate(vali_dataloader):
            batch_text0 = batch_text0.to(device)
            batch_text1 = batch_text1.to(device)
            batch_text2 = batch_text2.to(device)
            batch_image = batch_image.to(device)
            batch_event = batch_event.to(device)
            batch_label = batch_label.to(device)
            image_aligned, text_aligned = clip_module(batch_text0,batch_text1,batch_text2,batch_image) # N* 64
            y_pred, a_s, s_s = multi_model(batch_text0,batch_text1,batch_text2,batch_image, text_aligned, image_aligned)
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(batch_label.squeeze().cpu().numpy().tolist())
    print(classification_report(val_true, val_pred, target_names=['Real News','Fake News'], digits = 3))
    return accuracy_score(val_true, val_pred)

def prepare_data(text0,text1,text2, image, label):
    nr_index = [i for i, l in enumerate(label) if l == 0]
    if len(nr_index) < 2:
        nr_index.append(np.random.randint(len(label)))
        nr_index.append(np.random.randint(len(label)))
    text0_nr = text0[nr_index]
    text1_nr = text1[nr_index]
    text2_nr = text2[nr_index]
    image_nr = image[nr_index]
    fixed_text0 = copy.deepcopy(text0_nr)
    fixed_text1 = copy.deepcopy(text1_nr)
    fixed_text2 = copy.deepcopy(text2_nr)
    matched_image = copy.deepcopy(image_nr)
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)
    return fixed_text0, fixed_text1,fixed_text2,matched_image, unmatched_image

def soft_loss(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]
    
train_acc_vector = []
vali_acc_vector = []
def train_val_test():
    #train , test , validate 
    config = Config()
    train_dataset = pickle.load(open('./pickles/new_train_dataset.pkl','rb'))
    test_dataset = pickle.load(open('./pickles/new_test_dataset.pkl','rb'))
    print(len(train_dataset),len(test_dataset))
    print("dataset dump ok")

    train_loader = DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False)
    print('process data  Loader success')

    # ---  Build Model & Trainer  ---
    similarity_module = SimilarityModule(config.bert_path)  
    similarity_module.to(config.device)
    clip_module = CLIP(64, config.bert_path)
    clip_module.to(config.device)
    loss_func_similarity = torch.nn.CosineEmbeddingLoss(margin=0.2)
    loss_func_clip = torch.nn.CrossEntropyLoss()
    optim_task_similarity = torch.optim.Adam(
        similarity_module.parameters(), lr=config.lr, weight_decay=config.l2
    )  # also called task1
    optimizer_task_clip = torch.optim.AdamW(
        clip_module.parameters(), lr=0.001, weight_decay=5e-4)
    bert_multi_model = Multi_Model(config.bert_path)
    bert_multi_model.to(config.device)
    print("model init ok")
    print(get_parameter_number(bert_multi_model))
    #optimizer = torch.optim.Adam(bert_multi_model.parameters(),lr=config.lr, weight_decay=config.l2)
    optimizer = AdamW(bert_multi_model.parameters(), lr=2e-5, weight_decay=1e-4) 
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                           num_training_steps=config.epochs*len(train_loader))
    #criterion = nn.CrossEntropyLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_func_skl = torch.nn.KLDivLoss(reduction='batchmean')
    
    
    #official train
    best_acc = 0.0
    step = 0
    for i in range(config.epochs):
        start = time.time()
        similarity_module.train()
        clip_module.train()
        bert_multi_model.train()
        print("***** Running training epoch {} *****".format(i+1))
        train_loss_sum = 0.0
        corrects_pre_similarity = 0
        label_predict = []
        label_epoch = []
        loss_similarity_total = 0
        similarity_count = 0
        loss_clip_total = 0

        for index,(batch_text0,batch_text1,batch_text2,batch_image,batch_event,batch_label) in enumerate(train_loader):
            batch_text0 = Variable(batch_text0.to(config.device))
            batch_text1 = Variable(batch_text1.to(config.device))
            batch_text2 = Variable(batch_text2.to(config.device))
            batch_image = Variable(batch_image.to(config.device))
            batch_event = Variable(batch_event.to(config.device))
            batch_label = Variable(batch_label.to(config.device))

            fixed_text0,fixed_text1,fixed_text2, matched_image, unmatched_image = prepare_data(batch_text0,batch_text1,batch_text2,batch_image,batch_label)
            fixed_text0 = Variable(fixed_text0.to(config.device))
            fixed_text1 = Variable(fixed_text1.to(config.device))
            fixed_text2 = Variable(fixed_text2.to(config.device))
            matched_image = Variable(matched_image.to(config.device))
            unmatched_image = Variable(unmatched_image.to(config.device))

            #print("training")

            # ---  TASK1 Similarity  ---

            text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text0,fixed_text1,fixed_text2,matched_image)
            text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text0,fixed_text1,fixed_text2, unmatched_image)
            # 1:positive/match 0:negative/unmatch
            similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
            similarity_label_0 = torch.cat([torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])], dim=0).to(config.device)
            similarity_label_1 = torch.cat([torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])], dim=0).to(config.device)

            text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0) 
            image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
            

            # ---  TASK1 CLIP  ---
            text_sim, image_sim, _ = similarity_module(batch_text0,batch_text1,batch_text2,batch_image) 
            soft_label = torch.matmul(image_sim, text_sim.T) * math.exp(0.07)
            soft_label = soft_label.to(config.device)
            labels = torch.arange(batch_image.size(0))
            labels = labels.to(config.device)

            optim_task_similarity.zero_grad()
            loss_similarity = loss_func_similarity(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)
            loss_similarity.backward()
            optim_task_similarity.step()

            # corrects_pre_similarity += similarity_pred.eq(similarity_label_0).sum().item()

            image_aligned, text_aligned = clip_module(batch_text0,batch_text1,batch_text2,batch_image)
            logits = torch.matmul(image_aligned, text_aligned.T) * math.exp(0.07)
            labels = torch.arange(batch_image.size(0))
            labels = labels.to(config.device)
            text_sim, image_sim, _ = similarity_module(batch_text0,batch_text1,batch_text2,batch_image) 
            soft_label = torch.matmul(image_sim, text_sim.T) * math.exp(0.07)
            soft_label = soft_label.to(config.device)
            
            optimizer_task_clip.zero_grad()
            loss_clip_i = loss_func_clip(logits, labels)
            loss_clip_t = loss_func_clip(logits.T, labels)
            loss_clip = (loss_clip_i + loss_clip_t) / 2.
            image_loss = soft_loss(logits, F.softmax(soft_label,1))
            caption_loss = soft_loss(logits.T, F.softmax(soft_label.T,1))
            loss_soft = (image_loss + caption_loss) / 2.
            all_loss = loss_clip  + 0.2 * loss_soft
            all_loss.backward()
            step += 1
            optimizer_task_clip.step()
            # ---  TASK2 Detection  ---
            image_aligned, text_aligned = clip_module(batch_text0,batch_text1,batch_text2,batch_image) # N* 64
            label_pred, attention_score, skl_score = bert_multi_model(batch_text0,batch_text1,batch_text2,batch_image, text_aligned, image_aligned)
            loss = loss_func_detection(label_pred,batch_label) + 0.5 * loss_func_skl(attention_score, skl_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()   
          
            label_predict.extend(torch.argmax(label_pred, dim=1).cpu().numpy().tolist())
            label_epoch.extend(batch_label.squeeze().cpu().numpy().tolist())
            
            train_loss_sum += loss.item()
            
            if (index + 1) % (len(train_loader)//5) == 0:   
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                          i+1, index+1, len(train_loader), train_loss_sum/(index+1), time.time() - start))
        epoch_acc = accuracy_score(label_predict,label_epoch)
        train_acc_vector.append(epoch_acc)
       
        print("Train Accuracy:{} Recall:{}".format(epoch_acc,recall_score(label_predict,label_epoch)))
       
        acc = evaluate(clip_module, bert_multi_model, test_loader, config.device)
        vali_acc_vector.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(clip_module.state_dict(), "best_clip_module_wosen.pth") 
            torch.save(bert_multi_model.state_dict(), "best_multi_bert_model_wosen.pth") 
        print('---  TASK1 CLIP  ---')
        print('[Epoch: {}], losses: {}'.format(i, loss_clip_total / step))

        
        print("current acc is {:.4f}, best acc is {:.4f}".format(acc, best_acc))
        print("time costed = {}s \n".format(round(time.time() - start, 5)))
    clip_module = CLIP(64, config.bert_path)
    clip_module.load_state_dict(torch.load("best_clip_module_wosen.pth"))
    clip_module.to(config.device)
    test_model = Multi_Model(config.bert_path)
    test_model.load_state_dict(torch.load("best_multi_bert_model_wosen.pth"))
    test_model.to(config.device)
    evaluate(clip_module,test_model,test_loader,config.device)
              
def main():
    train_val_test()

if __name__ == '__main__':
    main()
   