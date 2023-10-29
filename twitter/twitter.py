import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
from dataset import FeatureDataset
from mymodel import SimilarityModule, DetectionModule
from clip import CLIP
from sklearn.metrics import accuracy_score, classification_report,recall_score
import math
import torch.nn.functional as F
import random

# Configs
DEVICE = "cuda:0"
NUM_WORKER = 1
BATCH_SIZE = 64
LR = 1e-3
L2 = 0 #1e-5
NUM_EPOCH = 50
seed = 825
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)

def prepare_data(text, image, label):
    nr_index = [i for i, l in enumerate(label) if l == 0]
    if len(nr_index) < 2:
        nr_index.append(np.random.randint(len(label)))
        nr_index.append(np.random.randint(len(label)))
    text_nr = text[nr_index]
    image_nr = image[nr_index]
    fixed_text = copy.deepcopy(text_nr)
    matched_image = copy.deepcopy(image_nr)
    unmatched_image = copy.deepcopy(image_nr).roll(shifts=3, dims=0)
    return fixed_text, matched_image, unmatched_image
    
def get_soft_label(label):
    soft_label = []
    bs = len(label)
    for i, l in enumerate(label):
        if l == 0:
            true_label = [0 for j in range(bs)]
            true_label[i] = 1
            soft_label.append(true_label)
        else:
            soft_label.append([1./bs for _ in range(bs)])
    return torch.tensor(soft_label)

def soft_loss(input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]

def train():
    # ---  Load Config  ---
    device = torch.device(DEVICE)
    num_workers = NUM_WORKER
    batch_size = BATCH_SIZE
    lr = LR
    l2 = L2
    num_epoch = NUM_EPOCH
    
    # ---  Load Data  ---
    dataset_dir = '../twitter/'
    train_set = FeatureDataset(
        "{}/train_text_with_label.npz".format(dataset_dir),
        "{}/train_image_with_label.npz".format(dataset_dir)
    )
    test_set = FeatureDataset(
        "{}/test_text_with_label.npz".format(dataset_dir),
        "{}/test_image_with_label.npz".format(dataset_dir)
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # ---  Build Model & Trainer  ---
    similarity_module = SimilarityModule()  
    similarity_module.to(device)
    detection_module = DetectionModule()  
    detection_module.to(device)
    clip_module = CLIP(64)
    clip_module.to(device)
    loss_func_similarity = torch.nn.CosineEmbeddingLoss(margin=0.2)
    loss_func_clip = torch.nn.CrossEntropyLoss()
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_func_skl = torch.nn.KLDivLoss(reduction='batchmean')
    optim_task_similarity = torch.optim.Adam(
        similarity_module.parameters(), lr=lr, weight_decay=l2
    )  # also called task1
    optimizer_task_clip = torch.optim.AdamW(
        clip_module.parameters(), lr=0.001, weight_decay=5e-4)
    optim_task_detection = torch.optim.Adam(
        detection_module.parameters(), lr=lr, weight_decay=l2
    )  # also called task2

    # ---  Model Training  ---
    loss_similarity_total = 0
    loss_detection_total = 0
    best_acc = 0
    step = 0
    for epoch in range(num_epoch):

        similarity_module.train()
        clip_module.train()
        corrects_pre_similarity = 0
        corrects_pre_detection = 0
        loss_similarity_total = 0
        loss_clip_total = 0
        loss_detection_total = 0
        similarity_count = 0
        detection_count = 0

        for i, (text, image, label) in tqdm(enumerate(train_loader)):
            batch_size = text.shape[0]
            text = text.to(device)
            image = image.to(device)
            label = label.to(device)

            fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
            fixed_text.to(device)
            matched_image.to(device)
            unmatched_image.to(device)


            # ---  TASK1 Similarity  ---

            text_aligned_match, image_aligned_match, pred_similarity_match = similarity_module(fixed_text, matched_image)
            text_aligned_unmatch, image_aligned_unmatch, pred_similarity_unmatch = similarity_module(fixed_text, unmatched_image)
            # 1:positive/match 0:negative/unmatch
            similarity_pred = torch.cat([pred_similarity_match.argmax(1), pred_similarity_unmatch.argmax(1)], dim=0)
            similarity_label_0 = torch.cat([torch.ones(pred_similarity_match.shape[0]), torch.zeros(pred_similarity_unmatch.shape[0])], dim=0).to(device)
            similarity_label_1 = torch.cat([torch.ones(pred_similarity_match.shape[0]), -1 * torch.ones(pred_similarity_unmatch.shape[0])], dim=0).to(device)
            text_aligned_4_task1 = torch.cat([text_aligned_match, text_aligned_unmatch], dim=0) # N*64 cat N*64 -> 2N * 64
            image_aligned_4_task1 = torch.cat([image_aligned_match, image_aligned_unmatch], dim=0)
            loss_similarity = loss_func_similarity(text_aligned_4_task1, image_aligned_4_task1, similarity_label_1)

            optim_task_similarity.zero_grad()
            loss_similarity.backward()
            optim_task_similarity.step()

            corrects_pre_similarity += similarity_pred.eq(similarity_label_0).sum().item()

            # ---  TASK1 CLIP  ---
            image_aligned, text_aligned = clip_module(image, text)
            logits = torch.matmul(image_aligned, text_aligned.T) * math.exp(0.07)
            labels = torch.arange(text.size(0))
            labels = labels.to(device)
            text_sim, image_sim, _ = similarity_module(text, image) # N* 64
            soft_label = torch.matmul(image_sim, text_sim.T) * math.exp(0.07)
            soft_label = soft_label.to(device)

            optimizer_task_clip.zero_grad()
            loss_clip_i = loss_func_clip(logits, labels)
            loss_clip_t = loss_func_clip(logits.T, labels)
            loss_clip = (loss_clip_i + loss_clip_t) / 2.
            image_loss = soft_loss(logits, F.softmax(soft_label,1))
            caption_loss = soft_loss(logits.T, F.softmax(soft_label.T,1))
            loss_soft = (image_loss + caption_loss) / 2.
            all_loss = loss_clip + 0.2 * loss_soft
            all_loss.backward()
            step += 1
            optimizer_task_clip.step()
            

            # ---  TASK2 Detection  ---
            image_aligned, text_aligned = clip_module(image, text) # N* 64
            pre_detection, attention_score, skl_score = detection_module(text, image, text_aligned, image_aligned)
            loss_detection = loss_func_detection(pre_detection, label) + 0.5 * loss_func_skl(attention_score, skl_score)

            optim_task_detection.zero_grad()
            loss_detection.backward()
            optim_task_detection.step()
            
            pre_label_detection = pre_detection.argmax(1)
            corrects_pre_detection += pre_label_detection.eq(label.view_as(pre_label_detection)).sum().item()
            
            # ---  Record  ---
            loss_clip_total += loss_soft.item()
            loss_detection_total += loss_detection.item() * text.shape[0]
            similarity_count += (2 * fixed_text.shape[0] * 2)
            detection_count += text.shape[0]

        loss_detection_train = loss_detection_total / detection_count
        acc_detection_train = corrects_pre_detection / detection_count

        # ---  Test  ---

        acc_detection_test, loss_detection_test, cm_detection, cr_detection = test(clip_module, detection_module, test_loader)

        # ---  Output  ---

        print('---  TASK1 CLIP  ---')
        print('[Epoch: {}], losses: {}'.format(epoch, loss_clip_total / step))


        print('---  TASK2 Detection  ---')
        if acc_detection_test > best_acc:
            best_acc = acc_detection_test
            print(cr_detection)
            torch.save(clip_module.state_dict(), "best_clip_module_wosen.pth")
            torch.save(detection_module.state_dict(), "best_detection_model_wosen.pth") 
        print(
            "EPOCH = %d \n acc_detection_train = %.3f \n acc_detection_test = %.3f \n  best_acc = %.3f \n loss_detection_train = %.3f \n loss_detection_test = %.3f \n" %
            (epoch + 1, acc_detection_train, acc_detection_test, best_acc, loss_detection_train, loss_detection_test)
        )

        print('---  TASK2 Detection Confusion Matrix  ---')
        print('{}\n'.format(cm_detection))


def test(clip_module, detection_module, test_loader):
    clip_module.eval()
    detection_module.eval()

    device = torch.device(DEVICE)
    loss_func_detection = torch.nn.CrossEntropyLoss()
    loss_func_clip = torch.nn.CrossEntropyLoss()
    loss_func_skl = torch.nn.KLDivLoss(reduction='batchmean')

    similarity_count = 0
    detection_count = 0
    loss_similarity_total = 0
    loss_clip_total = 0
    loss_detection_total = 0
    similarity_label_all = []
    detection_label_all = []
    similarity_pre_label_all = []
    detection_pre_label_all = []

    with torch.no_grad():
        for i, (text, image, label) in enumerate(test_loader):
            batch_size = text.shape[0]
            text = text.to(device)
            image = image.to(device)
            label = label.to(device)
            
            fixed_text, matched_image, unmatched_image = prepare_data(text, image, label)
            fixed_text.to(device)
            matched_image.to(device)
            unmatched_image.to(device)

             # ---  TASK1 CLIP  ---
            image_aligned, text_aligned = clip_module(image, text)

            logits = torch.matmul(image_aligned, text_aligned.T) * math.exp(0.07)
            labels = torch.arange(text.size(0))
            labels = labels.to(device)
            loss_clip = loss_func_clip(logits, labels)

            # ---  TASK2 Detection  ---
            pre_detection, attention_score, skl_score = detection_module(text, image, text_aligned, image_aligned)
            loss_detection = loss_func_detection(pre_detection, label) + 0.2 * loss_func_skl(attention_score, skl_score)
            pre_label_detection = pre_detection.argmax(1)

            # ---  Record  ---

            loss_clip_total += loss_clip.item()
            loss_detection_total += loss_detection.item() * text.shape[0]
            # similarity_count += (fixed_text.shape[0] * 2)
            detection_count += text.shape[0]

            # similarity_pre_label_all.append(similarity_pred.detach().cpu().numpy())
            detection_pre_label_all.append(pre_label_detection.detach().cpu().numpy())
            # similarity_label_all.append(similarity_label_0.detach().cpu().numpy())
            detection_label_all.append(label.detach().cpu().numpy())

        # loss_similarity_test = loss_similarity_total / similarity_count
        loss_detection_test = loss_detection_total / detection_count

        # similarity_pre_label_all = np.concatenate(similarity_pre_label_all, 0)
        detection_pre_label_all = np.concatenate(detection_pre_label_all, 0)
        # similarity_label_all = np.concatenate(similarity_label_all, 0)
        detection_label_all = np.concatenate(detection_label_all, 0)

        # acc_similarity_test = accuracy_score(similarity_pre_label_all, similarity_label_all)
        acc_detection_test = accuracy_score(detection_pre_label_all, detection_label_all)
        # cm_similarity = confusion_matrix(similarity_pre_label_all, similarity_label_all)
        cm_detection = confusion_matrix(detection_pre_label_all, detection_label_all)
        cr_detection = classification_report(detection_pre_label_all, detection_label_all, target_names=['Real News','Fake News'], digits = 3)


    return acc_detection_test, loss_detection_test, cm_detection, cr_detection


if __name__ == "__main__":
    train()

