import os
import sys
import random
import argparse
import numpy as np
import random
import json
#---------------------------------------------
import pickle
from model.clip import clip
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#---------------------------------------------
import torch
from utils.config import _C as cfg
from model import *
import timm

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="", help="path to config file")
parser.add_argument("--noise_mode", default=None)
parser.add_argument("--noise_ratio", default=None)
parser.add_argument("--gpuid", default=None)
parser.add_argument("--backbone", default=None)

args = parser.parse_args()
cfg.defrost()
cfg.merge_from_file(args.cfg)
if args.noise_mode is not None:
    cfg.noise_mode = args.noise_mode
if args.noise_ratio is not None:
    cfg.noise_ratio = float(args.noise_ratio)
if args.gpuid is not None:
    cfg.gpuid = int(args.gpuid)
if args.backbone is not None:
    cfg.backbone = args.backbone

def set_seed():
    torch.cuda.set_device(cfg.gpuid)
    seed = cfg.seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# Train
def train(epoch, dataloader, pseudo_labels, point_filter):
    model.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    correct, total = 0, 0
    
    for batch_idx, (inputs, targets, _, index) in enumerate(dataloader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # Replace noisy targets with pseudo-labels
        targets_new = targets.clone()  # Start with a copy of the targets
        for i, current_idx in enumerate(index):
            if total_clean_idx[current_idx] == 0:  # If it's noisy
                # Replace the target with the pseudo-label (already computed)
                if pseudo_labels[current_idx] == -1 or len(pseudo_labels) <= current_idx:
                  print(f"Abort condition met for index: {current_idx}")
                  sys.exit("Aborting program due to pseudo-label condition.")
                targets_new[i] = pseudo_labels[current_idx]  # Use the corresponding pseudo-label
        
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        loss_per_sample = criterion(outputs, targets_new)
        loss = loss_per_sample[point_filter[index]].mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += targets.size(0)
        correct += predicted.eq(targets).cpu().sum().item()

        sys.stdout.write('\r')
        sys.stdout.write('Epoch [%3d/%3d] Iter[%3d/%3d]\t loss: %.3f'
                %( epoch, cfg.epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

    return 100.*correct/total

# Test
def test(epoch, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()
    acc = 100. * correct / total
    print("\n| Test Epoch #%d\t Accuracy: %.2f\n" %(epoch, acc))
    return acc

@torch.no_grad()
def pseudo_label(dataloader, model, train_dataset):
    model.eval()

    # To store the count of total noisy labels
    count=10557
    
    # Initialize pseudo_labels with -1 for all samples
    point_filter = [1] * len(dataloader.dataset) 
    pseudo_labels = [-1] * len(dataloader.dataset)

    point_filter = np.array(point_filter)

    # # Iterate through the dataloader
    # for batch_idx, (inputs, targets, _, index) in enumerate(dataloader):
    #     inputs = inputs.to(device)

    #     for j, current_idx in enumerate(index):
    #         # Only process noisy samples (where total_clean_idx[current_idx] == 0)
    #         if total_clean_idx[current_idx] == 0:
    #             count+=1
                
    #             # Encode image features for noisy samples
    #             image = inputs[j].unsqueeze(0).cuda()  # Add batch dimension
    #             image_features = model.encode_image(image)
    #             image_features /= image_features.norm(dim=-1, keepdim=True)

    #             # Encode candidate labels as text prompts
    #             text_prompts = candidate_labels
    #             text_tokens = clip.tokenize(text_prompts).cuda()
    #             text_features = model.encode_text(text_tokens)
    #             text_features /= text_features.norm(dim=-1, keepdim=True)
                
    #             # Compute similarities
    #             similarities = image_features @ text_features.T
    #             max_similarity_idx = similarities.argmax().item()
    #             max_similarity = similarities[0,max_similarity_idx].item()
    #             best_class = candidate_labels[max_similarity_idx]
    #             pseudo_labels[current_idx.item()] = max_similarity_idx
    
    # # File path to save the JSON object
    # file_path = 'psuedolabels.json'

    # # Write list to JSON file
    # with open(file_path, 'w') as f:
    #     json.dump(pseudo_labels, f)

    # Open and load the JSON file
    with open('psuedolabels.json', 'r') as f:
        pseudo_labels = json.load(f)

    # print(f"Pseudo Label List saved to {file_path}")

    accuracy = 0
    for idx, pseudo_label in enumerate(pseudo_labels):
        if(pseudo_label != -1):
            if (train_data['fine_labels'][idx] == pseudo_label) :
                accuracy+=1
            else :
                point_filter[idx] = 0

    print(f"Final number of training points {np.sum(np.array(point_filter))}")   

    print(f"CORRECTLY LABELED IMAGES: {accuracy}")
    print(f"TOTAL NUMBER OF NOISY DATA POINTS: {count}")

    accuracy = (float(accuracy)*100) / count

    print(f"ACCURACY OF PSEUDO-LABELLING: {accuracy:.2f}%\n")

    return pseudo_labels, point_filter


# def pseudo_label_2(dataloader, device):
#     i = 2
#     flag = False
#     # Iterate through the dataloader
#     for batch_idx, (inputs, targets, _, index) in enumerate(dataloader):
#         inputs = inputs.to(device)
#         for j, current_idx in enumerate(index):
#             if current_idx.item() == i:
#                 print(inputs[j].shape)
#                 input_image = inputs[j].cpu().numpy()  # Move image to CPU and convert to numpy array
#                 input_image = input_image.transpose(1, 2, 0)  # Change shape to (32, 32, 3)
#                 flag = True
#                 break
#         if flag == True:
#             break

#     # Plot and save the image
#     if flag:
#         plt.imshow(input_image)
#         plt.title(f"Image {i}")  # Title or label can be added if required
#         plt.axis('off')  # Hide axes
        
#         # Save the image in the root directory
#         image_path = os.path.join(os.getcwd(), 'input_image.png')
#         plt.savefig(image_path)  # Save the image as PNG
#         plt.close()  # Close the plot to free memory
#         print(f"Image saved at {image_path}")
#     else:
#         print("Image not found.")

#     return input_image

# ======== Data ========
if cfg.dataset.startswith("cifar"):
    from dataloader import dataloader_cifar as dataloader
    loader = dataloader.cifar_dataloader(cfg.dataset, noise_mode=cfg.noise_mode, noise_ratio=cfg.noise_ratio,\
                                        batch_size=cfg.batch_size, num_workers=cfg.num_workers, root_dir=cfg.data_path, model=cfg.model)
    train_loader = loader.run('train')
    test_loader = loader.run('test')
elif cfg.dataset.startswith("cub"):
    from dataloader import dataloader_cub as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "stanford_cars":
    from dataloader import dataloader_stanford_cars as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
elif cfg.dataset == "tiny_imagenet":
    from dataloader import dataloader_tiny_imagenet as dataloader
    train_loader, _, test_loader = dataloader.build_loader(cfg)
num_class = cfg.num_class

# ======== Model ========
if cfg.backbone == 'vit':
    model = timm.create_model('vit_base_patch16_224.augreg_in1k', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/vit.npz'))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-5)
elif cfg.backbone == 'resnet':
    model = timm.create_model('resnet50.a1_in1k', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/resnet.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
elif cfg.backbone == 'convnext':
    model = timm.create_model('convnext_tiny.fb_in1k', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/convnext.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
elif cfg.backbone == 'mae':
    model = timm.create_model('vit_base_patch16_224.mae', pretrained=True, num_classes=num_class, pretrained_cfg_overlay=dict(file='./model/weights/mae.pth'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
else:
    model, optimizer = load_clip(cfg)
    cfg.backbone == 'clip'
model.cuda()
criterion = torch.nn.CrossEntropyLoss(reduction='none')
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
total_clean_idx = torch.load("./phase1/{}/{}.pt".format(cfg.dataset, cfg.noise_mode + str(cfg.noise_ratio)), weights_only = False)
best_acc = 0

# LOAD vocab.json
with open('class_map.json', 'r') as f:
    class_map = json.load(f)
candidate_labels = [class_name for class_name in list(class_map)]
# for i in range(10):
#     print(candidate_labels[i])

# LOAD THE TRUE TRAIN DATASET
def load_cifar100_file(filepath):
    with open(filepath, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
    return data_dict
train_data = load_cifar100_file('./data/cifar-100/train')

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/16", device=device)

pseudo_labels, point_filter = pseudo_label(train_loader, clip_model, train_data)

for epoch in range(1, cfg.epochs + 1):
    train_acc = train(epoch, train_loader, pseudo_labels, point_filter)
    test_acc = test(epoch, test_loader)
    best_acc = max(best_acc, test_acc)
    if epoch == cfg.epochs:
        print("Best Acc: %.2f Last Acc: %.2f" % (best_acc, test_acc))
    scheduler.step()