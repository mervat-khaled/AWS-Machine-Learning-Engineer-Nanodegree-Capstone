#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
# Importing Required Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import copy
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
# We were facing Truncated Images error, so to avoid that using this

ImageFile.LOAD_TRUNCATED_IMAGES = True

from smdebug import modes
from smdebug.pytorch import get_hook

#For Logging
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Test Function 
# Metrics are added here to be Logged during Model Testing

def test(model, test_loader, criterion, device):
    model.eval()
    running_loss=0
    running_corrects=0
    total_data_len = 0
    pred = []
    label = []
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device) #FOR GPU
        labels=labels.to(device) #FOR GPU  
        outputs=model(inputs)
        logger.info(f"Outputs: {outputs}")
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        logger.info(f"Prediction is:{preds}")
        logger.info(f"Label is: {labels.data}")
        new_pred = preds.tolist()
        new_label= labels.data.tolist()
        logger.info(f"Prediction List:{new_pred}")
        logger.info(f"Label List: {new_label}")
        pred.extend(new_pred)
        label.extend(new_label)
        logger.info(f"Final Prediction List Updated:{pred}")
        logger.info(f"Final Label List Updated: {label}")
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_data_len+= len(labels.data)
        logger.info(f"Correct are: {torch.sum(preds == labels.data)}")
        logger.info(f"Running Corrects are: {running_corrects}")

    metrics = {0:{"tp":0, "fp":0}, 1:{"tp":0, "fp":0}, 2:{"tp":0, "fp":0}, 3:{"tp":0, "fp":0}, 4:{"tp":0, "fp":0}}
    label_count = {0:0, 1:0, 2:0, 3:0, 4:0}
    for l, p in zip(label, pred):
        label_count[l]+=1
        if(p==l):
            metrics[l]["tp"]+=1
        else:
            metrics[p]["fp"]+=1
            
    logger.info(f"Metrics Computed: {metrics}")
    logger.info(f"Label Count Computed: {label_count}")
    Precision = {0:0, 1:0, 2:0, 3:0, 4:0}
    Recall = {0:0, 1:0, 2:0, 3:0, 4:0}
    F1 = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    for c in Precision:
        denom = metrics[c]["tp"] + metrics[c]["fp"]
        if(denom==0):
            Precision[c]==0
        else:
            num = metrics[c]["tp"]
            Precision[c] = num/denom
    
    
    for c in Recall:
        denom = label_count[c]
        if(denom==0):
            Recall[c]==0
        else:
            num = metrics[c]["tp"]
            Recall[c] = num/denom
            
    for c in F1:
        if(Precision[c]==0 and Recall[c]==0):
            F1[c]=0
        else:
            num = 2*Precision[c]*Recall[c]
            denom = Precision[c] + Recall[c]
            F1[c] = num/denom
            
    
    logger.info(f"Precision Computed: {Precision}")
    logger.info(f"Recall Computed: {Recall}")
    logger.info(f"F1 Computed: {F1}")
    
    
    logger.info(f"Test Len: {total_data_len}")
    logger.info(f"{running_corrects.double()}")
    total_loss = running_loss // len(test_loader)
    new_acc = float(running_corrects)/float(total_data_len)
    
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {new_acc}")

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_loss(criterion)
    epochs=10 #5 Epochs while fine tuning for HP Search, and 10 while training with best hyperparameter
    image_dataset={'train':train_loader, 'valid':validation_loader}
    
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        total_data_len = 0
        for phase in ['train', 'valid']:
            if phase=='train':
                model.train()
                if hook:
                    hook.set_mode(modes.TRAIN)
            else:
                model.eval()
                if hook:
                    hook.set_mode(modes.EVAL)
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(device) #FOR GPU
                labels=labels.to(device) #FOR GPU
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_data_len+= len(labels.data)

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = float(running_corrects) / float(total_data_len)
            
            logger.info('{} loss: {:.4f}, acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
            
            if(epoch==(epochs-1) and phase=="valid"): #Last epoch's validation is objective metric
                logger.info('Final Validation Loss: {:.4f}, acc: {:.4f}'.format(epoch_loss,epoch_acc))
        
    return model
    
def net():
    model = models.resnet50(pretrained=True) #Using a Resnet 50 Pre-Trained Model

    for param in model.parameters():
        param.requires_grad = False   

    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(),
                   nn.Linear(128, 5))
    return model

def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    logger.info(f'Hyperparameters are LR: {args.learning_rate}, Batch Size: {args.batch_size}')
    logger.info(f'Data Paths: {args.data}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)
    model=net()
    model=model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    
    logger.info("Starting Model Training")
    model=train(model, train_loader, validation_loader, criterion, optimizer, device)
    
    logger.info("Testing Model")
    test(model, test_loader, criterion, device)
    
    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)