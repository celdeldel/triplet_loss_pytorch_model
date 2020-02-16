import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance

from dataloader import dataloaderTriplet
from torch.utils.data import DataLoader
from loss import TripletLoss
from dataset import TripletDataset
from tqdm import tqdm
from models.resnet18 import Resnet18Triplet
from models.resnet34 import Resnet34Triplet
from models.resnet50 import Resnet50Triplet
from models.resnet101 import Resnet101Triplet
from models.inceptionresnetv2 import InceptionResnetV2Triplet

def main():
    dataroot = "2data"
    datatrainroot = "2data/training_data"
    datatestroot = "2data/testing_data"
    datavalidroot = "2data/valid_data"
    num_triplets_train = 1280
    num_triplets_test = 1280
    model_architecture = "restnet18"
    epochs = 3 
    resume_path = "resume"
    batch_size = 256
    num_workers = 1
    embedding_dimension = 128
    pretrained = True#args.pretrained
    optimizer = "sgd"#args.optimizer
    learning_rate = 0.1#args.lr
    margin = 0.5#args.margin
    start_epoch = 0
    img_size = (224, 224)

    
    data_transforms = transforms.Compose([
        transforms.Resize(size=img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


    # Set dataloaders
    print("train")
    train_dataloader = dataloaderTriplet(datatrainroot, data_transforms, num_triplets =num_triplets_train,   batchsize=batch_size, resolution = img_size )
    print("test") 
    #test_dataloader = DataLoader(dataset=TripletDataset(datatestroot, data_transforms, num_triplets = 1300, resolution = img_size), batch_size=batch_size, shuffle=False, num_workers=2)


    # Instantiate model
    model = Resnet18Triplet(embedding_dimension=embedding_dimension,pretrained=pretrained)

    print("Using {} model architecture.".format(model_architecture))

    # Load model to GPU or multiple GPUs if available
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    cuda0 = torch.device("cuda:0")
    model.to(cuda0)
    # Set optimizers
    if optimizer == "sgd":
        optimizer_model = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
    elif optimizer == "adagrad":
        optimizer_model = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        
    elif optimizer == "rmsprop":
        optimizer_model = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        
    elif optimizer == "adam":
        optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
 

    # Start Training loop
    print("\nTraining using triplet loss on {} triplets starting for {} epochs:\n".format(
            num_triplets_train,
            epochs-start_epoch
        )
    )

    total_time_start = time.time()
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    l2_distance = PairwiseDistance(2).to(cuda0)

    for epoch in range(start_epoch, end_epoch):
        epoch_time_start = time.time()
        triplet_loss_sum = 0
        num_valid_training_triplets = 0

        # Training pass
        model.train()
        for batch_idx, (batch_sample) in enumerate(train_dataloader):
            anc_img = batch_sample[0].to(cuda0)
            pos_img = batch_sample[1].to(cuda0)
            neg_img = batch_sample[2].to(cuda0)

            # Forward pass - compute embedding
            
            anc_embedding = model(anc_img)
            pos_embedding = model(pos_img)
            neg_embedding = model(neg_img)


        # Model only trains on hard negative triplets
        avg_triplet_loss = 0 if (num_valid_training_triplets == 0) else triplet_loss_sum / num_valid_training_triplets
        epoch_time_end = time.time()

        # Print training statistics and add to log
        print('Epoch {}:\tAverage Triplet Loss: {:.4f}\tEpoch Time: {:.3f} hours\tNumber of valid training triplets in epoch: {}'.format(
                epoch+1,
                avg_triplet_loss,
                (epoch_time_end - epoch_time_start)/3600,
                num_valid_training_triplets
            )
        )
        with open('logs/{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch+1,
                avg_triplet_loss,
                num_valid_training_triplets
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')



        # Save model checkpoint
        state = {
            'epoch': epoch+1,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict()
        }

        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()


        # Save model checkpoint
        torch.save(state, 'Model_training_checkpoints/model_{}_triplet_epoch_{}.pt'.format(model_architecture, epoch+1))

    # Training loop end
    total_time_end = time.time()
    total_time_elapsed = total_time_end - total_time_start
    print("\nTraining finished: total time elapsed: {:.2f} hours.".format(total_time_elapsed/3600))


if __name__ == '__main__':
    main()
