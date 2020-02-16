from io import BytesIO
import os
from PIL import Image
from torch.utils.data import Dataset

import random
import copy
from tqdm import tqdm

class TripletDataset(Dataset):
    def __init__(self, path, transform, num_triplets, resolution=24):
        self.path = path
        self.dir = [os.path.join(path,x ) for x in os.listdir(path)]
        self.resolution = resolution
        self.transform = transform
        self.length = len(self.dir)
        self.num_triplets = num_triplets
        self.triplets = self.generate_triplets(num_triplets= self.num_triplets)


    def __len__(self):
        return self.length

    
    def generate_triplets(self, num_triplets): 
         
        triplets = []
        progress_bar = tqdm(range(num_triplets))
        for _ in progress_bar:
            list_dir_copy = copy.copy(self.dir)
            pos_class = list_dir_copy.pop(int(random.random()*len(list_dir_copy)))
            list_pos_file = [os.path.join(pos_class, x) for x in os.listdir(pos_class)]
            anc_file = list_pos_file.pop(int(random.random()*len(list_pos_file)))
            pos_file = list_pos_file.pop(int(random.random()*len(list_pos_file)))
            neg_class = list_dir_copy.pop(int(random.random()*len(list_dir_copy))) 
            list_neg_file = [os.path.join(neg_class, x) for x in os.listdir(neg_class)]
            neg_file = list_neg_file.pop(int(random.random()*len(list_neg_file))) 
            triplets.append({"anc": anc_file, "pos": pos_file, "neg": neg_file})
        self.triplets = triplets 
        return triplets


    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        anc_img = self.transform(Image.open(triplet["anc"]).convert('RGB'))
        pos_img = self.transform(Image.open(triplet["pos"]).convert('RGB'))
        neg_img = self.transform(Image.open(triplet["neg"]).convert('RGB'))
        sample = [anc_img,  pos_img, neg_img]
        return sample


        
