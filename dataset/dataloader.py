import os 
import random

import torch
import torchvision
from PIL import Image 
import pdb
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import copy

#from preprocess_dataset import read_label_file
try : 
    from augmentation import * 
    from label_to_heatmap import gen_heat, gen_head
except ModuleNotFoundError : 
    from dataset.augmentation import * 
    from dataset.label_to_heatmap import gen_heat, gen_head

EXTENSIONS = ['.jpg','.jpeg','.png']



class Default_Config():
    def __init__(self, **kwargs):
        self.input_size = 256
        self.heatmap_size = 64
        self.heatmap_method = "GAUSS"
        self.heatmap_sigma = 1.5
        self.per_stack_heatmap = 1 
        self.offset_size = 8
        self.offset_method = "GAUSS"
        self.offset_sigma = 1.0
        self.num_landmarks = None
        
        for key, value in kwargs.items():
            if not hasattr(self, key) : 
                raise KeyError(f'Invalid option : {key}')
            setattr(self, key, value)

def csv_to_array(label_path):
    df = pd.read_csv(label_path)
    df = df[['x','y']] 
    labels = df.to_numpy(dtype=float)
    return labels

class Dataset(torch.utils.data.Dataset): 
    def __init__(self, data_dir, input_size=512, output_size=None, train=True, 
                 offset=True, top_view=True, test_size = 0.2, offset_size=4, test_dir='/home/server18/hdd/hangyul/foot_deformity/dataset/sinchon_valid/') : 
        super().__init__()

        if train: ## collecting data from training directory
            image_dir, label_dir = os.path.join(data_dir, 'image'), os.path.join(data_dir, 'label')
            total_list = sorted([x.split('.')[0] for x in os.listdir(label_dir)])

        else: ## collecting data from test directory
            image_dir, label_dir = os.path.join(test_dir, 'image'), os.path.join(test_dir, 'label')
            total_list = sorted([x.split('.')[0] for x in os.listdir(label_dir)])

        self.totensor = torchvision.transforms.ToTensor()
        self.train = train 
        self.offset = offset 
        self.input_size = input_size
        self.output_size = input_size if output_size is None else output_size
        self.top_view = top_view
        
        self.config = Default_Config(input_size=input_size, heatmap_size=self.output_size, 
                                     num_landmarks = 16 if top_view else 11, offset_size=offset_size)
        self.heatmap = gen_heat(sigma=self.config.heatmap_sigma)

        self.data = []         

        for filename in sorted(os.listdir(image_dir)) : 
            is_top_view = 'A' in filename 
            if is_top_view != top_view : 
                continue 
            
            name, extension =  os.path.splitext(filename)
            if not extension.lower() in EXTENSIONS : 
                print(f'Not an Image: {filename}')
                continue 
            
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, name+'.csv')
            
            if not os.path.isfile(label_path) : 
                print(f'Label not exists for: {filename}')
                continue 
            target = csv_to_array(label_path) 
            if len(target) != self.config.num_landmarks : 
                print(f'{label_path} has {len(target)} landmarks, expected {self.config.num_landmarks}.')
                continue 
            
            basename = os.path.basename(image_path).split('.')[0].replace('AL','A').replace('AR','A')

            if not basename in total_list : 
                if basename+'L' in total_list : 
                    basename = basename+'L'
                elif basename+'R' in total_list : 
                    basename = basename+'R'
                    
                elif basename.replace('R','').replace('L','')+'LR' in total_list : 
                    basename = basename.replace('R','').replace('L','')+'LR'
                elif basename.replace('R','').replace('L','')+'RL' in total_list : 
                    basename = basename.replace('R','').replace('L','')+'RL'
                elif basename.replace('R','L') in total_list : 
                    basename = basename.replace('R','L')
                else : 
                    print(f'{basename} not in the list')
                    continue 
            
            if basename in total_list :
                self.data.append((image_path, label_path))
                
        '''
        if csv_filepath is None : 
            train_data, test_data = train_test_split(self.data, test_size = test_size, random_state=3141592)
            self.data = train_data if train else test_data
        else : 
            dataframe = pd.read_csv(csv_filepath) # 1629
            pdb.set_trace()'''
        
        print(f'Number of data : {len(self.data)}')
        
    
    def __len__(self) : 
        return len(self.data)
    
    def __getitem__(self, index) : 
        image_path, label_path = self.data[index] 
        image_name = os.path.basename(image_path)
        
        img = Image.open(image_path)
        width, height = img.size # opposite with pytorch tensor 
        target = csv_to_array(label_path) 
        
        if len(target) != self.config.num_landmarks : 
            print(image_path, label_path, len(target), self.config.num_landmarks)
            
        
        target[:,0] = target[:,0]/width
        target[:,1] = target[:,1]/height
        
        #img = copy.deepcopy(img)
        #target = copy.deepcopy(target)
    

    ## options for augmentation are inserted
        if self.train:
            img = random_brightness(img, 0.3)
            img = random_contrast(img, 0.3)
            img, target = pad_to_square(img, target, 100)   
            img, target = random_translate(img,target, 30)
            img, target = random_flip(img, target)
            img, target = random_rotate(img, target, 15 if self.top_view else 180)
            # img = random_blur(img, 3)
            # img = random_occlusion(img, 0.3)
            
            img, target = pad_crop(img,target)        
        else : 
            img, target = pad_to_square(img, target, 0)   

        resize_scale = img.size[0]/self.input_size
        img = img.resize((self.input_size, self.input_size))
        
        img = self.totensor(img) # self.totensor -> scale the image between [0., 1.]
        
        target_map, offset_map  = gen_head(target,self.heatmap, self.config, self.offset)
        # offset_map is None if self.offset=False 
        
        return img, target_map, torch.from_numpy(target)*self.output_size, offset_map, image_name, resize_scale
        

if __name__ == '__main__' : 
    from torchvision.utils import save_image
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./dataset/final")
    args = parser.parse_args()    
    
    data_dir = args.dataset_dir
    save_dir = os.path.join(data_dir, 'augmentation_samples_test')
    if not os.path.isdir(save_dir) : 
        os.mkdir(save_dir)
    
    for top_view in [True, False] : 
        for train in [True] : # for train in [True, False] if want to use both train & test sets
            dataset = Dataset(data_dir, top_view=top_view, train=train)
            
            for i in range(len(dataset)) : 
                image, label, target_coord, offset, image_name, resize_scale = dataset[i]
                label_vis = torch.nn.functional.interpolate(torch.from_numpy(label), image.size(-1)).sum(0).clip(min=0.0, max=1.0)
                save_image(image.squeeze()*0.5+label_vis*0.5, os.path.join(save_dir, image_name))