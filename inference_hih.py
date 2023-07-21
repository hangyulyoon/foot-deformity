'''Train CIFAR10 with PyTorch.'''
'''From https://github.com/kuangliu/pytorch-cifar'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
# from torchvision.utils import save_image
from utils.save_image import save_image
import matplotlib.pylab as plt 

import os
import argparse
import datetime
import random 
import pickle 

from dataset.dataloader import Dataset
from utils.heatmap_to_coords import Heatmap_to_Coords, Get_Pixel_Errors
import pdb

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')

parser.add_argument('--data_dir', default='./dataset/real_final', type=str)
parser.add_argument('--test_dir', default='./dataset', type=str)
parser.add_argument('--log_dir', default='./inference_result', type=str)
parser.add_argument('--test_size', type=float, default=0.2) # Ratio of validation data 
parser.add_argument('--input_size', type=int, default=512) # Input image size
parser.add_argument('--output_size', type=int, default=128) # Output heatmap size 
parser.add_argument('--offset_size', type=int, default=4) # Output offset size 

parser.add_argument('--model', type=str, default='HIH')
parser.add_argument('--offset', default=True, action='store_false')
parser.add_argument('--criterion', type=str, default='HIH')
parser.add_argument('--exp_name', type=str, default='HIH_test')
parser.add_argument('--num_epoch', type=int, default=300)
parser.add_argument('--warmup_epoch', type=float, default=3)

parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--print_freq', type=int, default=100)

parser.add_argument('--fast_pass', default=False, action='store_true') # fast inference for debug 
parser.add_argument('--save_every_epoch', default=False, action='store_true') # fast inference for debug 
parser.add_argument('--top_view', default=False, action='store_true') # choose side view or top view.  
parser.add_argument('--num_landmarks', default=None) # Number of landmarks. Default value is 16(top view)/11(side view).  
parser.add_argument('--refine_heatmap', default=True, action='store_false')


parser.add_argument('--optim', default='sgd', type=str)
parser.add_argument('--resume', default=None, type=str)

args = parser.parse_args()

IMAGE_MPP = 0.15


if args.num_landmarks is None : 
    args.num_landmarks = 16 if args.top_view else 11

args.in_out_ratio = args.input_size/args.output_size

if not os.path.isdir(args.log_dir) : 
    os.mkdir(args.log_dir) 
    
timestemp = str(datetime.datetime.now()).replace(':','')
args.log_subdir = os.path.join(args.log_dir, timestemp)
os.mkdir(args.log_subdir) 

print(args)

best_score = 0  
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


# testset = Dataset(args.data_dir, train=False, input_size=args.input_size, output_size=args.output_size, 
#                    offset=args.offset, top_view = args.top_view, test_size=args.test_size)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=args.batch_size*2, shuffle=False, num_workers=0)

testset = Dataset(args.data_dir, train=False, input_size=args.input_size, output_size=args.output_size, 
                   offset=args.offset, top_view = args.top_view, test_size=args.test_size, offset_size=4, test_dir=args.test_dir)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size*2, shuffle=False, num_workers=0)

# Model
print('==> Building model..')
if args.model == 'HIH' : 
    from hih.get_model import HIH_model
    net = HIH_model(input_size=args.input_size, heatmap_size=args.output_size, 
                    num_landmarks=args.num_landmarks, offset_size=args.offset_size, 
                    refine_heatmap = args.refine_heatmap)
else : 
    raise NameError(f'Unknown model name: {args.model}')

n_param = 0 
for param in net.parameters() : 
    n_param += param.numel()
print(f'Number of parameters: {n_param}')


if args.criterion == 'HIH' : 
    from hih.loss import calc_loss, calc_inference_loss 
    criterion = calc_loss 
    inference_criterion = calc_inference_loss 
else : 
    raise NameError(f'Unknown criterion: {args.criterion}')


try : 
    inference_criterion
except : 
    inference_criterion = criterion 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = net.to(device)
if device == 'cuda':
    if torch.cuda.device_count() > 1: 
        net = torch.nn.DataParallel(net) 
    cudnn.benchmark = True
else : 
    print('CPU mode. Would be very slow...')




if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_score = checkpoint['score']
    start_epoch = checkpoint['epoch']
else : 
    raise Exception('Checkpoint needed for inference')




def test(epoch):
    global best_score
    net.eval()
    test_loss = 0
    pixel_distances = [] 
    count = 0 
    count_sample = 0 
    
    
    count_landmark = 0 
    SDR_count = {1:0, 2:0, 3:0, 4:0} 
    SDR_count_per_landmark = 0
    
    coord_dict = {} 
    distance_dict = {}
    NMEs = [] 

    with torch.no_grad():
        for batch_idx, (inputs, targets, target_coords, target_offsets, image_names, resize_scales) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.offset : 
                target_offsets = target_offsets.to(device) 
                outputs, offsets = net.module.inference(inputs, -1)
                loss = inference_criterion(outputs, targets, offsets, target_offsets)
            else : 
                outputs = net(inputs) 
                loss = criterion(outputs, targets)
                offsets = None 
                
                
            test_loss += loss.item()
            
            pred_coords = Heatmap_to_Coords(outputs, offsets) +0.5 # , offsets
            distance = (target_coords - pred_coords).pow(2).sum(axis=-1).pow(0.5)
            NMEs.append((distance/args.output_size).sum(0))
            
            pixel_distance = distance *args.in_out_ratio *resize_scales.unsqueeze(1)
                
            pixel_distances.append(pixel_distance.sum(0))
            count += len(inputs) 
            
            for i in range(len(inputs)) : 
                coord_dict[image_names[i].split('.')[0]] = pred_coords[i].cpu().numpy()
                distance_dict[image_names[i].split('.')[0]] = pixel_distance[i].cpu().numpy()

            count_landmark += pixel_distance.numel()
            for key in SDR_count.keys() : 
                threshold = key/IMAGE_MPP
                SDR_count[key] += (pixel_distance<threshold).int().sum().cpu().item() 
            
            threshold = 2/IMAGE_MPP
            SDR_count_per_landmark += (pixel_distance<threshold).int().sum(0)
                
            if batch_idx % args.print_freq == 0 : 
                print(str(datetime.datetime.now())+f' | {batch_idx}/{len(testloader)} | '+ 'Loss: %.5f | '
                             % (test_loss/(batch_idx+1),))
    
            if args.fast_pass and batch_idx==10 : 
                break             
            
            for idx in range(len(inputs)) : 
                '''
                label_vis = outputs[idx].sum(0, keepdim=True).clip(min=0.0, max=1.0).unsqueeze(0)
                label_vis = F.interpolate(label_vis, size=(args.input_size, args.input_size))
                vis = inputs[idx]*0.3 + label_vis.squeeze(0)*0.7
                save_image(vis, os.path.join(args.log_subdir, f'sample_{count_sample}_heatmap.jpg'))'''
                
                img = inputs[idx].cpu().repeat(3,1,1)
                pred_coords_ = (pred_coords[idx]*args.in_out_ratio).round().int()
                gt_coords_ = (target_coords[idx]*args.in_out_ratio).round().int()
                for j in range(args.num_landmarks) : 
                    img[:, pred_coords_[j][1]-1:pred_coords_[j][1]+1, 
                        pred_coords_[j][0]-1:pred_coords_[j][0]+1] = torch.Tensor([1,0,0]).reshape(3,1,1) # Red : pred
                    img[:, gt_coords_[j][1]-1:gt_coords_[j][1]+1, 
                        gt_coords_[j][0]-1:gt_coords_[j][0]+1] = torch.Tensor([0,0,1]).reshape(3,1,1) # Blue : gt 
                save_image(img, os.path.join(args.log_subdir, image_names[idx].replace('.jpg','.png')))
                count_sample += 1 
                
                
                
                
            
            
            
    # Save checkpoint.
    score = test_loss/(batch_idx+1)

    print(f'Validation loss : {score}')    
    
    pixel_distance = sum(pixel_distances)/count 
    mean_pixel_distance = pixel_distance.mean()
    print(f'Mean pixel distance for landmarks : {pixel_distance}, Average : {mean_pixel_distance}')
    for key in SDR_count : 
        print(f'SDR {key}mm : {SDR_count[key]/count_landmark}')
    
    NME = sum(NMEs)/count 
    mean_NME = NME.mean() 

    print(f'Normalized mean error for landmarks : {NME}, Average : {mean_NME}')    
    
    SDR = SDR_count_per_landmark/count
    n = len(SDR)
    xaxis = list(range(1,n+1))
    plt.figure(figsize=(6,2))
    plt.bar(xaxis, SDR, width=0.5)
    plt.xticks(xaxis)
    plt.xlabel('Landmarks')
    plt.title('SDR (2mm)')
    plt.savefig(os.path.join(args.log_subdir, 'SDR.png'))
    plt.clf()

    plt.figure(figsize=(6,2))
    plt.bar(xaxis, NME, width=0.5)
    plt.xticks(xaxis)
    plt.xlabel('Landmarks')
    plt.title('NME')
    plt.savefig(os.path.join(args.log_subdir, 'NME.png'))    
    pdb.set_trace()
        
    
    #with open(os.path.join(args.log_subdir, f'{args.exp_name}_landmarks.tar'),'wb') as f : pickle.dump(coord_dict, f)    
    with open(f'{args.exp_name}_landmarks.tar','wb') as f : pickle.dump(coord_dict, f)    
    with open(f'{args.exp_name}_distances.tar','wb') as f : pickle.dump(distance_dict, f) 
        
    
    
    return score, pixel_distance, mean_pixel_distance 

loss, pixel_distance, mean_pixel_distance  = test(0)

with open(os.path.join(args.log_subdir, 'log.tar'),'wb') as f : 
    pickle.dump((args, loss, pixel_distance, mean_pixel_distance), f)

print('Inference is over!')
print(args)

