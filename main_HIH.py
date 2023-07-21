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
import numpy as np

import os
from tqdm import tqdm
import argparse
import datetime
import random 
import pickle 

from dataset.dataloader import Dataset
from utils.heatmap_to_coords import Heatmap_to_Coords, Get_Pixel_Errors
from PIL import Image, ImageDraw
import pdb
import time 

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  
parser.add_argument('--dataset_dir', default='./dataset/final', type=str) 
parser.add_argument('--log_dir', default='/home/server18/hdd/hangyul/foot_deformity/dataset/sinchon/training_logs', type=str)
parser.add_argument('--test_size', type=float, default=0.2) # Ratio of validation data 
parser.add_argument('--input_size', type=int, default=512) # Input image size
parser.add_argument('--output_size', type=int, default=128) # Output heatmap size of integer heatmap
parser.add_argument('--offset_size', type=int, default=4) # Output heatmap size of decimal heatmap
parser.add_argument('--num_feature', type=int, default=256) # feature size 

parser.add_argument('--model', type=str, default='HIH')
parser.add_argument('--offset', default=True, action='store_false') # Use offset (heatmap in heatmap) or not - should be disable if use Unet, etc. 
parser.add_argument('--criterion', type=str, default='HIH')
parser.add_argument('--exp_name', type=str, default='HIH_test')
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--warmup_epoch', type=float, default=3) # Linear warmup for n epoch

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--print_freq', type=int, default=100)

parser.add_argument('--fast_pass', default=False, action='store_true') # fast inference for debug 
parser.add_argument('--save_every_epoch', default=False, action='store_true') # Save model checkpoints for every single epoch. 
parser.add_argument('--top_view', default=False, action='store_true') # choose viewpoint, top (AP) or side.  
parser.add_argument('--num_landmarks', default=None) # Number of landmarks. Default value is 16(top view)/11(side view).  
parser.add_argument('--pretrained', default=False, action='store_true')   
parser.add_argument('--refine_heatmap', default=True, action='store_false')


parser.add_argument('--optim', default='adamw', type=str)
parser.add_argument('--resume', default=None, type=str)

args = parser.parse_args()

if args.num_landmarks is None : 
    args.num_landmarks = 16 if args.top_view else 11

args.in_out_ratio = args.input_size/args.output_size

if not os.path.isdir(args.log_dir) : 
    os.mkdir(args.log_dir) 
    
timestemp = str(datetime.datetime.now()).replace(':','')
args.log_subdir = os.path.join(args.log_dir, timestemp)
os.mkdir(args.log_subdir) 

print(args)

best_score = 1e+10
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')

# Data
print('==> Preparing data..')

trainset = Dataset(args.dataset_dir, train=True, input_size=args.input_size, output_size=args.output_size, 
                   offset=args.offset, top_view = args.top_view, test_size=args.test_size, offset_size=args.offset_size)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

testset = Dataset(args.dataset_dir, train=False, input_size=args.input_size, output_size=args.output_size, 
                   offset=args.offset, top_view = args.top_view, test_size=args.test_size, offset_size=args.offset_size)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size*2, shuffle=False, num_workers=0)


# def pil_draw_point(image, point, idx):
#     x, y = point
#     draw = ImageDraw.Draw(image)
#     radius = 2
#     draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='red')
#     draw.text(xy=(x, y - radius * 8), text = str(idx+1), fill='red')

#     return image

# os.makedirs('example_figure', exist_ok=True)
# for train_idx in tqdm(range(len(trainset))):
#     img, target_map, output, offset_map, image_name, resize_scale = trainset[train_idx]
#     img_pil = Image.fromarray((img.squeeze().numpy() * 255.).astype('uint8'))
#     img_pil = img_pil.convert(mode='RGB')
#     for i in range(output.shape[0]):
#         landmark_point = output[i].numpy() * args.in_out_ratio
#         img_pil = pil_draw_point(img_pil, landmark_point, i)

#     img_pil.save(f'example_figure/{image_name}')
# pdb.set_trace()

# landmark_list = [np.array([[], []])] * 11

# for train_idx in tqdm(range(len(trainset))):
#     img, target_map, output, offset_map, image_name, resize_scale = trainset[train_idx]
#     for i in range(output.shape[0]):
#         landmark_point = output[i].unsqueeze(0).numpy() * args.in_out_ratio
#         if train_idx == 0:
#             landmark_list[i] = landmark_point
#         else:
#             landmark_list[i] = np.concatenate([landmark_list[i], landmark_point], axis = 0)
# mean_list = [np.round(np.mean(landmark_array, axis=0), 1) for landmark_array in landmark_list]
# std_list = [np.round(np.std(landmark_array, axis=0), 1) for landmark_array in landmark_list]
# pdb.set_trace()

# Model
print('==> Building model..')
if args.model == 'HIH' : 
    from hih.get_model import HIH_model
    if args.pretrained : 
        print('Use pretrained model from WFLW')
        net = HIH_model(input_size=args.input_size, heatmap_size=args.output_size, num_feature = args.num_feature,
                        num_landmarks=args.num_landmarks, offset_size=args.offset_size, 
                        pretrained=True, pretrained_path='./checkpoint/WFLW.pth', 
                        refine_heatmap = args.refine_heatmap)
    else : 
        net = HIH_model(input_size=args.input_size, heatmap_size=args.output_size, num_feature = args.num_feature,
                        num_landmarks=args.num_landmarks, offset_size=args.offset_size, 
                        refine_heatmap = args.refine_heatmap)
else : 
    raise NameError(f'Unknown model name: {args.model}')

n_param = 0 
for param in net.parameters() : 
    n_param += param.numel()
print(f'Number of parameters: {n_param}')


# loss: MSE loss & KL Divergence loss between the two heatmaps
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


if args.optim == 'sgd' : 
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.optim == 'adam' :
    optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay=5e-4)
elif args.optim == 'adamw' :
    optimizer = optim.AdamW(net.parameters(), lr = args.lr, weight_decay=5e-4)
else : 
    raise NameError(f'Unknown optimizer name: {args.optim}')
    
n = len(trainloader)
if args.warmup_epoch == 0  : 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch*n)
else : 
    from utils.warmup import GradualWarmupScheduler
    after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.num_epoch-args.warmup_epoch)*n)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epoch*n, after_scheduler=after_scheduler) 
    print(f'Warmup for the first {args.warmup_epoch*n} steps.')
    
    


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['net'])
    best_score = checkpoint['score']
    start_epoch = checkpoint['epoch']
    for _ in range((start_epoch-1)*len(trainloader)) : 
        scheduler.step() 


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets, target_coords, target_offsets, image_names, resize_scales) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if args.offset : 
            target_offsets = target_offsets.to(device) 
            outputs, offsets = net(inputs)

            # pdb.set_trace()

            loss = criterion(outputs, targets, offsets, target_offsets)
        else : 
            outputs = net(inputs) 
            loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() 

        scheduler.step()                        
        
        train_loss += loss.item()
        if batch_idx % args.print_freq == 0 or batch_idx == len(trainloader)-1 : 
            print(str(datetime.datetime.now())+f' | {batch_idx}/{len(trainloader)} | '+ 'Loss: %.5f'
                         % (train_loss/(batch_idx+1), ))
        if args.fast_pass and batch_idx==10 : 
            break 

        if epoch==0 and batch_idx==0 : 
            save_image(inputs[0].cpu(), os.path.join(args.log_subdir, 'sample_train.jpg'))

    
    return train_loss/(batch_idx+1)
            


def test(epoch):
    print('Validation')
    global best_score
    net.eval()
    test_loss = 0
    pixel_distances = [] 
    count = 0 

    with torch.no_grad():
        for batch_idx, (inputs, targets, target_coords, target_offsets, image_names, resize_scales) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if args.offset : 
                target_offsets = target_offsets.to(device) 
                if torch.cuda.device_count() > 1: 
                    outputs, offsets = net.module.inference(inputs, -1)
                else:
                    outputs, offsets = net.inference(inputs, -1)
                loss = inference_criterion(outputs, targets, offsets, target_offsets)
            else : 
                outputs = net(inputs) 
                loss = criterion(outputs, targets)
                offsets = None 
                
                
            test_loss += loss.item()
            
            pred_coords = Heatmap_to_Coords(outputs, offsets)+0.5
            distance = (target_coords - pred_coords).pow(2).sum(axis=-1).pow(0.5)
            pixel_distance = distance *args.in_out_ratio *resize_scales.unsqueeze(1)
                
            pixel_distances.append(pixel_distance.sum(0))
            count += len(inputs) 
    
    
            if batch_idx % args.print_freq == 0 : 
                print(str(datetime.datetime.now())+f' | {batch_idx}/{len(testloader)} | '+ 'Loss: %.5f | '
                             % (test_loss/(batch_idx+1),))
    
            if args.fast_pass and batch_idx==10 : 
                break             
            
            # if batch_idx==0 : 
            #     idx = random.randint(0, len(inputs)-1)
            #     label_vis = outputs[idx].sum(0, keepdim=True).clip(min=0.0, max=1.0).unsqueeze(0)
            #     label_vis = F.interpolate(label_vis, size=(args.input_size, args.input_size))
            #     vis = inputs[idx]*0.3 + label_vis.squeeze(0)*0.7
            #     save_image(vis, os.path.join(args.log_subdir, f'sample_epoch{epoch}_heatmap.jpg')) # can change the size setting of the save_image func
                
            #     img = inputs[idx].cpu().repeat(3,1,1)
            #     pred_coords_ = (pred_coords[idx]*args.in_out_ratio).round().int()
            #     gt_coords_ = (target_coords[idx]*args.in_out_ratio).round().int()
            #     for j in range(args.num_landmarks) : 
            #         img[:, gt_coords_[j][1]-1:gt_coords_[j][1]+1, 
            #             gt_coords_[j][0]-1:gt_coords_[j][0]+1] = torch.Tensor([0,0,1]).reshape(3,1,1) # Blue : gt
            #         img[:, pred_coords_[j][1]-1:pred_coords_[j][1]+1, 
            #             pred_coords_[j][0]-1:pred_coords_[j][0]+1] = torch.Tensor([1,0,0]).reshape(3,1,1) # Red : pred
            #     save_image(img, os.path.join(args.log_subdir, f'sample_epoch{epoch}_positions.png'))
    
    # Save checkpoint.
    score = test_loss/(batch_idx+1)

    print(f'Validation loss : {score}')    
    
    val_pixel_distance = sum(pixel_distances)/count 
    mean_pixel_distance = val_pixel_distance.mean()
    print(f'Mean pixel distance for landmarks : {val_pixel_distance}, Average : {mean_pixel_distance}')

    print('Saving..')    
    state = {
        'net': net.state_dict(),
        'score': score,
        'epoch': epoch,
    }
    if args.save_every_epoch : 
        torch.save(state, os.path.join(args.log_subdir, f'{args.exp_name}_epoch{epoch}.pth'))      
    if score < best_score :        
        torch.save(state, os.path.join(args.log_subdir, f'{args.exp_name}_best.pth'))
        best_score = score  
    if True : 
        torch.save(state, os.path.join(args.log_subdir, f'{args.exp_name}_last.pth'))
    
    return score, pixel_distance, mean_pixel_distance 

train_loss_log = [] 
val_loss_log = [] 
mean_dist_log = [] 
dist_log = []

for epoch in tqdm(range(start_epoch, args.num_epoch)):
    print(f'Current learning rate : {optimizer.param_groups[0]["lr"]}')
    train_loss = train(epoch)
    val_loss, dist, mean_dist = test(epoch)
    
    train_loss_log.append(train_loss)
    val_loss_log.append(val_loss)
    mean_dist_log.append(mean_dist)
    dist_log.append(dist)
    
    try : 
        plt.plot(train_loss_log)
        plt.title('Training loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(args.log_subdir, 'loss_train.png'))
        plt.clf()
    
        plt.plot(val_loss_log)
        plt.title('Validation loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(args.log_subdir, 'loss_val.png'))    
        plt.clf()

        plt.plot(mean_dist_log)
        plt.title('Mean pixel distance')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(args.log_subdir, 'pixel_distance.png'))    
        plt.clf()        
        
        
        with open(os.path.join(args.log_subdir, 'log.tar'),'wb') as f : 
            pickle.dump((args, train_loss, val_loss, dist_log, mean_dist_log), f)
    except Exception as e : 
        print(e)    
        pdb.set_trace()
    
    
    if args.fast_pass and epoch == 9 : 
        break

print('Training is over!')
print(args)

