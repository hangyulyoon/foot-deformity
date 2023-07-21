
import os
from PIL import Image,ImageFilter, ImageEnhance
import numpy as np
import random
from math import floor,ceil
import pdb

# !! Must do deep copy before data augmentation

"""
    args:
        image: type is PIL.image 
        target: numpy array type, (landmark_num,2)
"""

def scaling_function(image):
    image_mean = np.array([[[0.485, 0.456, 0.406]]])
    image_std = np.array([[[0.229, 0.224, 0.225]]])

    return (image.astype(np.float32)-image_mean)/image_std
    
def descaling_function(image):
    image_mean = np.array([[[0.485, 0.456, 0.406]]])
    image_std = np.array([[[0.229, 0.224, 0.225]]])

    return (image.astype(np.float32)*image_std)+image_mean


def random_translate(image, target, max_value = 30):
    # PIL type
    if random.random() > 0.5:
        image = Image.fromarray((image*255.).astype('uint8'))
        image_height, image_width = image.size
        a = 1
        b = 0
        c = int((random.random()-0.5) * max_value*2) # -30 ~ + 30
        d = 0
        e = 1
        f = int((random.random()-0.5) * max_value*2) # -30 ~ + 30
        image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))
        target[:, 0] -= 1.*c
        target[:, 2] -= 1.*c
        target[:, 1] -= 1.*f
        target[:, 3] -= 1.*f
        return np.asarray(image)/255., target
    else:
        return image, target


def random_blur(image:np.ndarray, max_value=5):
    # PIL type
    if random.random() > 0.5 :
        image = Image.fromarray((image*255.).astype('uint8'))
        image = image.filter(ImageFilter.GaussianBlur(random.random()*max_value))
    return np.asarray(image)/255.

def random_flip(image,target): # ,points_flip
    # PIL type
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        #target = target[points_flip, :]
        target[:,0] = 1-target[:,0]
        return image, target
    else:
        return image, target

def random_rotate(image, target, max_value=30):
    # PIL type
    if random.random() > 0: #.5:
        image = Image.fromarray((image*255.).astype('uint8'))
        image_height, image_width = image.size
        center_x = 0.5 * image_width
        center_y = 0.5 * image_height

        landmark_num= target.shape[0]
        target_center = target - np.array([center_x, center_y, center_x, center_y, 0])
        theta_max = np.radians(max_value)
        theta = random.uniform(-theta_max, theta_max)
        angle = np.degrees(theta)
        image = image.rotate(angle)

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c,-s), (s, c)))
        target_center_rot_1 = np.matmul(target_center[:, :2], rot)
        target_center_rot_2 = np.matmul(target_center[:, 2:4], rot)
        target_rot_1 = target_center_rot_1 + np.array([center_x, center_y])
        target_rot_2 = target_center_rot_2 + np.array([center_x, center_y])
        return np.asarray(image)/255., np.concatenate((target_rot_1, target_rot_2, target[:, -1:]), axis=1)
    else:
        return image, target


def random_occlusion(image, max_value=0.4):
    # Need change data type
    if random.random() > 0.5:
        image = Image.fromarray((image*255.).astype('uint8'))
        image_np = np.array(image).astype(np.uint8)
        image_height, image_width = image_np.shape # , _
        occ_height = int(image_height*max_value*random.random())
        occ_width = occ_height #int(image_width*max_value*random.random())
        occ_xmin = int((image_width - occ_width - 10) * random.random())
        occ_ymin = int((image_height - occ_height - 10) * random.random())
        image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width] = int(random.random() * 255)
        #image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 1] = int(random.random() * 255)
        #image_np[occ_ymin:occ_ymin+occ_height, occ_xmin:occ_xmin+occ_width, 2] = int(random.random() * 255)
        return image_np/255.
    else:
        return image


def pad_crop(image,target):
    """
        add pad for the overflow points
        image need change data type
        border_pad : 8px
    """
    image_height, image_width = image.size

    l,t = np.min(target,axis=0)
    r,b = np.max(target,axis=0)
    
    # if the over border is left than grid_size, pass
    grid_size = 0.5 / image_height 
    
    if l > -grid_size and t > -grid_size and r < (1 + grid_size) and b < (1 + grid_size):
        target = np.maximum(target,0)
        target = np.minimum(target,1)
        return image,target
    border_pad_value = 8
    image_np = np.array(image).astype(np.uint8)
    border_size = np.zeros(4).astype('int') # upper bottom left right
    if l < 0:
        border_size[2] = ceil(-l * image_height) + border_pad_value #left
    if t < 0:
        border_size[0] = ceil(-t * image_width) + border_pad_value #upper
    if r > 1:
        border_size[3] = ceil((r-1) * image_height) + border_pad_value #right
    if b > 1:
        border_size[1] = ceil((b-1) * image_width) + border_pad_value #bottom
    border_img = np.zeros((image_width  + border_size[0] + border_size[1],
                           image_height + border_size[2] + border_size[3])).astype(np.uint8) # ,1

    border_img[border_size[0] : border_size[0]+image_height, 
               border_size[2] : border_size[2]+image_width] = image_np #np.expand_dims(image_np, axis=-1)
               
    image_pil = Image.fromarray(border_img.astype('uint8'), 'L')
    image_pil = image_pil.resize((image_height,image_width))
    target = (target * np.array([image_height,image_width]) + 
              np.array([border_size[2],border_size[0]])) /  np.array([border_img.shape[1],border_img.shape[0]])

    return image_pil, target


def pad_to_square(image, target, max_value) : 
    image = Image.fromarray((image*255.).astype('uint8'))
    width, height = image.size
    square_size = max(width, height) 
    if random.random() > 0.5 : 
        square_size = square_size+ round(random.random()*max_value)
    
    right = (square_size-width)//2 
    left = square_size-width-right 
    top = (square_size-height)//2 
    bottom = square_size-height-top 
    new_image = Image.new(image.mode, (square_size, square_size))
    
    new_image.paste(image, (left, top))
    
    target[:,0] = (target[:,0] + left) 
    target[:,2] = (target[:,2] + left) 
    target[:,1] = (target[:,1] + top)
    target[:,3] = (target[:,3] + top)
    
    #new_image = new_image.resize((target_size, target_size))
    
    return np.asarray(new_image)/255., target 

def random_contrast(image, max_value = 0.5) : 
    if random.random() > 0.5 : 
        image = Image.fromarray((image*255.).astype('uint8'))
        factor = (random.random()-0.5)*2*max_value+1 
        enhancer = ImageEnhance.Contrast(image)
        return np.asarray(enhancer.enhance(factor))/255.
    else : 
        return image 

def random_brightness(image, max_value = 0.5) : 
    if random.random() > 0.5 : 
        image = Image.fromarray((image*255.).astype('uint8'))
        factor = (random.random()-0.5)*2*max_value+1 
        enhancer = ImageEnhance.Brightness(image)
        return np.asarray(enhancer.enhance(factor))/255.
    else : 
        return image 
        

def ignore_crop(target):
    target = np.maximum(target,0)
    target = np.minimum(target,1)
    return target