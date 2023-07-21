import os
img_dir = '/home/server18/hangyul/gangnam_image/' ## specify img directory

FLIPPING_DICT = {}
for img in os.listdir(img_dir):
    FLIPPING_DICT[img] = False

## Adding the img names that should be flipped
flip_list = []

for k, v in flip_list:
    FLIPPING_DICT[k] = v