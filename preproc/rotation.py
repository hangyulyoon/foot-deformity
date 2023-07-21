import os
img_dir = '/home/server18/hangyul/gangnam_image/' ## specify img directory

ROTATION_DICT = {}
for img in os.listdir(img_dir):
    ROTATION_DICT[img] = 0

## Adding the img names and the angles that should be rotated
rot_90_list = ['326583_A.jpg', '1281286_R.jpg', '1509454_R.jpg']
rot_180_list = ['353487_LR.jpg', '521383_LR.jpg', '650972_LR.jpg', '731567_LR.jpg', '1281286_AR.jpg', '1468233_LR.jpg', '6067412_LR.jpg', '6100215_LR.jpg',\
     '6106046_LR.jpg', '6110370_LR.jpg', '6112924_LR.jpg', '6114906_LR.jpg', '6115984_LR.jpg', '6118142_LR.jpg']
rot_270_list = ['1281286_L.jpg', '1509454_L.jpg', '6105436_A.jpg', '6166389_A.jpg', '9102264_A.jpg']

for k in rot_90_list:
    ROTATION_DICT[k] = 90
for k in rot_180_list:
    ROTATION_DICT[k] = 180
for k in rot_270_list:
    ROTATION_DICT[k] = -90
