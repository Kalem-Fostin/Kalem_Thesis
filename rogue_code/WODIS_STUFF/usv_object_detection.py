# The code WODIS has been taken from elsewhere. The appropriate references are in the WODIS files.

import os
import pandas as pd
import time
import numpy as np
import torch 
import sys 
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
sys.path.append('/home/bluebouy/kalem_stuff/object_detection/ASV_Segmentation/WODIS')
from WODIS import WODIS_model
from matplotlib import pyplot as plt
import cv2
import cfg

DEVICE = 'cuda'

runningState = 0 # 0 equal zed camera and 1 equal to svo

def adjust_gamma(image, gamma=0.6):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def main():
    
    device = torch.device(DEVICE)

    '''
    create the model and start the inference...
    '''
    # load model
    net = WODIS_model(is_training=False, num_classes=cfg.NUM_CLASSES).to(device)
    checkpoint = torch.load(cfg.MODEL_WEIGHTS, map_location='cpu')
    net.load_state_dict(checkpoint)
    pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []

    # read label color from the class dict i.e. class_dict.csv which has rgb values for sea sky and obstacles
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8') # order of rgb values should be obstacle, sea then sky
    print("cm: ", cm, np.shape(cm))

    # create output folder if it does not exist.
    #if not os.path.exists(cfg.SAVE_DIR):
    #    os.makedirs(cfg.SAVE_DIR)

    # get number of lines in text file
    #num_imgs = sum(1 for line in open(cfg.SEQ_TXT))

    # perform inferences on dataset
    #f_id = open(cfg.SEQ_TXT, 'r')

    counter = 1
    sum_times = 0


    
    num_imgs = 1
    # read image
    img = cv2.imread("/home/bluebouy/kalem_stuff/object_detection/image_of_shore.JPG")
    img = cv2.resize(img, dsize=(512, 384))

    # lets apply gaussian filtering to reduce the noise !
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img)
    #plt.show()
    #img = adjust_gamma(img, gamma=1.1)

    #kernel = np.ones((2,2),np.float32)/25
    #sigma = 10 + (i * 10)
    #img = cv2.bilateralFilter(img, 5, sigma, sigma)
    #plt.imshow(img)
    #plt.show()
    

    img_reverse = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([transforms.Resize(cfg.IMG_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read one image for the inference process, it needs to expand the first dimension to match the tensor.
    img_out = transform(img_reverse).unsqueeze(0) 

    # inference starting...
    start_time = time.time()
    valImg = img_out.to(device) # load image to gpu for fast encoding
    out, cx1 = net(valImg) # I think this is the encoder ?

    elapsed_time = time.time() - start_time
    sum_times += elapsed_time

    out = F.log_softmax(out, dim=1) # this function is used for smoothing the probablity of class classification. shape of out is the (batch size, num of classes, height pixels, width pixels)

    # now for pre_label we remove the batch size and num classes from the tensor sets all pixels that are in the classes to the asscoaited number i.e. 0, 1 or 2 depending on classification
    pre_label = out.max(dim=1)[1].squeeze().cpu().data.numpy() # converts all pixels to 0, 1, 2 value which is used to fill in the rgb i.e. 0 is obstacle, 1 is water and 2 is the sky

    pre = cm[pre_label] # fills image detectionms with the segment colours
    save_name = 'image1'
    cv2.imwrite('/home/bluebouy/kalem_stuff/object_detection/objects_detected/' + save_name + '.png', pre)
    counter += 1
    
if __name__ == "__main__":
    main()
    print("Finished algo")
    exit(0)