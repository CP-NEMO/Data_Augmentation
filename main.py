from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import os
from skimage.util import random_noise
np.random.seed(42) ## random seed, change it accordingly
from src.geometric_aug import rotate,shear,flip,xml2dim
from src.data_split import split_data
from src.pascal_to_yolo import convert_pascal_to_yolo

#Read image daa and labels
def read_files(img_dir, lbl_dir):
  lbls_dataset = []
  images = os.listdir(img_dir)
  img_names = []
  for i, image_name in enumerate(tqdm(images)): #Load images and labels from source
    img_names.append(image_name)
    with open(lbl_dir+'/'+image_name.split('.')[0]+'.txt') as f:
        lbls_dataset.append(f.readlines())
        lbls_dataset[i][0]=lbls_dataset[i][0].replace('\n', '').split(' ') #remove unwanted characters and split data
  return lbls_dataset, img_names

#Reading and data storing functions

def find_data(labels_dataset, typ):
    """
    Description: Find files with the given category.
    labels_dataset = list
    typ = int, numberic value of the desired category to be found 
    """
    index=[]
    for d in range(len(labels_dataset)):
        if int(labels_dataset[d][0][0])==typ:
            index.append(d)
    if len(index) == 0:
        print('The dataset does not contain any requested label')
    return index



def create_imgnlbl(name, lbl, img, x1, x2, y1, y2,labels_directory,image_directory):
    """
    Description: Save augmented image and label in a pre-defined directory
    name=file name without extension (str)
    lbl = category number(int)
    img= MxN list
    x1, x2, y1, y2 = coordinates of bbox
    Note: Modify the label and image directories accordingly
    """
    labels_path = Path(f"{labels_directory}")#labels path
    h,w,_=img.shape
    x1, y1 = x1/w, y1/h #escalate x and y (0 to 1)
    x2, y2 = x2/w, y2/h
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    cv2.imwrite(image_directory+"/"+name+'.jpg', img)
    name_l=name+".txt"
    with (labels_path/name_l).open(mode="w") as label_file:
        label_file.write(
            f"{lbl} {x1 + bbox_width / 2} {y1 + bbox_height / 2} {bbox_width} {bbox_height}\n"
        )
    return
def create_lblbox(img, lbl, x1, x2, y1, y2):
    """
    Description: Create a label in the YOLOv5 format
    lbl = category number(int)
    x1, x2, y1, y2 = coordinates of bbox
    Note: Modify the label and image directories accordingly
    """
    h,w,_=img.shape
    x1, y1 = x1/w, y1/h #escalate x and y (0 to 1)
    x2, y2 = x2/w, y2/h
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    return [[float(lbl), float(x1 + bbox_width / 2), float(y1 + bbox_height / 2), float(bbox_width), float(bbox_height)]]



if __name__=="__main__":

    image_directory = './samples/images/train'
    labels_directory = './samples/output_labels/'
    output_label_dir = './samples/aug_labels/'
    output_img_dir = './samples/aug_images'
    dataset_path = './dataset_xml_format'

    #Split images from xml files
    split_data(image_directory,labels_directory,dataset_path)
    
    # Convert Pascal to yolo format
    convert_pascal_to_yolo(labels_directory,output_label_dir,image_directory)
    labels_dataset, image_names=read_files(image_directory,labels_directory)
    etiqueta=0
    cat_lbls=find_data(labels_dataset, etiqueta)
    print(cat_lbls)
    bsize=100 #number of samples to be selected randomly
    select=np.random.choice(cat_lbls, bsize, replace=False)
    for c in tqdm(range(len(select))):
        index=select[c]
        img=np.array(cv2.imread(image_directory+'/'+image_names[index]))
        lbl_dataset=labels_dataset[index]
        nombre=image_names[index].split('.')[0]
        
        # The following three lines generate coordinates for photometric transformations only
        _, a, b, bbox_width, bbox_height=xml2dim(lbl_dataset)#lbl, a, b, bbox_width, bbox_height
        w = int(img.shape[1])
        h = int(img.shape[0])
        x1, y1 = np.round_((a-bbox_width/2)*w,0), np.round_((b-bbox_height/2)*h,0)
        x2, y2 = np.round_((a+bbox_width/2)*w,0), np.round_((b+bbox_height/2)*h,0)
        
        img_gnoise = (255*random_noise(img, mode='gaussian', var=0.05**2)).astype(np.uint8) #_GN
        create_imgnlbl(nombre+"_GN", etiqueta, img_gnoise, x1, x2, y1, y2,output_label_dir,output_img_dir)
        
        img_spnoise = (255*random_noise(img, mode='s&p', amount=0.05, salt_vs_pepper=0.5)).astype(np.uint8) #_SP
        create_imgnlbl(nombre+"_SP", etiqueta,  img_spnoise, x1, x2, y1, y2,output_label_dir,output_img_dir)
            
        img_shx, x1_sx, x2_sx, y1_sx, y2_sx=shear(img,lbl_dataset,0.01,0) #sheared_img, u1, u2, v1, v2 _SX
        create_imgnlbl(nombre+"_SX", etiqueta, img_shx, x1_sx, x2_sx, y1_sx, y2_sx,output_label_dir,output_img_dir)
        
        img_shy, x1_sy, x2_sy, y1_sy, y2_sy=shear(img,lbl_dataset,0,0.01) #sheared_img, u1, u2, v1, v2 _SY
        create_imgnlbl(nombre+"_SY", etiqueta, img_shy, x1_sy, x2_sy, y1_sy, y2_sy,output_label_dir,output_img_dir)
        
        img_fliplr, x1_fl, x2_fl, y1_fl, y2_fl=flip(img,lbl_dataset,0) #flip_img, x1, x2, y1, y2 _LR
        create_imgnlbl(nombre+"_LR", etiqueta, img_fliplr, x1_fl, x2_fl, y1_fl, y2_fl,output_label_dir,output_img_dir)
        
        img_flipud, x1_fu, x2_fu, y1_fu, y2_fu=flip(img,lbl_dataset,1) #flip_img, x1, x2, y1, y2 _UD
        create_imgnlbl(nombre+"_UD", etiqueta, img_flipud, x1_fu, x2_fu, y1_fu, y2_fu,output_label_dir,output_img_dir)
        
        img_r90,x1_90, x2_90, y1_90, y2_90 =rotate(img,lbl_dataset,0) #rot_img, x1, x2, y1, y2 _R90
        create_imgnlbl(nombre+"_R90", etiqueta, img_r90,x1_90, x2_90, y1_90, y2_90,output_label_dir,output_img_dir)
        
        img_r180,x1_180, x2_180, y1_180, y2_180 =rotate(img,lbl_dataset,1) #rot_img, x1, x2, y1, y2 _R180
        create_imgnlbl(nombre+"_R180", etiqueta, img_r180,x1_180, x2_180, y1_180, y2_180,output_label_dir,output_img_dir)
        
        img_r270,x1_270, x2_270, y1_270, y2_270 =rotate(img,lbl_dataset,2) #rot_img, x1, x2, y1, y2 _R270
        create_imgnlbl(nombre+"_R270", etiqueta, img_r270,x1_270, x2_270, y1_270, y2_270,output_label_dir,output_img_dir)
        
        # img_re=rand_erasing(img,lbl_dataset,2) #img _RE
        # create_imgnlbl(nombre+"_RE", etiqueta, img_re, x1, x2, y1, y2)