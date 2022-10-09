import cv2
import numpy as np

def xml2dim(labels_dataset):
    """
    Description: Clean and organize the given labels_dataset
    labels_dataset = list
    """
    lbl=labels_dataset[0][0] #category
    a=float(labels_dataset[0][1]) #box center X
    b=float(labels_dataset[0][2]) #box center Y
    bbox_width=float(labels_dataset[0][3]) #box width
    bbox_height=float(labels_dataset[0][4]) #box height
    return lbl, a, b, bbox_width, bbox_height

def shear(img,labels_dataset,shx,shy):
    """
    Description: Shear transformation and its corresponding bounding box coordinates
    img= MxN list
    labels_dataset = list
    shx, shy = float
    """
    M = np.float32([[1, shx, 0],[shy, 1  , 0],[0, 0  , 1]])
    sheared_img = cv2.warpPerspective(img,M,(int(img.shape[1]*(1+shx)),int(img.shape[0]*(1+shy))))
    _,a,b,bbox_width,bbox_height=xml2dim(labels_dataset)
    h,w,_=img.shape
    x1, y1 = np.round_((a-bbox_width/2)*w,0), np.round_((b-bbox_height/2)*h,0)
    x2, y2 = np.round_((a+bbox_width/2)*w,0), np.round_((b+bbox_height/2)*h,0)
    #Affine Transformation
    u1, u2=int(M[0][0]*x1+M[0][1]*y1+M[0][2]), int(M[0][0]*x2+M[0][1]*y2+M[0][2])
    v1, v2=int(M[1][0]*x1+M[1][1]*y1+M[1][2]), int(M[1][0]*x2+M[1][1]*y2+M[1][2])
    return sheared_img, u1, u2, v1, v2
def flip(img,labels_dataset,mode):
    """
    Description: Flip an image and its corresponding bounding box coordinates
    img= MxN list
    labels_dataset = list
    modes: 
        0 = left to right
        1 = up to down 
    """
    _,a,b,bbox_width,bbox_height=xml2dim(labels_dataset)
    h,w,_=img.shape
    if mode==0:
        a=1-a
        flip_img=cv2.flip(img,1)
    elif mode==1:
        b=1-b
        flip_img=cv2.flip(img,0)
    else:
        print('Your selected mode does not exist')
        return
    x1, y1 = np.round_((a-bbox_width/2)*w,0), np.round_((b-bbox_height/2)*h,0)
    x2, y2 = np.round_((a+bbox_width/2)*w,0), np.round_((b+bbox_height/2)*h,0)
    return flip_img, x1, x2, y1, y2
def rotate(img,labels_dataset,mode):
    """
    Description: Rotate an image and its corresponding bounding box coordinates
    img= MxN list
    labels_dataset = list
    modes: 
        0 = 90° counterclockwise
        1 = 180°
        2 = 270° counterclowise / 90° clockwise
    """
    _,a,b,bbox_width,bbox_height=xml2dim(labels_dataset)
    h,w,_=img.shape
    if mode==0:
        rot_img=cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        x1, y1 = np.round_((b-bbox_height/2)*h,0), np.round_((1-a-bbox_width/2)*w,0)
        x2, y2 = np.round_((b+bbox_height/2)*h,0), np.round_((1-a+bbox_width/2)*w,0)
    elif mode==1:
        rot_img=cv2.rotate(img, cv2.cv2.ROTATE_180)
        a=1-a
        b=1-b
        x1, y1 = np.round_((a-bbox_width/2)*w,0), np.round_((b-bbox_height/2)*h,0)
        x2, y2 = np.round_((a+bbox_width/2)*w,0), np.round_((b+bbox_height/2)*h,0)
    elif mode==2:
        rot_img=cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        x1, y1 = np.round_((1-b-bbox_height/2)*h,0), np.round_((a-bbox_width/2)*w,0)
        x2, y2 = np.round_((1-b+bbox_height/2)*h,0), np.round_((a+bbox_width/2)*w,0)
    else:
        print('Your selected mode does not exist')
        return
    return rot_img, x1, x2, y1, y2