import shutil
import os

def split_data(image_directory,output_labels_directory,dataset_path):
    """
    Description: Split image and labels 
    image_directory = images Directory path(str)
    output_labels_directory = output label directory(str)
    dataset_path = image,label combined data directory path(str)
    """
    image_directory = './samples/images/train'
    output_labels_directory = './samples/output_labels/'
    dataset_path = './dataset_xml_format'
    if os.path.exists(image_directory) != True:
        os.mkdir(image_directory)
    if os.path.exists(output_labels_directory) != True:
        os.mkdir(output_labels_directory)
    list_data = os.listdir('./image_aug/dataset_xml_format')
    for name in list_data:
        if name[-3:] == 'png' or name[-3:] == 'jpg'or name[-3:] == 'JPG':
            shutil.move(os.path.join(dataset_path,name), os.path.join(image_directory,name))
        else:
            shutil.move(os.path.join(dataset_path,name), os.path.join(output_labels_directory,name))