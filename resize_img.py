import cv2
import os


path = 'path/to/all/3/folders'
out_path = 'output/image/path'
dir_list = os.listdir(path)
img_dir_dict = {}
for folder in dir_list:
    img_fldr_path = os.path.join(path,folder)
    img_list = os.listdir(img_fldr_path)
    img_dir_dict[img_fldr_path] = img_list

for foldr_path, img_list in img_dir_dict.items():
    for img in img_list:
        img_path = os.path.join(foldr_path,img)
        image = cv2.imread(img_path)
        image = cv2.resize(image,(412,412))
        output_img_path = os.path.join(out_path,img)
        cv2.imwrite(output_img_path,image)