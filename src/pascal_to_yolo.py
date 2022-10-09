import xml.etree.ElementTree as ET
import glob
import os
import json

def xml_to_yolo_bbox(bbox, w, h):
    """
    Description: Convert pascal labels to yolo format 
    bbox = list of bounding box coordinates(list)
    w = width of bounding box(int)
    h = height of bounding box(int)
    """
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]


def convert_pascal_to_yolo(input_dir,output_dir,image_dir):
    """
    Description: Convert pascal labels to yolo format 
    input_dir = xml label Directory path(str)
    output_dir = output txt format label directory(str)
    image_dir = image directory path(str)
    """
    classes = []
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    files = glob.glob(os.path.join(input_dir, '*.xml'))
    # loop through each 
    for fil in files:
        basename = os.path.basename(fil)
        filename = os.path.splitext(basename)[0]
        # check if the label contains the corresponding image file
        if os.path.exists(os.path.join(image_dir, f"{filename}.jpg")) == False :
            if os.path.exists(os.path.join(image_dir, f"{filename}.JPG")):
                if os.path.exists(os.path.join(image_dir, f"{filename}.png")):
                    print(f"{filename} image does not exist!")
                    continue

        result = []

        # parse the content of the xml file
        tree = ET.parse(fil)
        root = tree.getroot()
        width = int(root.find("size").find("width").text)
        height = int(root.find("size").find("height").text)

        for obj in root.findall('object'):
            label = obj.find("name").text
            # check for new classes and append to list
            if label not in classes:
                classes.append(label)
            index = classes.index(label)
            pil_bbox = [int(x.text) for x in obj.find("bndbox")]
            yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
            # convert data to string
            bbox_string = " ".join([str(x) for x in yolo_bbox])
            result.append(f"{index} {bbox_string}")

        if result:
            # generate a YOLO format text file for each xml file
            with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(result))

    with open('classes.txt', 'w', encoding='utf8') as f:
        f.write(json.dumps(classes))


