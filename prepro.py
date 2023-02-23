import glob
import os
import random
import shutil
import xml.etree.ElementTree as ET

from tqdm import tqdm
import argparse

random.seed(12)

classes = ["fire", "smoke"]


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(in_file, out_file):
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(out_file, "w") as fout:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes:  # or int(difficult)==1 不关心difficult
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text),
                 float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            fout.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def transfer_files(source_files, mode):
    """

    :param source_files: train_img_files
    :param mode: "images/train", "labels/val", ...
    :return:
    """
    for source_file in tqdm(source_files):
        target_dir = os.path.join(args.data_out_root, mode)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        shutil.copy(source_file, target_dir)


def before_execute():
    """
    clean generated dirs and files
    :return:
    """
    generated_dirs = [
        os.path.join(args.data_root, "labels"),
        os.path.join(args.data_out_root)
    ]
    for _dir in generated_dirs:
        if os.path.isdir(_dir):
            print("delete directory:", _dir)
            shutil.rmtree(_dir)


def voc2yolo():
    """
    voc Annotations to yolo Annotations
    :return:
    """
    # all_xml_files = glob.glob(os.path.join(args.data_root, "Annotations/*.xml"))
    all_xml_files = glob.glob(os.path.join(args.data_root, "annotations/*.xml"))
    # print(all_xml_files)
    for xml_file in tqdm(all_xml_files):
        target_dir = os.path.join(args.data_root, "labels")
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        filename = os.path.basename(xml_file).replace(".xml", "")
        yolo_file = os.path.join(target_dir, filename + ".txt")
        convert_annotation(xml_file, yolo_file)


def split_data():
    """
    split data to train and val
    :return:
    """
    # all_img_files = glob.glob(os.path.join(args.data_root, "JPEGImages/*.jpg"))
    all_img_files = glob.glob(os.path.join(args.data_root, "images/*.jpg"))
    all_label_files = glob.glob(os.path.join(args.data_root, "labels/*.txt"))
    all_img_files = sorted(all_img_files)
    all_label_files = sorted(all_label_files)
    assert len(all_img_files) == len(all_label_files)

    train_img_files = []
    train_label_files = []
    val_img_files = []
    val_label_files = []
    for img_file, label_file in zip(all_img_files, all_label_files):
        randint = random.randint(1, 100)
        if randint > 20:
            train_img_files.append(img_file)
            train_label_files.append(label_file)
        else:
            val_img_files.append(img_file)
            val_label_files.append(label_file)

    if args.demo:
        train_img_files = train_img_files[:5]
        train_label_files = train_label_files[:5]
        val_img_files = train_img_files
        val_label_files = train_label_files

    transfer_files(train_img_files, "images/train")
    transfer_files(train_label_files, "labels/train")
    transfer_files(val_img_files, "images/val")
    transfer_files(val_label_files, "labels/val")


def main():
    before_execute()
    voc2yolo()
    split_data()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, help="data directory downloaded")
    parser.add_argument("--data_out_root", type=str, default="./datasets/fire",
                        help="directory where data processed saving with train/val split")
    parser.add_argument("--demo", action="store_true", help="whether execute in demo mode")
    args = parser.parse_args()

    main()
