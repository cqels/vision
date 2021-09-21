import os
import urllib.request
import requests

DATASETS = ['cityscapes', 'coco', 'detrac', 'kitti', 'oid', 'visual_genome']
DATASETS_URLS = {'cityscapes': {'train': '', 'val': '', 'tearm_and_conditions': ''},
                 'coco': {'train': 'http://images.cocodataset.org/zips/train2017.zip', 'val': 'http://images.cocodataset.org/zips/val2017.zip', 'tearm_and_conditions': 'By downloading this dataset, you agree to COCO\'s Terms of Use (https://cocodataset.org/#termsofuse).'},
                 'detrac': {'train': '', 'val': '', 'tearm_and_conditions': ''},
                 'kitti': {'train': '', 'val': '', 'tearm_and_conditions': ''},
                 'oid': {'train': '', 'val': '', 'tearm_and_conditions': ''},
                 'visual_genome': {'train': '', 'val': '', 'tearm_and_conditions': ''}}


def list_datasets():
    return DATASETS


def prepare_data(images, DATA_ROOT_PATH=None):
    if not DATA_ROOT_PATH:
        print("DATA path did not set! Path will set default at /tmp")
        DATA_ROOT_PATH = "/tmp"
    dataset_list = {}
    is_ok = True
    for image in images:
        tmp = image['image_path'].split("/")
        tmp.pop()
        path = DATA_ROOT_PATH + "/".join(tmp)
        dataset = tmp[-1]
        if dataset not in dataset_list:
            dataset_list[dataset] = {'path': path,
                                     'missing': []}

        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        if not os.path.exists(DATA_ROOT_PATH + image['image_path']):
            isSuccess = False
            if dataset == "visual_genome":
                # perform download
                pass
            else:
                if image['url']:
                    print("Image", image['file_name'],
                          'is available online. Downloading..')
                    try:
                        urllib.request.urlretrieve(image['url'], )
                        isSuccess = True
                        print(image['file_name'], "downloaded!")
                    except:
                        isSuccess = False
                        print(image['file_name'], "download failed!")
            if not isSuccess:
                dataset_list[tmp[-1]]['missing'].append(image['file_name'])
    for dataset in dataset_list:
        if len(dataset_list[dataset]['missing']) > 0:
            is_ok = False
            print("\nThe following images of the ", dataset, "dataset are not exists. Please download and put them at",
                  dataset_list[dataset]['path'], ":")
            print(", ".join(dataset_list[dataset]['missing']))
            print("")
    if not is_ok:
        print("\nDataset folder structure should be prepare as bellow:")
        print("DATA_ROOT_PATH/data/image_dataset")
        print("  |-- dataset name")
        print("    |-- images")
        print("      |-- xxxxxx.jpg")
        print("      |-- xxxxx2.jpg")
        print("      |-- ...")
        print("    |-- annotations")
        print("      |-- annotation1.json")
