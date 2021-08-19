import os
import urllib.request

DATASETS = ['cityscapes', 'coco', 'detrac', 'kitti', 'oid', 'visual_genome']
DATASETS_URLS = {'cityscapes': {'train': '', 'val': '', 'tearm_and_conditions': ''},
                 'coco': {'train': 'http://images.cocodataset.org/zips/train2017.zip', 'val': 'http://images.cocodataset.org/zips/val2017.zip', 'tearm_and_conditions': 'By downloading this dataset, you agree to COCO\'s Terms of Use (https://cocodataset.org/#termsofuse).'},
                 'detrac': {'train': '', 'val': '', 'tearm_and_conditions': ''},
                 'kitti': {'train': '', 'val': '', 'tearm_and_conditions': ''},
                 'oid': {'train': '', 'val': '', 'tearm_and_conditions': ''},
                 'visual_genome': {'train': '', 'val': '', 'tearm_and_conditions': ''}}


def list_datasets():
    return DATASETS


def prepare_data(images, DATA_ROOT_PATH="/mnt"):
    dataset_list = {}
    for image in images:
        tmp = image['image_path'].split("/")
        tmp.pop()
        path = DATA_ROOT_PATH + "/".join(tmp)
        if tmp[-1] not in dataset_list:
            dataset_list[tmp[-1]] = {'isExist': False,
                                     'path': path,
                                     'missing': []}

        if os.path.isdir(path):
            dataset_list[tmp[-1]]['isExist'] = True
        if not os.path.exists(DATA_ROOT_PATH + image['image_path']):
            isSuccess = False
            if image['url']:
                print("Image", image['file_name'],
                      'is available online. Downloading..')
                try:
                    urllib.request.urlretrieve(image['url'], "00000001.jpg")
                    isSuccess = True
                    print(image['file_name'], "downloaded!")
                except:
                    isSuccess = False
                    print(image['file_name'], "download failed!")
            if not isSuccess:
                dataset_list[tmp[-1]]['missing'].append(image)
    for dataset in dataset_list:
        if not dataset_list[dataset]['isExist']:
            print(dataset, "is not exist! Please download and put it at",
                  dataset_list[dataset]['path'])
        else:
            if len(dataset_list[dataset]['missing']) > 0:
                print("The following images are not exists. Please download and put them at",
                      dataset_list[dataset]['path'], ":")
                print(",".join(dataset_list[dataset]['missing']))

    print("dataset folder structure should be prepare as bellow:")
    print("DATA_ROOT_PATH/data/image_dataset")
    print("  |-- dataset name")
    print("    |-- images")
    print("      |-- xxxxxx.jpg")
    print("      |-- xxxxx2.jpg")
    print("      |-- ...")
    print("    |-- annotations")
    print("      |-- annotation1.json")
