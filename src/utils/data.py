import os

DATASETS = ['cityscapes', 'coco', 'detrac', 'kitti', 'oid', 'visual_genome']


def list_datasets():
    return DATASETS


def prepare_data(images):
    dataset_list = {}
    for image in images:
        tmp = image['image_path'].split("/")
        tmp.pop()
        path = "/".join(tmp)
        if tmp[-1] not in dataset_list:
            dataset_list[tmp[-1]] = {'isExist': False,
                                     'path': path,
                                     'missing': []}

        if os.path.isdir(path):
            dataset_list[tmp[-1]]['isExist'] = True
        if not os.path.exists(image['image_path']):
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
