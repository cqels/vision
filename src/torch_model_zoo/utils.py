import numpy as np
from sklearn.model_selection import train_test_split
import funcy
from tabulate import tabulate
import coloredlogs, logging
from glob import glob
import itertools, os, json, urllib.request
from tqdm import tqdm
from os.path import join as opj
import cv2

coloredlogs.install()
logging.basicConfig(format='[%(asctime)s : %(message)s %(filename)s]',
                    log_colors='green', loglevel=logging.ERROR)


def check_instances_categories(file, annotations, class_names):
    """
    #### category index should start from 1
    """
    num_classes = len(class_names)
    hist_bins = np.arange(num_classes + 1)
    histogram = np.zeros((num_classes,))
    for anno in annotations:
        classes = np.asarray(
            [anno["category_id"] - 1]
        )
        if len(classes):
            assert classes.min() >= 0, f"Got an invalid category_id={classes.min()}"
            assert (
                    classes.max() < num_classes
            ), f"Got an invalid category_id={classes.max()} for a dataset of {num_classes} classes"
        histogram += np.histogram(classes, bins=hist_bins)[0]

    N_COLS = min(6, len(class_names) * 2)

    def short_name(x):
        # make long class names shorter. useful for lvis
        if len(x) > 13:
            return x[:11] + ".."
        return x

    data = list(
        itertools.chain(*[[short_name(class_names[i]), int(v)] for i, v in enumerate(histogram)])
    )
    total_num_instances = sum(data[1::2])
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    if num_classes > 1:
        data.extend(["total", total_num_instances])
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "#instances"] * (N_COLS // 2),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    logging.basicConfig(format='[%(asctime)s : %(message)s %(filename)s]',
                        log_colors='green', loglevel=logging.INFO)

    logging.info('\n' + '\033[92m' + 'Categories and Instances in the ' + file + ':' + '\033[96m' + '\n' + table)


def save_coco(file, images, annotations, categories):
    check_instances_categories(file, annotations, [category['name'] for category in categories])
    with open(file, 'wt') as coco:
        json.dump({'images': images, 'annotations': annotations, 'categories': categories}, coco, indent=2,
                  sort_keys=False)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def dataset_split(annotation_file, train_val_test, ratio):
    with open(annotation_file, 'rt') as annotations:
        coco = json.load(annotations)
        images = coco['images']

        annotations = coco['annotations']
        categories = coco['categories']

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)
        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        images_trn_val_tst = {}

        images_trn_val_tst["train_val"], images_trn_val_tst["test"] = train_test_split(images,
                                                                                       train_size=ratio)
        images_trn_val_tst["train"], images_trn_val_tst["val"] = train_test_split(
            images_trn_val_tst["train_val"], train_size=ratio)

        for set_nms in train_val_test:
            img_ids = images_trn_val_tst[set_nms.split('.')[0]]
            save_coco(opj(os.path.abspath(os.path.dirname(annotation_file) + os.path.sep + "."), set_nms),
                      img_ids, filter_annotations(annotations, img_ids),
                      categories)


def check_download_images(imgs_info):
    download_error = {}
    for num, img_info in enumerate(tqdm(imgs_info)):
        image_path = img_info['image_path']
        if isinstance(img_info['url'], str):
            image_url = [''.join(img_info['url'])]
        else:
            image_url = img_info['url']
        download_sucess = False
        f_path = os.path.abspath(os.path.dirname(image_path) + os.path.sep + ".")
        if os.access(image_path, mode=os.R_OK):
            continue
        else:
            os.makedirs(f_path, exist_ok=True)
            for url in image_url:
                try:
                    urllib.request.urlretrieve(url, image_path)
                    download_sucess = True
                    break
                except Exception as e:
                    continue
            if download_sucess is False:
                download_error[img_info['file_name']] = image_path
                continue
        img = cv2.imread(image_path, -1)
        dim = (img.shape[1], img.shape[0])
        dim_origin = (img_info['width'], img_info['height'])
        if dim != dim_origin:
            img = cv2.resize(img, dim_origin, cv2.INTER_AREA)
            cv2.imwrite(image_path, img)
    images_with_expired_urls = list(download_error.values())
    if len(images_with_expired_urls) != 0:
        for img_dir in images_with_expired_urls:
            print('\n' + 'The image " ' + img_dir + ' " is not exist.')
        logging.info('\n' + 'You need to download those images by yourself to: ' + f_path + '\n')
    else:
        logging.info('\n' + 'All the needed images have been downloaded to: ' + f_path + '\n')

    # hints: provide with links and tell users which datasets they need to download and where to download them


def check_anno_index(path_to_anno):
    with open(path_to_anno) as coco_format_anno:
        anno = json.load(coco_format_anno)
    annotations = anno['annotations']
    categories = anno['categories']
    index_start_zero = False
    if categories[0]['id'] != 0:
        return index_start_zero, anno
    else:
        index_start_zero = True
        for category in categories:
            category['id'] += 1
        for annotation in annotations:
            annotation['category_id'] += 1
    anno_sorted_index = {
        "images": anno['images'],
        "annotations": annotations,
        "categories": categories
    }
    return index_start_zero, anno_sorted_index


def checkpoint_verify(work_dir, ckpt_file=None):
    if ckpt_file is not None:
        ckpt_file = os.path.join(work_dir, ckpt_file)
    else:
        for ckpt_file in glob(work_dir + "best_bbox_mAP_epoch_*.pth"):
            if os.path.isfile(ckpt_file):
                return os.path.abspath(ckpt_file)
        ckpt_file = os.path.join(work_dir, "latest.pth")
    assert os.path.isfile(ckpt_file), '{} not exist'.format(ckpt_file)
    return os.path.abspath(ckpt_file)


def images_categories_distribution(path_to_anno):
    """
        analysis the images and categories distributions of mixedDatasets
        1. draw a pie figure for images distribution
        2. draw a histogram for categories distribution
        3. .. other better visualization and analysis for mixedDatasets
        4. could also be used to analysis the detected performance in different datasets
		Note: which need to the source of specific image
    """

    pass


def image_from_google_drive(img_info):
    """
		also need to the source of specific image
    """

    pass


def prepare_for_training(path_to_anno_mixedDatasets, anno):
    os.makedirs(os.path.abspath(os.path.join(path_to_anno_mixedDatasets, "..")), exist_ok=True)
    with open(path_to_anno_mixedDatasets, "w") as f:
        json.dump(anno, f)
    check_download_images(anno["images"])

    nms_categories = [category['name'] for category in anno['categories']]
    num_categories = len(nms_categories)

    return set_params(path_to_anno_mixedDatasets, num_categories, nms_categories)


def set_params(path_to_anno_mixedDatasets, num_categories, nms_categories):

    # set deafalt params
    lr = 0.005
    seed = 12345
    batch_size = 4
    start_epoch = 0
    total_epochs = 13
    eval_only = False
    output_dir = './save_model'
    model = 'fasterrcnn_resnet50_fpn'
    weight_pth = './save_model/final.pth'
    train_val_test = ['train.json', 'val.json', 'test.json']
    dataset_split(path_to_anno_mixedDatasets, train_val_test, ratio=0.8)

    params = {'imgs_path': '/data/image_dataset/coco', 'train_anno': 'data/mixedDatasets/train.json',
              'test_anno': 'data/mixedDatasets/val.json',
              'model': model, 'weight_pth': weight_pth,
              'output_dir': output_dir, 'batch_size': batch_size, 'learning_rate':lr,
              'epochs': total_epochs, 'num_cat': num_categories, 'test_only': eval_only,
              "setup_seed": seed, 'start_epoch': start_epoch,
              'cat_nms': nms_categories}

    print("SETTED PARAMS:")
    for k, v in params.items():
        print(k.upper(), " : ", v)

    print("Params which may need to be reset manually:"
          "imgs_path: path to your images"
          "train_anno: path to your training annotations"
          "test_anno: path to your test annotations"
          "num_cat: number of categories your would like to train"
          "cat_nms: name of each category")

    return params