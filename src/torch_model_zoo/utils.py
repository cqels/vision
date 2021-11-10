import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import funcy
from tabulate import tabulate
import coloredlogs, logging
from glob import glob
import itertools, os, json, urllib.request
from tqdm import tqdm
from os.path import join as opj
import cv2
from pycocotools.coco import COCO
from .coco_utils import color_val_matplotlib
from mpl_toolkits.axes_grid1 import host_subplot
import errno
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

coloredlogs.install()


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


def anno_filter(anno_path, filter_cat_nms):
    coco = COCO(anno_path)
    cat_ids = coco.getCatIds(catNms=filter_cat_nms)
    categoreis_info = coco.loadCats(cat_ids)
    cat_nms = [cat_name['name'] for cat_name in categoreis_info]
    for filter_cat_name in filter_cat_nms:
        if filter_cat_name not in cat_nms:
            filter_cat_nms.remove(filter_cat_name)
    if not filter_cat_nms:
        return filter_cat_nms
    img_ids = []
    for idx in cat_ids:
        img_ids += coco.getImgIds(catIds=idx)
    images_info = coco.loadImgs(list(set(img_ids)))
    ann_ids = list(set(coco.getAnnIds(imgIds=img_ids, catIds=cat_ids, iscrowd=None)))
    annotations_info = coco.loadAnns(ann_ids)
    cls_ids, coco_labels_inverse = [], {}
    for c in categoreis_info:
        new_cat_idx = len(cls_ids) + 1
        coco_labels_inverse[c['id']] = new_cat_idx
        cls_ids.append(new_cat_idx)
        c['id'] = new_cat_idx

    for anno in annotations_info:
        anno['category_id'] = coco_labels_inverse[anno['category_id']]

    filter_annos = {"images": images_info,
                    "annotations": annotations_info,
                    "categories": categoreis_info}
    return filter_annos


def prepare_for_training(path_to_anno_mixedDatasets, anno, filter_cat_nms=None):
    os.makedirs(os.path.abspath(os.path.join(path_to_anno_mixedDatasets, "..")), exist_ok=True)
    with open(path_to_anno_mixedDatasets, "w") as f:
        json.dump(anno, f)
    if filter_cat_nms:
        filter_result = anno_filter(path_to_anno_mixedDatasets, filter_cat_nms)
        if not filter_result:
            anno = filter_result
    check_download_images(anno["images"])
    nms_categories = [category['name'] for category in anno['categories']]
    num_categories = len(nms_categories)

    return set_params(path_to_anno_mixedDatasets, num_categories, nms_categories)


def set_params(path_to_anno_mixedDatasets, num_categories, nms_categories):
    # set deafalt params
    lr = 0.005
    seed = 65375
    batch_size = 4
    start_epoch = 0
    total_epochs = 13
    eval_only = False
    output_dir = 'save_model'
    model = 'retinanet_resnet50_fpn'
    weight_pth = 'save_model/final.pth'
    train_val_test = ['train.json', 'val.json', 'test.json']
    dataset_split(path_to_anno_mixedDatasets, train_val_test, ratio=0.8)

    params = {'IMGS_PATH': '/data/image_dataset/coco', 'TRAIN_ANNO': 'data/mixedDatasets/train.json',
              'TEST_ANNO': 'data/mixedDatasets/val.json',
              'MODEL': model, 'WEIGHT_PTH': weight_pth,
              'OUTPUT_DIR': output_dir, 'BATCH_SIZE': batch_size, 'LEARNING_RATE': lr,
              'EPOCHS': total_epochs, 'NUM_CAT': num_categories, 'TEST_ONLY': eval_only,
              "SETUP_SEED": seed, 'START_EPOCH': start_epoch,
              'CAT_NMS': nms_categories}

    print("SETTED PARAMS:")
    for k, v in params.items():
        print(k, " : ", v)

    print("Params which may need to be reset manually: \n"
          "IMGS_PATH: path to your images \n"
          "TRAIN_ANNO: path to your training annotations \n"
          "TEST_ANNO: path to your test annotations \n"
          "NUM_CAT: number of categories your would like to train \n"
          "CAT_NMS: name of each category")

    return params


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def draw_box(coordinates, img_raw, cat_dict):
    img = np.array(img_raw)
    win_name = ''
    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)

    dpi = fig.get_dpi()
    EPS = 1e-2
    width, height = img.shape[1], img.shape[0]
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')
    polygons = []
    color = []
    bbox_color = color_val_matplotlib((30, 188, 50))
    text_color = color_val_matplotlib((30, 188, 100))
    thickness = 2
    for coordinate in coordinates:
        xmin, ymin, xmax, ymax, catId = map(int, coordinate)
        if catId not in list(cat_dict.keys()):
            continue
        poly = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(bbox_color)
        label_text = cat_dict[catId]
        ax.text(
            xmin,
            ymin,
            f'{str(label_text)}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=13,
            verticalalignment='top',
            horizontalalignment='left')
    plt.imshow(img)
    p = PatchCollection(
        polygons, facecolor='none', edgecolors=color, linewidths=thickness)
    ax.add_collection(p)
    plt.show()


def show_annotation(annotations_dir, cat_nms, show_num=6, img_dir=None):
    coco = COCO(annotations_dir)
    cat_ids = coco.getCatIds(catNms=cat_nms)

    categories = coco.loadCats(cat_ids)
    cat_dict = {}
    for category in categories:
        cat_dict[category['id']] = category['name']
    imgIds = []
    for cat_id in cat_ids:
        imgIds.extend(coco.getImgIds(catIds=cat_id))
    list(set(imgIds))
    for i, imgId in enumerate(imgIds):
        if i == show_num:
            break
        img = coco.loadImgs(imgId)[0]
        dim = (img['width'], img['height'])
        if img_dir:
            image_path = os.path.join(img_dir, img['file_name'])
        else:
            image_path = img['image_path']
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=[], iscrowd=None)
        anns = coco.loadAnns(annIds)
        coordinates = []
        if not os.access(image_path, mode=os.R_OK):
            urllib.request.urlretrieve(img['url'], image_path)
        img_raw = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        img_raw = cv2.resize(img_raw, dim, cv2.INTER_AREA)
        for j in range(len(anns)):
            coordinate = anns[j]['bbox']
            coordinate[2] += coordinate[0]
            coordinate[3] += coordinate[1]
            coordinate.append(anns[j]['category_id'])
            coordinates.append(coordinate)
        draw_box(coordinates, img_raw, cat_dict)


def show_cat_distribution(annotations_dir, cat_nms):
    plt.figure(figsize=(16, 10))
    font_size = 20
    coco = COCO(annotations_dir)
    cats = coco.loadCats(coco.getCatIds(cat_nms))
    cat_nms = [cat['name'] for cat in cats]
    img_num = []
    bbox_num = []
    large_instances = []
    middle_instances = []
    small_instances = []

    for cat_name in cat_nms:
        catId = coco.getCatIds(catNms=cat_name)
        imgIds = coco.getImgIds(catIds=catId)
        annIds = coco.getAnnIds(imgIds=imgIds, catIds=catId, iscrowd=None)
        num_large = 0
        num_middle = 0
        num_small = 0
        for annId in annIds:
            area_size = coco.loadAnns(annId)[0]['area']
            if area_size > 96 * 96:
                num_large += 1
            elif area_size < 32 * 32:
                num_small += 1
            else:
                num_middle += 1
        img_num.append(len(imgIds))
        bbox_num.append(len(annIds))
        large_instances.append(num_large)
        middle_instances.append(num_middle)
        small_instances.append(num_small)

    # set params
    plt.rcParams['font.size'] = font_size
    plt.rcParams["axes.unicode_minus"] = False

    len_control = np.arange(len(cat_nms))
    bar_width = 0.35

    plt.bar(len_control, img_num, bar_width, align="center", color="c", label="num_images", alpha=0.6)
    plt.bar(len_control + bar_width, small_instances, bar_width, color="b", align="center", label="num_small_bbox",
            alpha=0.6)
    plt.bar(len_control + bar_width, middle_instances, bar_width, bottom=small_instances, color="m", align="center",
            label="num_middle_bbox", alpha=0.6)
    plt.bar(len_control + bar_width, large_instances, bar_width,
            bottom=np.sum([small_instances, middle_instances], axis=0), color="g", align="center",
            label="num_large_bbox",
            alpha=0.6)

    plt.xlabel("CATEGORY")
    plt.ylabel("NUMBER")
    rotation = 0
    if len(cat_nms) >= 10:
        rotation = 90
    plt.xticks(len_control + bar_width / 2, [cat_nm.capitalize() for cat_nm in cat_nms], fontsize=font_size,
               rotation=rotation)
    plt.legend(fontsize=font_size)
    plt.tight_layout()
    plt.show()


def plot_loss_mAP(loss_metrics, mAp):
    plt.figure(figsize=(16, 8))
    host = host_subplot(111)
    plt.subplots_adjust(right=0.8)
    par1 = host.twinx()

    # set labels
    host.set_xlabel("Epochs", fontsize=15)
    host.set_ylabel("Loss", fontsize=15)
    par1.set_ylabel("mAP (%)", fontsize=15)

    # plot curves
    for k, v in loss_metrics.items():
        epoch_loss_mean = []
        epoch_loss_std = []
        for epoch_loss in v:
            epoch_loss_mean.append(np.mean(epoch_loss))
            epoch_loss_std.append(np.std(epoch_loss))
        p = host.plot(range(len(epoch_loss_mean)), epoch_loss_mean, label=k)
        colour = p[-1].get_color()
        plt.fill_between(range(len(epoch_loss_mean)),
                         np.array(epoch_loss_mean) - np.array(epoch_loss_std),
                         np.array(epoch_loss_mean) + np.array(epoch_loss_std),
                         facecolor=colour, alpha=0.2)

    par1.plot(range(len(mAp)), mAp, label="mAP")
    host.legend(loc=5, fontsize=15)

    plt.tick_params(labelsize=15)
    plt.draw()
    plt.show()


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
