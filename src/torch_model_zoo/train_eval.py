import datetime
import os
import time
from .coco_utils import DetectionPresetTrain, DetectionPresetEval
import random
import math
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from .coco_utils import get_coco, train_one_epoch, evaluate, save_on_master, inference
from .utils import plot_loss_mAP, mkdir
from .group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_transform(train, data_augmentation):
    return DetectionPresetTrain(data_augmentation) if train else DetectionPresetEval()


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_detection(num_classes, model='retinanet_resnet50_fpn'):
    if "rrcnn" in model:
        model = torchvision.models.detection.__dict__[model](pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    else:

        model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

        in_features = model.head.classification_head.conv[0][0].in_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head.num_classes = num_classes

        cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, - math.log((1 - 0.01) / 0.01))

        model.head.classification_head.cls_logits = cls_logits

    return model


def print_models():
    print("TILL NOW CONSISTS MODELS: \n"
          "RetinaNet: \n"
          "--- retinanet_resnet50_fpn (default) \n"
          "FRCNN Series: \n"
          "--- fasterrcnn_resnet50_fpn \n"
          "--- fasterrcnn_mobilenet_v3_large_fpn \n"
          "--- fasterrcnn_mobilenet_v3_large_320_fpn \n"

          "BELOW MODELS WILL BE ADDED SOON: \n"
          "SSD: \n"
          "--- ssd300_vgg16 \n"
          "ssdlite320_mobilenet_v3_large \n"
          "MaskRCNN: \n"
          "--- maskrcnn_resnet50_fpn \n"
          "KPRCNN: \n"
          "--- keypointrcnn_resnet50_fpn")


def train_eval_pipeline(params):
    print("SETTED PARAMS:")
    for k, v in params.items():
        print(k, " : ", v)

    print(print_models())

    # load params
    epochs = params["EPOCHS"]
    cat_nms = params['CAT_NMS']
    lr = params['LEARNING_RATE']
    test_only = params["TEST_ONLY"]
    model_pth = params['WEIGHT_PTH']
    setup_seed(params['SETUP_SEED'])
    output_dir = params['OUTPUT_DIR']
    batch_size = params['BATCH_SIZE']
    start_epoch = params["START_EPOCH"]
    num_classes = params['NUM_CAT'] + 1

    mkdir(output_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("Loading data")
    dataset_test = get_coco(params['IMGS_PATH'], params['TEST_ANNO'], get_transform(False, 'hflip'))

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=4, collate_fn=collate_fn
    )

    print("Creating model")
    has_anno = False
    if test_only:
        model_saved = torch.load(model_pth)
        if has_anno:
            evaluate(model_saved, data_loader_test, device=device)
        for img_num, (img, _) in enumerate(dataset_test):
            inference(model_saved, img, img_num, device, threshold=0.7, cat_nms=cat_nms)
        return
    else:
        dataset_train = get_coco(params['IMGS_PATH'], params['TRAIN_ANNO'], get_transform(True, 'hflip'), train=True)
        train_sampler = torch.utils.data.RandomSampler(dataset_train)
        group_ids = create_aspect_ratio_groups(dataset_train, k=3)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size)

        data_loader = torch.utils.data.DataLoader(
            dataset_train, batch_sampler=train_batch_sampler, num_workers=4, collate_fn=collate_fn
        )
        model = get_model_detection(num_classes, model=params['MODEL'])

    model.to(device)

    model_without_ddp = model

    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=0.0001)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)

    print("Start training")
    start_time = time.time()
    mAP = []
    loss_metrics = {}

    for epoch in range(start_epoch, epochs):
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=20)

        if epoch == start_epoch:
            for i, metric in enumerate(metric_logger.meters):
                if i == 2 or i == 3:
                    loss_metrics[str(metric)] = []
        for k, v in loss_metrics.items():
            loss_metrics[k].append(list(getattr(metric_logger, k).deque))

        lr_scheduler.step()
        if output_dir and epoch == epochs - 1:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epochs,
            }
            save_on_master(checkpoint, os.path.join(output_dir, "final.pth"))
            save_on_master(checkpoint, os.path.join(output_dir, "ckpt_final.pth"))

        # evaluate after every epoch
        _, mAP_rcall_stats = evaluate(model, data_loader_test, device=device)
        mAP.append(int(mAP_rcall_stats[0] * 1000) / 10)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    for img_num, (img, _) in enumerate(dataset_test):
        inference(model, img, img_num, device, cat_nms=cat_nms)
    plot_loss_mAP(loss_metrics, mAP)
    print("Training time {}".format(total_time_str))
