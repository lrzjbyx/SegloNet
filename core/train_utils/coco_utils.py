import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import torchvision



def coco_remove_images_without_annotations(dataset, ids):
    """
    删除coco数据集中没有目标，或者目标面积非常小的数据
    refer to:
    https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
    :param dataset:
    :param cat_list:
    :return:
    """
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False

        return True

    valid_ids = []
    for ds_idx, img_id in enumerate(ids):
        ann_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.loadAnns(ann_ids)

        if _has_valid_annotation(anno):
            valid_ids.append(img_id)

    return valid_ids


def convert_coco_poly_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        # 如果mask为空，则说明没有目标，直接返回数值为0的mask
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks



def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0
    ann_id = 0
    dataset = {'images': [], 'categories': [], 'annotations': []}
    dataset["categories"] = [{
      "supercategory": "seal",
      "id": 1,
      "name": "seal",
      "keypoints": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9"
      ],
      "skeleton": [
        [
          0,
          1
        ],
        [
          0,
          9
        ],
        [
          0,
          8
        ],
        [
          9,
          1
        ],
        [
          9,
          8
        ],
        [
          1,
          8
        ],
        [
          1,
          2
        ],
        [
          1,
          7
        ],
        [
          8,
          7
        ],
        [
          2,
          8
        ],
        [
          2,
          3
        ],
        [
          2,
          7
        ],
        [
          2,
          6
        ],
        [
          6,
          7
        ],
        [
          3,
          7
        ],
        [
          3,
          4
        ],
        [
          3,
          6
        ],
        [
          3,
          5
        ],
        [
          4,
          5
        ],
        [
          5,
          6
        ],
        [
          4,
          6
        ]
      ]
    }]
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        image_info, annotation_infos = ds.coco_index(img_idx)

        dataset['images'].append(image_info)

        for ann in annotation_infos:
            ann["id"] = ann_id
            ann_id += 1
            dataset['annotations'].append(ann)


    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = "keypoints"
    return iou_types
