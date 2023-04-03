"""
Copyright (C) 2021 Microsoft Corporation
"""
import itertools
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional, Union, List, Dict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def add_margin(original_roi: np.ndarray, margin: Union[int, float]) -> np.ndarray:
    roi = original_roi.copy()
    if isinstance(margin, int):
        m = margin
    elif isinstance(margin, float):
        m = roi.max() * margin
    else:
        logging.warning(f"invalid type of margin {margin}, will replace it with int zero")
        m = 0
    roi[:2] -= m
    roi[2:] += m
    return roi


def parse_xml(xml_file: Union[str, Path], min_row_required: int = -1, min_col_required: int = -1) -> List[Dict]:
    """

    Args:
        xml_file: labelMe annotations created by CVAT
    Returns: List[Dict]
        each Dict consist of the following keys:
            "table": [1, 4] array of coordinate of the table in absolute LTRB format
            "table_structures": [K, 4] array of coordinate of the table_structures in absolute LTRB format
            "image_name": name of the image
            "xml_name": name of the xml
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name: str = root.find("filename").text

    # use defaultdict to separate table and table_structure with one loop
    objects = defaultdict(list)
    for object_ in root.iter('object'):
        name: str = object_.find("name").text

        if name != "table" and name != "table_structure":
            continue

        poly = object_.find("polygon")
        pts = poly.findall("pt")

        xs = [float(pt.find("x").text) for pt in pts]
        x_min = float(min(xs))
        x_max = float(max(xs))

        ys = [float(pt.find("y").text) for pt in pts]
        y_min = float(min(ys))
        y_max = float(max(ys))

        if name == "table_structure":
            type_: str = object_.find("type").text
            assert type_ == "bounding_box"
            label: str = object_.find("attributes").text.split("=")[-1]

            objects[name].append({
                "box": [x_min, y_min, x_max, y_max],
                "label": label,
            })
        else:
            objects[name].append({
                "box": [x_min, y_min, x_max, y_max],
            })

    # convert to numpy array
    tables_arr: np.ndarray = np.asarray([o["box"] for o in objects["table"]], dtype=np.float32)
    table_structures_arr: np.ndarray = np.asarray([o["box"] for o in objects["table_structure"]], dtype=np.float32)
    labels: List[str] = [o["label"] for o in objects["table_structure"]]

    # return empty list if either table or table_structure is empty
    if len(tables_arr) < 1 or len(table_structures_arr) < 1:
        logging.warning(f"{xml_file.name} has no table/table_structure")
        return []

    table_structures_centres_arr: np.ndarray = np.zeros((table_structures_arr.shape[0], 2), dtype=table_structures_arr.dtype)
    table_structures_centres_arr[:, 0] = 0.5 * (table_structures_arr[:, 0] + table_structures_arr[:, 2])
    table_structures_centres_arr[:, 1] = 0.5 * (table_structures_arr[:, 1] + table_structures_arr[:, 3])

    # Let M be # of table instances and N be that of table_structure instances
    # all mask below are of shape (M, N)
    left_bounded: np.ndarray = tables_arr[:, None, 0] < table_structures_centres_arr[:, 0]
    right_bounded: np.ndarray = table_structures_centres_arr[:, 0] < tables_arr[:, None, 2]
    top_bounded: np.ndarray = tables_arr[:, None, 1] < table_structures_centres_arr[:, 1]
    bottom_bounded: np.ndarray = table_structures_centres_arr[:, 1] < tables_arr[:, None, 3]

    assert all([x.ndim == 2 for x in [left_bounded, right_bounded, top_bounded, bottom_bounded]])
    # reduce the (M, N, 4) mask along the last axis with prod ops as logical AND ops
    is_inside = np.prod(np.dstack([left_bounded, right_bounded, top_bounded, bottom_bounded]), -1).astype(bool)

    results: List[Dict] = []
    for table_idx in range(is_inside.shape[0]):
        table_region = tables_arr[table_idx]
        table_structures_instances = table_structures_arr[is_inside[table_idx]]
        instance_labels = [label for label, keep in zip(labels, is_inside[table_idx]) if keep]
        assert len(instance_labels) == table_structures_instances.shape[0]
        c = Counter(instance_labels)
        if c["row"] < min_row_required or c["column"] < min_col_required:
            continue

        results.append({
            "table": table_region,
            "table_structures": table_structures_instances,
            "labels": instance_labels,
            "image_name": image_name,
            "xml_name": xml_file.name
        })

    return results


class CVATLabelMeTableStructure(Dataset):
    def __init__(
            self,
            root: Union[str, Path],
            transforms=None, max_size=None, do_crop=True, make_coco=False,
            include_eval=False, max_neg=None, negatives_root=None, xml_fileset="filelist.txt",
            image_extension='.png', class_map: Dict[str, int] = None,
            margin: Union[float, int] = 0.1,
            min_row_required: int = -1,
            min_col_required: int = -1,
            image_dir: str = "images",
            xml_dir: str = "annotations",
    ):
        """

        Args:
            root: path to the dataset root dir
            image_dir: directory of images relative to root, default: "images"
            xml_dir: directory of xml relative to root, default: "annotations"
            augmentations: augmentation to be used
            debug: if True, then return boxes in absolute LTRB format for easier visualization.
                Else normalized to relative cxcywh  format. Default: False
        """
        self.root = root if isinstance(root, Path) else Path(root)
        self.transforms = transforms
        self.do_crop = do_crop
        self.make_coco = make_coco
        self.image_extension = image_extension
        self.include_eval = include_eval
        self.class_map = class_map
        self.class_list = list(class_map)
        self.class_set = set(class_map.values())
        self.class_set.remove(class_map['no object'])
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.margin = margin
        self.min_row_required = min_row_required
        self.min_col_required = min_col_required

        annotation_files = [x for x in self.root.joinpath(self.xml_dir).iterdir() if x.exists()]
        assert len(annotation_files) > 0

        annotations: List[List[Dict]] = [parse_xml(
            f, min_row_required=self.min_row_required, min_col_required=self.min_col_required
        ) for f in annotation_files]
        # flattening
        assert all([isinstance(x, list) for x in annotations])
        annotations: List[Dict] = [ann for anns in annotations for ann in anns if ann]
        assert all([isinstance(x, dict) for x in annotations])
        self.annotations: List[Dict] = annotations

        if self.make_coco:
            self.dataset = {}
            self.dataset['images'] = [{'id': idx} for idx, _ in enumerate(self.annotations)]
            self.dataset['annotations'] = []
            ann_id = 0
            for image_id, ann in enumerate(self.annotations):
                img_path = self.root.joinpath(self.image_dir, ann["image_name"])
                roi = add_margin(ann["table"], self.margin)
                original = Image.open(img_path).convert("RGB").crop(roi.astype(int))
                orig_w, orig_h = original.size
                w, h = orig_w, orig_h
                # load boxes and labels
                boxes = np.concatenate([ann["table"][None, :], ann["table_structures"]], 0)
                offsets = np.tile(roi[:2], 2)
                assert offsets.ndim == 1
                boxes -= offsets[None, :]
                boxes[:, ::2] = boxes[:, ::2].clip(0, orig_w)
                boxes[:, 1::2] = boxes[:, 1::2].clip(0, orig_h)
                labels = np.asarray([0] + [self.class_map[label] for label in ann["labels"]], dtype=np.int64)

                # Reduce class set
                keep_indices = [idx for idx, label in enumerate(labels) if label in self.class_set]
                boxes = [boxes[idx] for idx in keep_indices]
                labels = [labels[idx] for idx in keep_indices]

                for bbox, label in zip(boxes, labels):
                    ann = {'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                           'iscrowd': 0,
                           'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                           'category_id': label,
                           'image_id': image_id,
                           'id': ann_id,
                           'ignore': 0,
                           'segmentation': []}
                    self.dataset['annotations'].append(ann)
                    ann_id += 1
            self.dataset['categories'] = [{'id': idx} for idx in self.class_list[:-1]]

            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getImgIds(self):
        return range(len(self.annotations))

    def getCatIds(self):
        return range(10)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == len(areaRng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if
                                                   ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]

            ids = [ann['id'] for ann in anns]
        return ids

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx: int):
        # load images
        ann: Dict[str] = self.annotations[idx]
        img_path = self.root.joinpath(self.image_dir, ann["image_name"])
        roi = add_margin(ann["table"], self.margin)
        original = Image.open(img_path).convert("RGB").crop(roi.astype(int))
        orig_w, orig_h = original.size
        w, h = orig_w, orig_h
        # load boxes and labels
        boxes = np.concatenate([ann["table"][None, :], ann["table_structures"]], 0)
        offsets = np.tile(roi[:2], 2)
        assert offsets.ndim == 1
        boxes -= offsets[None, :]
        boxes[:, ::2] = boxes[:, ::2].clip(0, orig_w)
        boxes[:, 1::2] = boxes[:, 1::2].clip(0, orig_h)
        labels = np.asarray([0] + [self.class_map[label] for label in ann["labels"]], dtype=np.int64)

        num_objs = boxes.shape[0]
        # Create target
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] =  torch.as_tensor(labels,  dtype=torch.int64)
        target["image_id"] = torch.as_tensor([idx])
        target["area"] = boxes[:, 2] * boxes[:, 3]  # COCO area
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        img_tensor, target = self.transforms(original, target)

        return img_tensor, target

    @staticmethod
    def where_rb_greater_than_lt(boxes):
        return (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
