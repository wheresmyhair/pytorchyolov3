import config
import numpy as np
import os
from numpy.core.fromnumeric import ndim
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_siez=416, S=[13,26,52], C=54, transform=None):
        # Train csv or test csv
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors//3
        self.C = C
        # Maybe several boxes in the same cell that are all good at predicting. 
        # If iou > 0.5, we will ignore that one.
        self.ignore_iou_threshold = 0.5
    
    def __len__(self):
        return len(self.annotations)

    def __getitem(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        # [class, x, y, w, h], want [x, y, w, h, class]. So np.roll:
        boxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        # Bounding box transform with the image transformation:
        if self.transform:
            augmentations = self.transform(image=image, boxes=boxes)
            image = augmentations["image"]
            boxes = augmentations["boxes"]
        # 3 scale predictions in the model
        # Same number of anchors at each scale prediction
        # S = [13,26,52] in YOLO 
        # 6 values represents [probablity that is an object, x, y, w, h, class]
        targets = [torch.zeros((self.num_anchors//3, S, S, 6)) for S in self.S]
        for box in boxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            # The first one would be the best anchor
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            # Want each bounding box has three scales with three anchors
            # Should be [True, True, True] finally
            anchor_status = [False, False, False]
            for anc in anchor_indices:
                # Which scale does the given anchor belongs to?
                scale_index = anc // self.num_anchors_per_scale
                # Which anchor is the given scale using?
                anchor_on_scale = anc % self.num_anchors_per_scale
                S = self.S[scale_index]
                # Which y cell and x cell are the anchor associated with?
                i, j = int(S*y), int(S*x)
                anchor_taken = targets[scale_index][anchor_on_scale, i ,j, 0]
                if not anchor_taken and not anchor_status[scale_index]:
                    targets[scale_index][anchor_on_scale, i, j, 0] = 1
                    # What are the x and y value in the cell? 
                    # Should be a relative position between 0 and 1
                    x_cell, y_cell = S*x - j, S*y - i
                    width_cell, height_cell = (width * S, height * S)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_index][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_index][anchor_on_scale, i, j, 5] = int(class_label)
                    anchor_status[scale_index] = True
                elif not anchor_taken and iou_anchors[anc] > self.ignore_iou_threshold:
                    targets[scale_index][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        return image, tuple(targets)


            








        
