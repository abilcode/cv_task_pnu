# %%
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
import torch
import supervision as sv
import pytorch_lightning as pl
import cv2

# %%
from transformers import DetrImageProcessor, DetrForObjectDetection

# %%
import os
from pathlib import Path
import json
from PIL import Image

# %%
### CONFIG

CSV_PATH = Path("../data/train_solution_bounding_boxes.csv")
TRAIN_IMAGE_DIR = Path('../data/training_images')
OUTPUT_TRAIN_JSON_PATH = os.path.join(TRAIN_IMAGE_DIR, 'annotations_coco.json')

# %%
train = pd.read_csv(CSV_PATH)

# %%
CSV_PATH

# %%
"""
prepping coco dataset
"""

# %%
# Prepare coco format
coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "car",
            "supercategory": "vehicle"
        }
    ]
}

# %%
#Map the image filename to image id
image_id_map = {}
annotation_id = 1
image_id_counter = 1

images = train['image'].unique()

images

# %%
for img_name in images:
    img_path = os.path.join(TRAIN_IMAGE_DIR, img_name)
    if not os.path.exists(img_path):
        print(f'Warning: {img_path} not found!')
        continue
    with Image.open(img_path) as img:
        width, height = img.size
    # add image's info to coco
    image_info = {
        "file_name": img_name,
        "height": height,
        "width": width,
        "id": image_id_counter
    }
    coco['images'].append(image_info)
    image_id_map[img_name] = image_id_counter
    image_id_counter += 1

# %%
# process annotations
for idx, row in train.iterrows():
    img_name = row['image']
    if img_name not in image_id_map:
        continue # if the image is missing, then skip annotation
    x_min = row['xmin']
    y_min = row['ymin']
    x_max = row['xmax']
    y_max = row['ymax']

    # now to coco box format
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    annotation = {
        "id": annotation_id,
        "image_id": image_id_map[img_name],
        "category_id": 1, # car
        "bbox": [x_min, y_min, bbox_width, bbox_height],
        "area": bbox_width * bbox_height,
        "iscrowd": 0,
        "segmentation": []
    }
    coco['annotations'].append(annotation)
    annotation_id += 1

# %%
with open(OUTPUT_TRAIN_JSON_PATH, 'w') as f:
    json.dump(coco, f, indent = 4)
print(f'Coco annotations saved to {OUTPUT_TRAIN_JSON_PATH}')

# %%
"""
Training
"""

# %%


image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# %%
ANNOTATION_FILE_NAME = 'annotations_coco.json'

# %%
os.path.join(TRAIN_IMAGE_DIR, ANNOTATION_FILE_NAME)

# %%
import torchvision
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        train: bool = True
    ):
        annotation_file_path = os.path.join(TRAIN_IMAGE_DIR, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(TRAIN_IMAGE_DIR, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {
            "image_id": image_id,
            "annotations": annotations
        }
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors = 'pt')
        pixel_values = encoding['pixel_values'].squeeze()
        target = encoding['labels'][0]
        return pixel_values, target

# %%
TRAIN_DATASET = CocoDetection(
    image_directory_path = TRAIN_IMAGE_DIR, image_processor=image_processor,
    train = True)

# %%
print(f'Number of training examples: {len(TRAIN_DATASET)}')

# %%
image_ids = TRAIN_DATASET.coco.getImgIds()
image_id = random.choice(image_ids)
image_info = TRAIN_DATASET.coco.loadImgs(image_id)[0]
annotations = TRAIN_DATASET.coco.imgToAnns[image_id]
image_path = os.path.join(TRAIN_DATASET.root, image_info['file_name'])


image = cv2.imread(image_path)
if image is None:
    print(f"[ERROR] Failed to load image: {image_path}")
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    xyxy = []
    class_ids = []

    for ann in annotations:
        x, y, w, h = ann['bbox']
        xyxy.append([x, y, x + w, y + h])
        class_ids.append(ann['category_id'])

    categories = TRAIN_DATASET.coco.cats
    id2label = {k: v['name'] for k, v in categories.items()}
    labels = [id2label[cid] for cid in class_ids]

    detections = sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        class_id=np.array(class_ids),
        data={"label": labels}
    )

    box_annotator = sv.BoxAnnotator()
    annotated_image = box_annotator.annotate(scene=image, detections=detections)

    plt.figure(figsize=(8, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.show()

# %%
from torch.utils.data import DataLoader

def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors = 'pt')
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }

TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=4, shuffle=True)

# %%
class Detr(pl.LightningModule):

    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path="facebook/detr-resnet-50",
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step, and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation/loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        # DETR authors decided to use different learning rate for backbone
        # you can learn more about it here:
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L22-L23
        # - https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/main.py#L131-L139
        param_dicts = [
            {
                "params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },
        ]
        return torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER

# %%
model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

batch = next(iter(TRAIN_DATALOADER))
outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])

# %%
from pytorch_lightning import Trainer

# settings
MAX_EPOCHS = 10

trainer = Trainer( max_epochs=MAX_EPOCHS, gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=5)

trainer.fit(model)

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE

# %%
MODEL_PATH = 'car-object-detection-detr-finetuned_iter_1'
model.model.save_pretrained(MODEL_PATH)

# loading model
model = DetrForObjectDetection.from_pretrained(MODEL_PATH)
model.to(DEVICE)