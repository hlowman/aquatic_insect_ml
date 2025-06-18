import os
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
)
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
import evaluate
import pycocotools.mask as mask_util
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Mask2FormerCocoDataset(Dataset):
    """
    Dataset for Mask2Former universal segmentation using COCO JSON format.
    Supports Roboflow and standard COCO exports. Expects images in an 'images/' subdirectory or the root,
    and supports both .jpg and .png images. Masks are generated for each category_id.
    """
    def __init__(
        self,
        data_dir: str,
        processor: Mask2FormerImageProcessor,
        annotation_file: str = "_annotations.coco.json",
        image_size: tuple = (1280, 1280)
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.image_size = image_size
        
        # Load COCO annotations
        with open(self.data_dir / annotation_file, "r") as f:
            self.annotations = json.load(f)
        
        # Map image file names to their annotations
        self.img_to_ann = {img['file_name']: [] for img in self.annotations['images']}
        for ann in self.annotations['annotations']:
            img_id = ann['image_id']
            img_info = next(item for item in self.annotations['images'] if item['id'] == img_id)
            self.img_to_ann[img_info['file_name']].append(ann)
        
        # Find images in 'images/' subdir or root, support .jpg and .png
        images_dir = self.data_dir / "images"
        search_dir = images_dir if images_dir.exists() else self.data_dir

        self.image_files = [f for f in search_dir.glob("*.jpg") if f.name in self.img_to_ann]
        self.image_files += [f for f in search_dir.glob("*.png") if f.name in self.img_to_ann]
        
        # id2label and label2id for Mask2Former
        self.id2label = {category['id']: category['name'] for category in self.annotations['categories']}
        self.label2id = {name: id for id, name in self.id2label.items()}

    def __len__(self):
        return len(self.image_files)

    def _create_mask_from_annotation(self, annotation, height, width):
        if 'segmentation' in annotation:
            if isinstance(annotation['segmentation'], list):
                rles = mask_util.frPyObjects(annotation['segmentation'], height, width)
                rle = mask_util.merge(rles)
            else:
                rle = annotation['segmentation']
            mask = mask_util.decode(rle)
            return mask
        return np.zeros((height, width), dtype=np.uint8)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")
        
        original_width, original_height = image.size
        
        image = image.resize(self.image_size, Image.Resampling.BILINEAR)
        annotations = self.img_to_ann[image_path.name]
        
        segmentation_map = np.zeros(self.image_size, dtype=np.uint8)
        for ann in annotations:
            original_mask = self._create_mask_from_annotation(ann, original_height, original_width)
            mask = Image.fromarray(original_mask).resize(self.image_size, Image.Resampling.NEAREST)
            segmentation_map[np.array(mask) > 0] = ann['category_id']
            
        inputs = self.processor(
            images=image,
            segmentation_maps=segmentation_map,
            return_tensors="pt"
        )

        # --- START OF FINAL CORRECTED CODE ---
        # This loop correctly handles all data types from the processor
        processed_inputs = {}
        for k, v in inputs.items():
            # For tensors like 'pixel_values', remove the batch dimension.
            if isinstance(v, torch.Tensor):
                processed_inputs[k] = v.squeeze(0)
            # For lists like 'mask_labels' and 'class_labels', extract the tensor from the list.
            elif isinstance(v, list) and len(v) > 0:
                processed_inputs[k] = v[0]
            # Handle any other cases by just passing the value through.
            else:
                processed_inputs[k] = v
        
        return processed_inputs
        # --- END OF FINAL CORRECTED CODE ---

import torch # Make sure torch is imported at the top

def instance_segmentation_collate_fn(features):
    """
    Custom collate function for instance segmentation.
    It stacks pixel_values and gathers labels and masks into lists.
    """
    # pixel_values are always the same size (3, H, W), so we can stack them
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    
    # For labels and masks, the number of instances per image varies.
    # We simply collect them in a list instead of trying to stack.
    # The model is designed to handle this list format during training.
    mask_labels = [f["mask_labels"] for f in features]
    class_labels = [f["class_labels"] for f in features]
    
    return {
        "pixel_values": pixel_values,
        "mask_labels": mask_labels,
        "class_labels": class_labels,
    }

class Mask2FormerFinetuner:
    """
    Trainer for Mask2Former universal segmentation on COCO-style datasets.
    Handles model, processor, dataset, and training loop.
    """
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        annotation_file: str = "_annotations.coco.json", # <-- NEW PARAMETER
        output_dir: str = "mask2former-finetuned",
        model_checkpoint: str = "facebook/mask2former-swin-base-coco-instance",
        batch_size: int = 2,
        num_epochs: int = 50,
        learning_rate: float = 5e-5,
        use_gpu: bool = False,
        image_size: tuple = (1280, 1280)
    ):
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.annotation_file = annotation_file # <-- STORE IT
        self.output_dir = output_dir
        self.model_checkpoint = model_checkpoint
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.image_size = image_size
        if use_gpu:
            if not torch.cuda.is_available():
                logger.warning("GPU requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for training")
        self._init_processor()
        self._init_model()
        self._init_datasets()
        self._init_trainer()

    def _init_processor(self):
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            self.model_checkpoint,
            do_reduce_labels=False
        )

    def _init_model(self):
        # UPDATED: Use the stored annotation_file parameter
        with open(os.path.join(self.train_dir, self.annotation_file), "r") as f:
            train_annos = json.load(f)
        id2label = {category['id']: category['name'] for category in train_annos['categories']}
        label2id = {name: id for id, name in id2label.items()}
        logger.info(f"Number of classes in dataset: {len(id2label)}")
        logger.info(f"Class mappings: {id2label}")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_checkpoint,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        logger.info("Model initialized with new class predictor for your dataset")

    def _init_datasets(self):
        # UPDATED: Pass the annotation_file parameter
        self.train_dataset = Mask2FormerCocoDataset(
            self.train_dir,
            self.processor,
            annotation_file=self.annotation_file,
            image_size=self.image_size
        )
        self.val_dataset = Mask2FormerCocoDataset(
            self.val_dir,
            self.processor,
            annotation_file=self.annotation_file,
            image_size=self.image_size
        )

    def _init_trainer(self):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=200,
            logging_steps=50,
            load_best_model_at_end=False,
            remove_unused_columns=False,
        )
        self.metric = evaluate.load("mean_iou")
        
        # We pass our custom collate function to the Trainer here
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self._compute_metrics,
            data_collator=instance_segmentation_collate_fn # <-- ADD THIS LINE
        )

    def _compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        # This part could be improved, but let's keep it for now
        image_sizes = []
        for img_path in self.val_dataset.image_files:
            with Image.open(img_path) as img:
                image_sizes.append((img.height, img.width))
                
        predicted_labels = self.processor.post_process_semantic_segmentation(
            outputs=logits,
            target_sizes=image_sizes
        )
        metrics = self.metric.compute(
            predictions=predicted_labels,
            references=labels,
            num_labels=len(self.train_dataset.id2label),
            ignore_index=255,
            reduce_labels=False,
        )
        if metrics is None:
            return {"mean_iou": 0.0, "mean_accuracy": 0.0}
        return {
            "mean_iou": metrics.get("mean_iou", 0.0),
            "mean_accuracy": metrics.get("mean_accuracy", 0.0),
        }

    def train(self):
        logger.info("Starting Mask2Former fine-tuning...")
        self.trainer.train()
        logger.info("Fine-tuning complete.")
        self.trainer.save_model(os.path.join(self.output_dir, "final"))
        self.processor.save_pretrained(os.path.join(self.output_dir, "final"))

    def evaluate(self):
        logger.info("Evaluating Mask2Former model...")
        metrics = self.trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics


def main():
    # Now you can specify a different annotation filename if needed
    trainer = Mask2FormerFinetuner(
        train_dir="/Users/boyu/mask2former/mask2former_data/train",
        val_dir="/Users/boyu/mask2former/mask2former_data/valid",
        annotation_file="_annotations.coco.json", # <-- Explicitly define your file name here
        output_dir="mask2former-finetuned",
        batch_size=2,
        num_epochs=50,
        learning_rate=5e-5,
        use_gpu=True,
        image_size=(1280, 1280)
    )
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()