import os
import random
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from tqdm import tqdm
from transformers import SamProcessor, SamModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = 1.0 - (
            (2.0 * intersection + self.smooth) /
            (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        )
        dice = dice.mean()

        return bce + dice


@torch.no_grad()
def compute_iou_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection

    iou = ((intersection + eps) / (union + eps)).mean().item()
    dice = ((2 * intersection + eps) / (preds.sum(dim=1) + targets.sum(dim=1) + eps)).mean().item()

    return iou, dice


def merge_instance_masks(coco: COCO, anns: List[Dict[str, Any]], height: int, width: int) -> np.ndarray:
    final_mask = np.zeros((height, width), dtype=np.uint8)
    for ann in anns:
        ann_mask = coco.annToMask(ann)
        final_mask = np.maximum(final_mask, ann_mask.astype(np.uint8))
    return final_mask


def get_bounding_box(mask: np.ndarray, jitter: int = 0):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    if jitter > 0:
        h, w = mask.shape
        x_min = max(0, x_min - random.randint(0, jitter))
        y_min = max(0, y_min - random.randint(0, jitter))
        x_max = min(w - 1, x_max + random.randint(0, jitter))
        y_max = min(h - 1, y_max + random.randint(0, jitter))

    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def resize_image_and_mask(image: Image.Image, mask: np.ndarray, size: Optional[int]):

    if size is None:
        return image, mask

    image = image.resize((size, size), Image.BILINEAR)

    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    mask_pil = mask_pil.resize((size, size), Image.NEAREST)
    mask = (np.array(mask_pil) > 127).astype(np.uint8)

    return image, mask


class SAMCocoPolygonDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        ann_json: str,
        processor: SamProcessor,
        max_samples: Optional[int] = None,
        box_jitter: int = 10,
        resize_to: Optional[int] = None,
        shuffle_before_limit: bool = True,
    ):
        self.image_dir = image_dir
        self.coco = COCO(ann_json)
        self.processor = processor
        self.box_jitter = box_jitter
        self.resize_to = resize_to

        img_ids = self.coco.getImgIds()
        self.samples = []

        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anns = self.coco.loadAnns(ann_ids)

            if len(anns) == 0:
                continue

            self.samples.append({
                "img_id": img_id,
                "file_name": img_info["file_name"],
                "height": img_info["height"],
                "width": img_info["width"],
                "anns": anns,
            })

        if shuffle_before_limit:
            random.shuffle(self.samples)

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid annotated samples found in {ann_json}")

        print(f"Loaded {len(self.samples)} samples from: {ann_json}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image_path = os.path.join(self.image_dir, item["file_name"])
        image = Image.open(image_path).convert("RGB")

        mask = merge_instance_masks(
            coco=self.coco,
            anns=item["anns"],
            height=item["height"],
            width=item["width"]
        )

        # make binary
        mask = (mask > 0).astype(np.uint8)

        # resize both image and mask if requested
        image, mask = resize_image_and_mask(image, mask, self.resize_to)

        bbox = get_bounding_box(mask, jitter=self.box_jitter)
        if bbox is None:
            raise ValueError(f"Empty mask found for {item['file_name']}")

        inputs = self.processor(
            image,
            input_boxes=[[bbox]],
            return_tensors="pt"
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        inputs["ground_truth_mask"] = torch.tensor(mask, dtype=torch.float32)
        inputs["image_name"] = item["file_name"]
        inputs["bbox"] = torch.tensor(bbox, dtype=torch.float32)

        return inputs


def collate_fn(batch):
    pixel_values = torch.stack([x["pixel_values"] for x in batch])
    input_boxes = torch.stack([x["input_boxes"] for x in batch])
    ground_truth_mask = torch.stack([x["ground_truth_mask"] for x in batch])
    image_names = [x["image_name"] for x in batch]
    bboxes = torch.stack([x["bbox"] for x in batch])

    return {
        "pixel_values": pixel_values,
        "input_boxes": input_boxes,
        "ground_truth_mask": ground_truth_mask,
        "image_names": image_names,
        "bboxes": bboxes,
    }


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    count = 0

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        gt_masks = batch["ground_truth_mask"].to(device).unsqueeze(1)

        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False
        )

        pred_masks = outputs.pred_masks.squeeze(1)

        gt_masks_resized = F.interpolate(
            gt_masks,
            size=pred_masks.shape[-2:],
            mode="nearest"
        )

        loss = loss_fn(pred_masks, gt_masks_resized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iou, dice = compute_iou_dice(pred_masks.detach(), gt_masks_resized.detach())

        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        count += 1

        pbar.set_postfix(
            loss=f"{total_loss / count:.4f}",
            iou=f"{total_iou / count:.4f}",
            dice=f"{total_dice / count:.4f}",
        )

    return total_loss / max(count, 1), total_iou / max(count, 1), total_dice / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    count = 0

    pbar = tqdm(loader, desc="Eval ", leave=False)

    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        gt_masks = batch["ground_truth_mask"].to(device).unsqueeze(1)

        outputs = model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False
        )

        pred_masks = outputs.pred_masks.squeeze(1)

        gt_masks_resized = F.interpolate(
            gt_masks,
            size=pred_masks.shape[-2:],
            mode="nearest"
        )

        loss = loss_fn(pred_masks, gt_masks_resized)
        iou, dice = compute_iou_dice(pred_masks, gt_masks_resized)

        total_loss += loss.item()
        total_iou += iou
        total_dice += dice
        count += 1

        pbar.set_postfix(
            loss=f"{total_loss / count:.4f}",
            iou=f"{total_iou / count:.4f}",
            dice=f"{total_dice / count:.4f}",
        )

    return total_loss / max(count, 1), total_iou / max(count, 1), total_dice / max(count, 1)


def run_sam_finetuning(
    data_root,
    model_name,
    output_dir,
    max_train_samples=1000,
    max_test_samples=300,
    resize_to=512,
    batch_size=1,
    num_epochs=10,
    lr=1e-5,
    weight_decay=1e-4,
    num_workers=0,
    box_jitter=10,
    seed=42,
):
    """
        Fine-tunes SAM mask decoder on a COCO polygon segmentation dataset.

        Expected dataset structure:

        data_root/
            train/
                _annotations.coco.json
                image1.jpg
                image2.jpg
            test/
                _annotations.coco.json
                image1.jpg
                image2.jpg

        Args:
            data_root: Root dataset folder.
            model_name: Hugging Face SAM model name.
            output_dir: Folder to save best and last checkpoints.
            max_train_samples: Number of training samples to use.
            max_test_samples: Number of test samples to use.
            resize_to: Resize image and mask to this size.
            batch_size: Training batch size.
            num_epochs: Number of training epochs.
            lr: Learning rate.
            weight_decay: AdamW weight decay.
            num_workers: DataLoader workers.
            box_jitter: Random jitter added to bounding box prompt.
            seed: Random seed.
        Returns:
            None (saves checkpoints to output_dir)

        Outputs:
            - last checkpoint (every epoch)
            - best checkpoint (based on Dice score)

        Metrics:
            - IoU
            - Dice Score
        """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")

    train_json = os.path.join(train_dir, "_annotations.coco.json")
    test_json = os.path.join(test_dir, "_annotations.coco.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Model: {model_name}")
    print(f"Resize to: {resize_to}")
    print(f"Max train samples: {max_train_samples}")
    print(f"Max test samples: {max_test_samples}")

    processor = SamProcessor.from_pretrained(model_name)
    model = SamModel.from_pretrained(model_name).to(device)

    # freeze vision encoder + prompt encoder
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    train_dataset = SAMCocoPolygonDataset(
        image_dir=train_dir,
        ann_json=train_json,
        processor=processor,
        max_samples=max_train_samples,
        box_jitter=box_jitter,
        resize_to=resize_to,
        shuffle_before_limit=True,
    )

    test_dataset = SAMCocoPolygonDataset(
        image_dir=test_dir,
        ann_json=test_json,
        processor=processor,
        max_samples=max_test_samples,
        box_jitter=box_jitter,
        resize_to=resize_to,
        shuffle_before_limit=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    optimizer = AdamW(
        model.mask_decoder.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_fn = DiceBCELoss()
    best_test_dice = -1.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_iou, train_dice = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        test_loss, test_iou, test_dice = evaluate(
            model, test_loader, loss_fn, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Dice: {train_dice:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test IoU: {test_iou:.4f}, Test Dice: {test_dice:.4f}"
        )

        last_ckpt = os.path.join(output_dir, "sam_old_last.pth")
        torch.save({
            "epoch": epoch,
            "model_name": model_name,
            "resize_to": resize_to,
            "max_train_samples": max_train_samples,
            "max_test_samples": max_test_samples,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_iou": test_iou,
            "test_dice": test_dice,
        }, last_ckpt)

        if test_dice > best_test_dice:
            best_test_dice = test_dice

            best_ckpt = os.path.join(output_dir, "sam_old_best.pth")
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "resize_to": resize_to,
                "max_train_samples": max_train_samples,
                "max_test_samples": max_test_samples,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_iou": test_iou,
                "test_dice": test_dice,
            }, best_ckpt)

            print(f"Saved best checkpoint to: {best_ckpt}")

    print("Training complete.")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Sam Finetuning Script")

    parser.add_argument("--data_root", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_test_samples", type=int)
    parser.add_argument("--resize_to", type=int)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)

    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--box_jitter", type=int)
    parser.add_argument("--seed", type=int)

    return parser.parse_args()
def main():
    args = parse_args()

    config = {
        "data_root": r"origin_robot\oR_dataset\cracks_cleaned",
        "model_name": "facebook/sam-vit-base",
        "output_dir": r"origin_robot\sam_old_finetune_outputs",
        "max_train_samples": 1000,
        "max_test_samples": 300,
        "resize_to": 512,
        "batch_size": 1,
        "num_epochs": 10,
        "lr": 1e-5,
        "weight_decay": 1e-4,
        "num_workers": 0,
        "box_jitter": 10,
        "seed": 42,
    }

    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    print("Final Config:", config)

    run_sam_finetuning(**config)


if __name__ == "__main__":
    main()