import os
import json
import random
import argparse
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def move_labels_to_device(labels, device):
    moved = []

    for label in labels:
        new_label = {}

        for k, v in label.items():
            if isinstance(v, torch.Tensor):
                new_label[k] = v.to(device)
            else:
                new_label[k] = v

        moved.append(new_label)

    return moved


def draw_boxes(image: Image.Image, boxes, labels, save_path: str):
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, max(0, y1 - 12)), str(label), fill="red")

    image.save(save_path)


def print_class_distribution(title, items, cat_id_to_name):
    counts = Counter()

    for item in items:
        cls_id = item["primary_class_id"]
        counts[cat_id_to_name[cls_id]] += 1

    print(f"\n{title}")
    print("-" * len(title))

    total = sum(counts.values())
    print(f"Total images: {total}")

    for cls_name, count in counts.items():
        print(f"  {cls_name}: {count}")


def balanced_select_by_class(items, sample_limit, seed=42):
    if sample_limit is None or sample_limit >= len(items):
        return items

    rng = random.Random(seed)
    by_class = defaultdict(list)

    for item in items:
        by_class[item["primary_class_id"]].append(item)

    class_ids = sorted(by_class.keys())
    num_classes = len(class_ids)
    per_class = sample_limit // num_classes

    selected = []
    leftovers = []

    for cls_id in class_ids:
        cls_items = by_class[cls_id][:]
        rng.shuffle(cls_items)

        take = min(per_class, len(cls_items))
        selected.extend(cls_items[:take])

        if len(cls_items) > take:
            leftovers.extend(cls_items[take:])

    rng.shuffle(leftovers)

    need_more = sample_limit - len(selected)

    if need_more > 0:
        selected.extend(leftovers[:need_more])

    rng.shuffle(selected)

    return selected


class CocoGroundingDINODataset(Dataset):
    def __init__(
        self,
        image_dir,
        annotation_path,
        processor,
        text_labels,
        sample_limit=None,
        seed=42,
        image_size=512,
    ):
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.processor = processor
        self.text_labels = text_labels
        self.seed = seed
        self.image_size = image_size

        with open(annotation_path, "r", encoding="utf-8") as f:
            self.coco = json.load(f)

        self.images = self.coco["images"]
        self.annotations = self.coco["annotations"]
        self.categories = self.coco["categories"]

        self.cat_id_to_name = {
            cat["id"]: cat["name"]
            for cat in self.categories
        }

        self.anns_by_image = defaultdict(list)

        for ann in self.annotations:
            if "bbox" not in ann:
                continue

            bbox = ann["bbox"]

            if bbox is None or len(bbox) != 4:
                continue

            x, y, w, h = bbox

            if w <= 1 or h <= 1:
                continue

            self.anns_by_image[ann["image_id"]].append(ann)

        raw_items = []

        for img in self.images:
            image_id = img["id"]

            if image_id not in self.anns_by_image:
                continue

            anns = self.anns_by_image[image_id]
            class_counts = Counter([int(a["category_id"]) for a in anns])
            primary_class_id = class_counts.most_common(1)[0][0]

            raw_items.append(
                {
                    "image_info": img,
                    "image_id": image_id,
                    "primary_class_id": primary_class_id,
                }
            )

        print(f"\nLoaded dataset: {annotation_path}")
        print(f"Images with annotations before subset: {len(raw_items)}")
        print(
            f"Total annotations: "
            f"{sum(len(self.anns_by_image[item['image_id']]) for item in raw_items)}"
        )

        print_class_distribution(
            title="Before balanced subset",
            items=raw_items,
            cat_id_to_name=self.cat_id_to_name,
        )

        self.items = balanced_select_by_class(
            items=raw_items,
            sample_limit=sample_limit,
            seed=seed,
        )

        print_class_distribution(
            title=f"After balanced subset (limit={sample_limit})"
            if sample_limit
            else "After subset (full data)",
            items=self.items,
            cat_id_to_name=self.cat_id_to_name,
        )

        print(f"Final images used: {len(self.items)}")
        print(
            f"Final annotations used: "
            f"{sum(len(self.anns_by_image[item['image_id']]) for item in self.items)}"
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item_info = self.items[idx]
        image_info = item_info["image_info"]
        image_id = item_info["image_id"]

        image_path = os.path.join(self.image_dir, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        anns = self.anns_by_image[image_id]

        annotations_for_processor = []

        for ann in anns:
            x, y, w, h = ann["bbox"]

            annotations_for_processor.append(
                {
                    "image_id": image_id,
                    "category_id": int(ann["category_id"]),
                    "bbox": [x, y, w, h],
                    "area": float(ann.get("area", w * h)),
                    "iscrowd": int(ann.get("iscrowd", 0)),
                }
            )

        target = {
            "image_id": image_id,
            "annotations": annotations_for_processor,
        }

        # Image + COCO annotations. This creates "labels".
        encoding = self.processor.image_processor(
            images=image,
            annotations=target,
            return_tensors="pt",
            size={
                "shortest_edge": self.image_size,
                "longest_edge": self.image_size,
            },
        )

        # Text prompt separately.
        text_prompt = ". ".join(self.text_labels) + "."

        text_encoding = self.processor.tokenizer(
            [text_prompt],
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        encoding["input_ids"] = text_encoding["input_ids"]
        encoding["attention_mask"] = text_encoding["attention_mask"]

        out = {}

        for k, v in encoding.items():
            if k == "labels":
                out[k] = v[0]
            elif isinstance(v, torch.Tensor):
                out[k] = v.squeeze(0)
            else:
                out[k] = v

        out["image_path"] = image_path
        out["raw_image"] = image
        out["primary_class_id"] = item_info["primary_class_id"]

        return out


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": [item["labels"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "raw_images": [item["raw_image"] for item in batch],
        "primary_class_ids": [item["primary_class_id"] for item in batch],
    }


@torch.no_grad()
def evaluate_loss(model, dataloader, device, use_amp):
    model.eval()

    total_loss = 0.0
    count = 0

    amp_device = "cuda" if device == "cuda" else "cpu"

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = move_labels_to_device(batch["labels"], device)

        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        total_loss += outputs.loss.item()
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def save_sample_prediction(
    model,
    processor,
    dataset,
    device,
    save_path,
    text_labels,
    image_size,
):
    model.eval()

    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]

    image = sample["raw_image"]
    image_path = sample["image_path"]

    text_prompt = ". ".join(text_labels) + "."

    inputs = processor(
        images=image,
        text=[text_prompt],
        return_tensors="pt",
        size={
            "shortest_edge": image_size,
            "longest_edge": image_size,
        },
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]],
    )

    result = results[0]

    boxes = result["boxes"].cpu().tolist()
    labels = result["labels"]

    draw_boxes(image, boxes, labels, save_path)

    print(f"\nSaved sample prediction from: {image_path}")
    print(f"Saved visual to: {save_path}")


def run_grounding_dino_finetuning(
    checkpoint,
    dataset_root,
    output_dir,
    text_labels,
    image_size=512,
    batch_size=2,
    grad_accum_steps=4,
    num_epochs=10,
    learning_rate=1e-5,
    weight_decay=1e-4,
    num_workers=0,
    max_grad_norm=1.0,
    save_every_epoch=True,
    freeze_backbone=True,
    train_sample_limit=2000,
    test_sample_limit=600,
    seed=42,
):
    """
    Fine-tunes Grounding DINO on a COCO object detection dataset.

    Expected dataset structure:

    dataset_root/
        train/
            _annotations.coco.json
            image1.jpg
        test/
            _annotations.coco.json
            image1.jpg

    Outputs:
        output_dir/train_log.txt
        output_dir/epoch_*/
        output_dir/best_model/
        output_dir/sample_prediction.jpg
    """

    set_seed(seed)
    ensure_dir(output_dir)

    train_dir = os.path.join(dataset_root, "train")
    test_dir = os.path.join(dataset_root, "test")

    train_json = os.path.join(train_dir, "_annotations.coco.json")
    test_json = os.path.join(test_dir, "_annotations.coco.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Text labels: {text_labels}")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False

    model.to(device)

    train_dataset = CocoGroundingDINODataset(
        image_dir=train_dir,
        annotation_path=train_json,
        processor=processor,
        text_labels=text_labels,
        sample_limit=train_sample_limit,
        seed=seed,
        image_size=image_size,
    )

    test_dataset = CocoGroundingDINODataset(
        image_dir=test_dir,
        annotation_path=test_json,
        processor=processor,
        text_labels=text_labels,
        sample_limit=test_sample_limit,
        seed=seed,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device == "cuda"),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    amp_device = "cuda" if device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device=amp_device, enabled=use_amp)

    best_test_loss = float("inf")
    log_path = os.path.join(output_dir, "train_log.txt")

    for epoch in range(num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'=' * 60}")

        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = move_labels_to_device(batch["labels"], device)

            with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)

            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        test_loss = evaluate_loss(model, test_loader, device, use_amp)

        log_line = (
            f"Epoch {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f}"
        )

        print(log_line)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")

        if save_every_epoch:
            epoch_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            ensure_dir(epoch_dir)

            model.save_pretrained(epoch_dir)
            processor.save_pretrained(epoch_dir)

            print(f"Saved epoch checkpoint to: {epoch_dir}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss

            best_dir = os.path.join(output_dir, "best_model")
            ensure_dir(best_dir)

            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)

            print(f"Saved best model to: {best_dir}")

        if device == "cuda":
            torch.cuda.empty_cache()

    sample_pred_path = os.path.join(output_dir, "sample_prediction.jpg")

    save_sample_prediction(
        model=model,
        processor=processor,
        dataset=test_dataset,
        device=device,
        save_path=sample_pred_path,
        text_labels=text_labels,
        image_size=image_size,
    )

    print("\nTraining complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Grounding DINO Fine-tuning Script")

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--text_labels", nargs="+", type=str)

    parser.add_argument("--image_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--max_grad_norm", type=float)

    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--no_save_every_epoch", action="store_true")

    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--no_freeze_backbone", action="store_true")

    parser.add_argument("--train_sample_limit", type=int)
    parser.add_argument("--test_sample_limit", type=int)
    parser.add_argument("--seed", type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "checkpoint": "IDEA-Research/grounding-dino-tiny",
        "dataset_root": r"data\Dataset_oRobot",
        "output_dir": r"data\grounding_dino_finetuned_local",

        "text_labels": ["crack", "drywall join"],

        "image_size": 256,
        "batch_size": 1,
        "grad_accum_steps": 4,
        "num_epochs": 1,
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "num_workers": 0,
        "max_grad_norm": 1.0,

        "save_every_epoch": False,
        "freeze_backbone": False,

        "train_sample_limit": 500,
        "test_sample_limit": 100,
        "seed": 42,
    }

    for key, value in vars(args).items():
        if key.startswith("no_"):
            continue

        if value is not None:
            config[key] = value

    if args.no_save_every_epoch:
        config["save_every_epoch"] = False

    if args.no_freeze_backbone:
        config["freeze_backbone"] = False

    print("\nFinal Config")
    print("-" * 40)

    for key, value in config.items():
        print(f"{key}: {value}")

    print("-" * 40)

    run_grounding_dino_finetuning(
        checkpoint=config["checkpoint"],
        dataset_root=config["dataset_root"],
        output_dir=config["output_dir"],
        text_labels=config["text_labels"],
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        grad_accum_steps=config["grad_accum_steps"],
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        num_workers=config["num_workers"],
        max_grad_norm=config["max_grad_norm"],
        save_every_epoch=config["save_every_epoch"],
        freeze_backbone=config["freeze_backbone"],
        train_sample_limit=config["train_sample_limit"],
        test_sample_limit=config["test_sample_limit"],
        seed=config["seed"],
    )


if __name__ == "__main__":
    main()