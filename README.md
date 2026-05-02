#  SAM + Grounding DINO Finetuner

A simple and flexible CLI tool to fine-tune:

-  **Segment Anything Model (SAM)** → segmentation
-  **Grounding DINO** → prompt-based object detection


---
##  Features

- ✅ SAM mask decoder finetuning
- ✅ Grounding DINO text-conditioned detection
- ✅ CLI-based training (easy to use)
- ✅ COCO dataset support
- ✅ Balanced dataset sampling
- ✅ Mixed precision (AMP)
- ✅ Auto checkpoint saving
- ✅ Sample prediction visualization

---

##  Installation

### 🔹 Clone & install locally

```bash
git clone https://github.com/pratim4dasude/finetuning_grounded_dino_sam.git
cd finetuning_grounded_dino_sam
pip install -e .
```

---

##  Dataset Format

Your dataset must follow **COCO format**:

```
dataset_root/
│
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── image2.jpg
│
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    └── image2.jpg
```

---

##  Usage

### 🔹 1. Finetune Grounding DINO

```bash
python -m finetuning.cli dino --dataset_root data\Dataset_oRobot \
  --output_dir data\grounding_dino_test \
  --text_labels person cat wall \
  --image_size 512 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --num_epochs 10 \
  --learning_rate 1e-5 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --max_grad_norm 1.0 \
  --train_sample_limit 500 \
  --test_sample_limit 100 \
  --seed 42
```

### 🔹 2. Finetune SAM

```bash
 python -m finetuning.cli sam --data_root data\cracks_cleaned \
  --output_dir data\sam_test \
  --model_name facebook/sam-vit-base \
  --max_train_samples 1000 \
  --max_test_samples 300 \
  --resize_to 512 \
  --batch_size 2 \
  --num_epochs 10 \
  --lr 1e-5 \
  --weight_decay 1e-4 \
  --num_workers 0 \
  --box_jitter 10 \
  --seed 42
```

---

## Outputs

After training, your `output_dir` will contain:

```
output_dir/
│
├── epoch_1/               # Optional checkpoints
├── best_model/            # Best model weights
├── train_log.txt          # Training logs
└── sample_prediction.jpg  # Visualization
```

---

## Metrics

### Grounding DINO
- Train Loss
- Test Loss
- Detection results

### SAM
- IoU
- Dice Score
- Mask quality

---

## Notes

> ❗ Do **NOT** use very small image sizes (`<256`) for DINO

Use quotes for multi-word labels:
```bash
--text_labels person cat wall toys"
```

---

## Install PyTorch (GPU)

Install manually based on your system:

👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

**Example (CUDA 12.4):**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

##  Development Mode

Run without installing:

```bash
python -m finetuning.cli dino --dataset_root data --output_dir out
```

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| PyTorch | Deep learning backbone |
| Hugging Face Transformers | Model hub & utilities |
| SAM (Meta) | Segmentation model |
| Grounding DINO | Text-conditioned detection |
| COCO API | Dataset handling |

---
