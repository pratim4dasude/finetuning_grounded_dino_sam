import argparse

from .sam import run_sam_finetuning
from .grounding_dino import run_grounding_dino_finetuning


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tuning CLI for SAM and Grounding DINO"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # =========================
    # SAM CLI
    # =========================
    sam_parser = subparsers.add_parser(
        "sam",
        help="Fine-tune SAM on COCO polygon segmentation dataset",
    )

    sam_parser.add_argument("--data_root", type=str, required=True)
    sam_parser.add_argument("--model_name", type=str, default="facebook/sam-vit-base")
    sam_parser.add_argument("--output_dir", type=str, required=True)

    sam_parser.add_argument("--max_train_samples", type=int, default=1000)
    sam_parser.add_argument("--max_test_samples", type=int, default=300)
    sam_parser.add_argument("--resize_to", type=int, default=512)

    sam_parser.add_argument("--batch_size", type=int, default=1)
    sam_parser.add_argument("--num_epochs", type=int, default=10)
    sam_parser.add_argument("--lr", type=float, default=1e-5)
    sam_parser.add_argument("--weight_decay", type=float, default=1e-4)

    sam_parser.add_argument("--num_workers", type=int, default=0)
    sam_parser.add_argument("--box_jitter", type=int, default=10)
    sam_parser.add_argument("--seed", type=int, default=42)

    # =========================
    # Grounding DINO CLI
    # =========================
    dino_parser = subparsers.add_parser(
        "dino",
        help="Fine-tune Grounding DINO on COCO detection dataset",
    )

    dino_parser.add_argument("--checkpoint", type=str, default="IDEA-Research/grounding-dino-tiny")
    dino_parser.add_argument("--dataset_root", type=str, required=True)
    dino_parser.add_argument("--output_dir", type=str, required=True)

    dino_parser.add_argument(
        "--text_labels",
        nargs="+",
        type=str,
        default=["crack", "drywall join"],
    )

    dino_parser.add_argument("--image_size", type=int, default=512)
    dino_parser.add_argument("--batch_size", type=int, default=2)
    dino_parser.add_argument("--grad_accum_steps", type=int, default=4)
    dino_parser.add_argument("--num_epochs", type=int, default=10)
    dino_parser.add_argument("--learning_rate", type=float, default=1e-5)
    dino_parser.add_argument("--weight_decay", type=float, default=1e-4)
    dino_parser.add_argument("--num_workers", type=int, default=0)
    dino_parser.add_argument("--max_grad_norm", type=float, default=1.0)

    dino_parser.add_argument("--save_every_epoch", action="store_true")
    dino_parser.add_argument("--freeze_backbone", action="store_true")

    dino_parser.add_argument("--train_sample_limit", type=int, default=2000)
    dino_parser.add_argument("--test_sample_limit", type=int, default=600)
    dino_parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "sam":
        run_sam_finetuning(
            data_root=args.data_root,
            model_name=args.model_name,
            output_dir=args.output_dir,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            resize_to=args.resize_to,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            box_jitter=args.box_jitter,
            seed=args.seed,
        )

    elif args.command == "dino":
        run_grounding_dino_finetuning(
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            text_labels=args.text_labels,
            image_size=args.image_size,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            max_grad_norm=args.max_grad_norm,
            save_every_epoch=args.save_every_epoch,
            freeze_backbone=args.freeze_backbone,
            train_sample_limit=args.train_sample_limit,
            test_sample_limit=args.test_sample_limit,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()