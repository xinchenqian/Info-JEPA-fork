import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
import wandb
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset, random_split


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.video_classification_frozen.utils import make_transforms  # noqa: E402


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm", ".mpeg", ".mpg"}
DEFAULT_DATA_DIR = "~/autodl-tmp/data"
DEFAULT_WEIGHT_DIR = "~/autodl-tmp/weight"


def synchronize_if_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen ViT-L + average pooling + two-layer MLP linear probe on a small UCF101 subset."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="UCF101 subset root or csv path.")
    parser.add_argument("--weight-dir", default=DEFAULT_WEIGHT_DIR, help="Directory containing official .pt/.pth weights.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path. Overrides --weight-dir auto discovery.")
    parser.add_argument("--vjepa-version", choices=("2", "2.1"), default="2.1")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--split-id", type=int, default=1, help="Official UCF101 split id to use when splits/ exists.")
    parser.add_argument("--resolution", type=int, default=384, help="Use 384 for V-JEPA2.1 ViT-L and 256 for V-JEPA2 ViT-L.")
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--frame-step", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="~/autodl-tmp/outputs/vjepa2_ucf101_mlp_probe")
    parser.add_argument("--amp", action="store_true", help="Use CUDA autocast during backbone/probe forward.")
    parser.add_argument("--wandb-project", default="vjepa2-ucf101-mlp-probe")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")

    args = parser.parse_args()

    if args.config is not None:
        config_path = Path(args.config).expanduser()
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config is not None:
            for key, value in config.items():
                if not hasattr(args, key):
                    raise ValueError(f"Unknown config key in YAML: {key}")
                setattr(args, key, value)

    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_samples(csv_path):
    samples = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.reader(handle, delimiter=" ")
        for row in reader:
            row = [item for item in row if item != ""]
            if len(row) < 2:
                continue
            samples.append((row[0], int(row[1])))
    return samples


def read_ucf101_class_index(path):
    class_to_idx = {}
    with open(path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            class_to_idx[parts[1]] = int(parts[0]) - 1
    return class_to_idx


def read_ucf101_split(split_path, video_root, class_to_idx):
    samples = []
    with open(split_path, "r") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            rel_path = Path(parts[0])
            label = int(parts[1]) - 1 if len(parts) > 1 else class_to_idx[rel_path.parts[0]]
            samples.append((str(video_root / rel_path), label))
    return samples


def collect_ucf101_official_split(root, split_id):
    root = Path(root).expanduser()
    split_dir = root / "ucfTrainTestlist"
    video_root = root / "UCF-101"
    class_file = split_dir / "classInd.txt"
    train_file = split_dir / f"trainlist{split_id:02d}.txt"
    test_file = split_dir / f"testlist{split_id:02d}.txt"
    required = [video_root, class_file, train_file, test_file]
    if not all(path.exists() for path in required):
        return None

    class_to_idx = read_ucf101_class_index(class_file)
    train_samples = read_ucf101_split(train_file, video_root, class_to_idx)
    val_samples = read_ucf101_split(test_file, video_root, class_to_idx)
    return train_samples, val_samples, class_to_idx


def collect_class_folder_samples(root):
    root = Path(root).expanduser()
    split_dirs = {name: root / name for name in ("train", "val", "test") if (root / name).is_dir()}

    def collect_from(base):
        class_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
        class_to_idx = {p.name: idx for idx, p in enumerate(class_dirs)}
        samples = []
        for class_dir in class_dirs:
            label = class_to_idx[class_dir.name]
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in VIDEO_EXTENSIONS:
                    samples.append((str(path), label))
        return samples, class_to_idx

    if "train" in split_dirs and ("val" in split_dirs or "test" in split_dirs):
        train_samples, class_to_idx = collect_from(split_dirs["train"])
        eval_split = split_dirs.get("val", split_dirs.get("test"))
        eval_samples, _ = collect_from(eval_split)
        return train_samples, eval_samples, class_to_idx

    samples, class_to_idx = collect_from(root)
    return samples, None, class_to_idx


class UCFVideoDataset(Dataset):
    def __init__(self, samples, transform, frames_per_clip=16, frame_step=4, training=True):
        self.samples = samples
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.training = training

    def __len__(self):
        return len(self.samples)

    def _sample_indices(self, video_len):
        clip_len = self.frames_per_clip * self.frame_step
        if video_len <= 0:
            raise RuntimeError("empty video")
        if video_len >= clip_len:
            max_start = video_len - clip_len
            start = random.randint(0, max_start) if self.training else max_start // 2
            indices = start + np.arange(self.frames_per_clip) * self.frame_step
        else:
            indices = np.linspace(0, video_len - 1, self.frames_per_clip)
        return np.clip(indices, 0, video_len - 1).astype(np.int64)

    def __getitem__(self, index):
        path, label = self.samples[index]
        vr = VideoReader(path, num_threads=1, ctx=cpu(0))
        indices = self._sample_indices(len(vr))
        buffer = vr.get_batch(indices).asnumpy()
        views = self.transform(buffer)
        clip = views[0]
        return clip, torch.tensor(label, dtype=torch.long)


class AveragePoolMLPProbe(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens):
        pooled = tokens.mean(dim=1)
        return self.head(pooled)


def find_checkpoint(weight_dir, vjepa_version):
    weight_dir = Path(weight_dir).expanduser()
    candidates = sorted(list(weight_dir.glob("*.pt")) + list(weight_dir.glob("*.pth")))
    if not candidates:
        raise FileNotFoundError(f"No .pt/.pth checkpoint found in {weight_dir}")

    def score(path):
        name = path.name.lower()
        value = 0
        if "vitl" in name or "vit_l" in name or "large" in name:
            value += 10
        if vjepa_version == "2.1" and ("2_1" in name or "2.1" in name or "dist" in name):
            value += 5
        if vjepa_version == "2" and "2_1" not in name and "2.1" not in name:
            value += 3
        return value

    return max(candidates, key=score)


def clean_state_dict(state_dict):
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}


def load_backbone_checkpoint(backbone, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    for key in ("ema_encoder", "target_encoder", "encoder", "backbone", "model", "state_dict"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    checkpoint = clean_state_dict(checkpoint)
    model_state = backbone.state_dict()
    compatible = {}
    skipped = []
    for key, value in checkpoint.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        elif key in model_state:
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))

    msg = backbone.load_state_dict(compatible, strict=False)
    print(f"Loaded backbone weights from: {checkpoint_path}")
    print(f"Compatible tensors: {len(compatible)} | skipped shape mismatches: {len(skipped)}")
    if skipped[:5]:
        print("First skipped tensors:")
        for key, ckpt_shape, model_shape in skipped[:5]:
            print(f"  {key}: checkpoint={ckpt_shape}, model={model_shape}")
    print(f"Missing keys: {len(msg.missing_keys)} | unexpected keys: {len(msg.unexpected_keys)}")


def build_vitl_backbone(args):
    if args.vjepa_version == "2.1":
        from app.vjepa_2_1.models import vision_transformer as vit

        backbone = vit.vit_large(
            img_size=args.resolution,
            num_frames=args.frames_per_clip,
            patch_size=16,
            tubelet_size=2,
            uniform_power=True,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
            n_output_distillation=1,
        )
    else:
        from src.models import vision_transformer as vit

        backbone = vit.vit_large(
            img_size=args.resolution,
            num_frames=args.frames_per_clip,
            patch_size=16,
            tubelet_size=2,
            uniform_power=True,
            use_rope=True,
        )
    return backbone


def make_datasets(args):
    data_path = Path(args.data_dir).expanduser()
    train_transform = make_transforms(training=True, crop_size=args.resolution)
    val_transform = make_transforms(training=False, crop_size=args.resolution)

    official_split = collect_ucf101_official_split(data_path, args.split_id)
    if official_split is not None:
        train_samples, val_samples, class_to_idx = official_split
        train_dataset = UCFVideoDataset(train_samples, train_transform, args.frames_per_clip, args.frame_step, True)
        val_dataset = UCFVideoDataset(val_samples, val_transform, args.frames_per_clip, args.frame_step, False)
        return train_dataset, val_dataset, class_to_idx

    if data_path.is_file() and data_path.suffix == ".csv":
        all_samples = read_csv_samples(data_path)
        if not all_samples:
            raise RuntimeError(f"No samples found in {data_path}")
        generator = torch.Generator().manual_seed(args.seed)
        val_len = max(1, int(len(all_samples) * args.val_ratio))
        train_len = len(all_samples) - val_len
        base = UCFVideoDataset(all_samples, train_transform, args.frames_per_clip, args.frame_step, True)
        train_dataset, val_indices = random_split(base, [train_len, val_len], generator=generator)
        val_samples = [all_samples[i] for i in val_indices.indices]
        val_dataset = UCFVideoDataset(val_samples, val_transform, args.frames_per_clip, args.frame_step, False)
        return train_dataset, val_dataset, None

    train_samples, val_samples, class_to_idx = collect_class_folder_samples(data_path)
    if not train_samples:
        raise RuntimeError(f"No videos found under {data_path}")
    if val_samples is None:
        generator = torch.Generator().manual_seed(args.seed)
        val_len = max(1, int(len(train_samples) * args.val_ratio))
        train_len = len(train_samples) - val_len
        shuffled = list(train_samples)
        perm = torch.randperm(len(shuffled), generator=generator).tolist()
        val_ids = set(perm[:val_len])
        val_samples = [sample for i, sample in enumerate(shuffled) if i in val_ids]
        train_samples = [sample for i, sample in enumerate(shuffled) if i not in val_ids]

    train_dataset = UCFVideoDataset(train_samples, train_transform, args.frames_per_clip, args.frame_step, True)
    val_dataset = UCFVideoDataset(val_samples, val_transform, args.frames_per_clip, args.frame_step, False)
    return train_dataset, val_dataset, class_to_idx


def run_epoch(backbone, probe, loader, criterion, optimizer, device, training, use_amp, global_step=0):
    probe.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    epoch_start_time = time.perf_counter()

    amp_enabled = use_amp and device.type == "cuda"
    grad_context = torch.enable_grad() if training else torch.no_grad()

    for clips, labels in loader:
        synchronize_if_cuda(device)
        step_start_time = time.perf_counter()

        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
                tokens = backbone(clips)

        with grad_context:
            with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
                logits = probe(tokens)
                loss = criterion(logits, labels)

        if training:
            loss.backward()
            optimizer.step()
            synchronize_if_cuda(device)
            step_time = time.perf_counter() - step_start_time
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log(
                {
                    "train/global_step": global_step,
                    "train/step_loss": loss.item(),
                    "train/lr": current_lr,
                    "train/step_time_sec": step_time,
                }
            )
            global_step += 1

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    avg_loss = total_loss / max(1, total_count)
    avg_acc = 100.0 * total_correct / max(1, total_count)
    synchronize_if_cuda(device)
    epoch_time = time.perf_counter() - epoch_start_time
    if training:
        return avg_loss, avg_acc, global_step, epoch_time
    return avg_loss, avg_acc, epoch_time


def save_probe_checkpoint(path, probe, optimizer, scheduler, args, class_to_idx, epoch, best_val, val_acc):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "probe": probe.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "class_to_idx": class_to_idx,
            "best_val": best_val,
            "val_acc": val_acc,
        },
        path,
    )


def main():
    args = parse_args()
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.vjepa_version == "2" and args.resolution == 384:
        print("Tip: official V-JEPA2 ViT-L is usually evaluated at 256px. Use --resolution 256 if your checkpoint is vitl.pt.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else find_checkpoint(args.weight_dir, args.vjepa_version)

    train_dataset, val_dataset, class_to_idx = make_datasets(args)
    if class_to_idx is not None:
        print(f"Classes ({len(class_to_idx)}): {list(class_to_idx.keys())}")
        if args.num_classes is None:
            args.num_classes = len(class_to_idx)
    elif args.num_classes is None:
        labels = [label for _, label in train_dataset.samples]
        args.num_classes = max(labels) + 1
    print(f"Train videos: {len(train_dataset)} | Val videos: {len(val_dataset)}")

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        **loader_kwargs,
    )

    backbone = build_vitl_backbone(args)
    load_backbone_checkpoint(backbone, checkpoint)
    backbone.to(device).eval()
    for param in backbone.parameters():
        param.requires_grad = False

    probe = AveragePoolMLPProbe(backbone.embed_dim, args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        config=vars(args),
    )
    wandb.define_metric("train/global_step")
    wandb.define_metric("epoch")
    wandb.define_metric("train/step_loss", step_metric="train/global_step")
    wandb.define_metric("train/lr", step_metric="train/global_step")
    wandb.define_metric("train/step_time_sec", step_metric="train/global_step")
    wandb.define_metric("train/epoch_loss", step_metric="epoch")
    wandb.define_metric("train/epoch_acc", step_metric="epoch")
    wandb.define_metric("train/epoch_time_sec", step_metric="epoch")
    wandb.define_metric("val/loss", step_metric="epoch")
    wandb.define_metric("val/acc", step_metric="epoch")
    wandb.define_metric("val/epoch_time_sec", step_metric="epoch")
    wandb.define_metric("epoch/time_sec", step_metric="epoch")
    wandb.define_metric("val/best_acc", step_metric="epoch")

    best_val = -math.inf
    global_step = 0
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        train_loss, train_acc, global_step, train_epoch_time = run_epoch(
            backbone,
            probe,
            train_loader,
            criterion,
            optimizer,
            device,
            training=True,
            use_amp=args.amp,
            global_step=global_step,
        )
        val_loss, val_acc, val_epoch_time = run_epoch(
            backbone, probe, val_loader, criterion, optimizer, device, training=False, use_amp=args.amp
        )
        synchronize_if_cuda(device)
        epoch_time = time.perf_counter() - epoch_start_time
        scheduler.step()
        is_best = val_acc > best_val
        best_val = max(best_val, val_acc)
        save_probe_checkpoint(output_dir / "last.pt", probe, optimizer, scheduler, args, class_to_idx, epoch, best_val, val_acc)
        if is_best:
            save_probe_checkpoint(
                output_dir / "best.pt", probe, optimizer, scheduler, args, class_to_idx, epoch, best_val, val_acc
            )
        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "train/epoch_acc": train_acc,
                "train/epoch_time_sec": train_epoch_time,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/epoch_time_sec": val_epoch_time,
                "epoch/time_sec": epoch_time,
                "val/best_acc": best_val,
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f} acc {val_acc:.2f}% | best {best_val:.2f}% | "
            f"time {epoch_time:.2f}s"
        )
    wandb.finish()


if __name__ == "__main__":
    main()
