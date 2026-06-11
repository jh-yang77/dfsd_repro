from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from dfsd_repro.models.resnet import resnet50


TRANSFORM_IMAGENET = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ImageNet ResNet-50 train features used by DFSD.")
    parser.add_argument("--imagenet-train", default="id_datasets/imagenet/train", help="ImageNet train directory.")
    parser.add_argument("--output-dir", default="imagenet_feature", help="Directory for .npy feature files.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for feature extraction.")
    parser.add_argument("--gpu", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(int(args.gpu))

    dataset = torchvision.datasets.ImageFolder(args.imagenet_train, TRANSFORM_IMAGENET)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    model = resnet50(num_classes=1000, pretrained=True, p=None).eval().cuda()

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    feature_sum = None
    feature_sq_sum = None
    seen = 0
    part_id = 1

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            images = images.cuda(non_blocking=True)
            feature_map = model.features(images)
            pooled = F.adaptive_avg_pool2d(feature_map, 1).view(images.size(0), -1)
            features = pooled.cpu().numpy()
            labels_np = labels.numpy()

            if args.max_samples is not None:
                remaining = args.max_samples - seen
                if remaining <= 0:
                    break
                features = features[:remaining]
                labels_np = labels_np[:remaining]

            all_features.append(features)
            all_labels.append(labels_np)
            batch_sum = features.sum(axis=0)
            batch_sq_sum = np.square(features).sum(axis=0)
            feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
            feature_sq_sum = batch_sq_sum if feature_sq_sum is None else feature_sq_sum + batch_sq_sum
            seen += features.shape[0]

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    mean = feature_sum / seen
    var = feature_sq_sum / seen - np.square(mean)
    np.save(output_dir / "imagenet_train_resnet50_features.npy", features)
    np.save(output_dir / "imagenet_train_resnet50_labels.npy", labels)
    np.save(output_dir / "imagenet_train_resnet50_feature_stats.npy", np.stack([mean, var], axis=0))
    print(f"saved {seen} ImageNet features to {output_dir}")


if __name__ == "__main__":
    main()
