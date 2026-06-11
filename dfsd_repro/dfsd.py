from __future__ import annotations

import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.decomposition import KernelPCA, PCA

from dfsd_repro.config import ExperimentConfig, config_as_legacy_args, package_path

if "seaborn" not in sys.modules:
    sys.modules["seaborn"] = types.ModuleType("seaborn")

from dfsd_repro.utils.ood_utils import get_dataloader, get_model, get_outdataset


MODE_ALIASES = {
    "dfsd_main": "dfsd_main",
    "dfsd": "dfsd_main",
    "dfsd_kpca": "dfsd_main",
    "dfsd_main_pca": "dfsd_main_pca",
    "dfsd_pca": "dfsd_main_pca",
    "dfsd_main_kpca_linear": "dfsd_main_kpca_linear",
    "dfsd_kpca_linear": "dfsd_main_kpca_linear",
    "dfsd_main_kpca_poly": "dfsd_main_kpca_poly",
    "dfsd_kpca_poly": "dfsd_main_kpca_poly",
    "cpp_dice": "cpp_dice",
    "cpp": "cpp_dice",
    "c_pp": "cpp_dice",
    "csp_kpca": "csp_kpca",
    "c_sp_kpca": "csp_kpca",
    "csp_pca": "csp_pca",
    "c_sp_pca": "csp_pca",
    "knn": "knn",
    "nnguide": "nnguide",
}
VALID_MODES = set(MODE_ALIASES)


def normalize_mode(mode: str) -> str:
    try:
        return MODE_ALIASES[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported mode {mode}. Valid modes: {sorted(VALID_MODES)}") from exc


def set_reproducible_state(cfg: ExperimentConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True


def feature_dim_for(cfg: ExperimentConfig) -> int:
    if cfg.in_dataset in {"CIFAR-10", "CIFAR-100"} and cfg.model_arch == "densenet":
        return 342
    if cfg.in_dataset in {"CIFAR-10", "CIFAR-100"} and cfg.model_arch == "resnet18":
        return 512
    if cfg.in_dataset == "imagenet" and cfg.model_arch in {"resnet50", "efficientnet"}:
        return 2048
    if cfg.model_arch == "swin":
        return 1024
    raise ValueError(f"Unsupported feature dimension for {cfg.in_dataset}/{cfg.model_arch}")


def train_feature_paths(cfg: ExperimentConfig) -> tuple[Path, Path, Path]:
    if cfg.in_dataset in {"CIFAR-10", "CIFAR-100"} and cfg.model_arch == "densenet":
        key = cfg.in_dataset.lower().replace("-", "")
        return (
            package_path("features", f"{key}_train_{cfg.model_arch}_features.npy"),
            package_path("features", f"{key}_train_{cfg.model_arch}_labels.npy"),
            package_path("features", f"{key}_train_{cfg.model_arch}_feature_mean.npy"),
        )
    if cfg.in_dataset == "imagenet" and cfg.model_arch == "resnet50":
        return (
            package_path("imagenet_feature", "imagenet_train_resnet50_features.npy"),
            package_path("imagenet_feature", "imagenet_train_resnet50_labels.npy"),
            package_path("imagenet_feature", "imagenet_train_resnet50_feature_mean.npy"),
        )
    raise ValueError(f"Unsupported train feature source for {cfg.in_dataset}/{cfg.model_arch}")


def _train_dataset(cfg: ExperimentConfig):
    if cfg.in_dataset == "CIFAR-10":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        return torchvision.datasets.CIFAR10(root=package_path("id_datasets", "CIFAR-10"), train=True, download=False, transform=transform)
    if cfg.in_dataset == "CIFAR-100":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ]
        )
        return torchvision.datasets.CIFAR100(root=package_path("id_datasets", "CIFAR-100"), train=True, download=False, transform=transform)
    if cfg.in_dataset == "imagenet":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return torchvision.datasets.ImageFolder(package_path("id_datasets", "imagenet", "train"), transform=transform)
    raise ValueError(f"Unsupported train dataset for {cfg.in_dataset}/{cfg.model_arch}")


def _sample_imagefolder_by_class(dataset, class_ids: set[int], samples_per_class: int, seed: int):
    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in sorted(class_ids):
        class_indices = np.flatnonzero(targets == class_id)
        if len(class_indices) == 0:
            continue
        if len(class_indices) > samples_per_class:
            class_indices = rng.choice(class_indices, size=samples_per_class, replace=False)
        selected.extend(class_indices.tolist())
    rng.shuffle(selected)
    return torch.utils.data.Subset(dataset, selected)


def extract_train_features(cfg: ExperimentConfig, model, num_classes: int):
    dataset = _train_dataset(cfg)
    if cfg.in_dataset == "imagenet":
        available_classes = np.asarray(sorted({target for _, target in dataset.samples}))
        selected_idx = class_order(len(available_classes), cfg.fit_class_fraction, cfg.seed)
        selected_classes = set(available_classes[selected_idx].tolist())
        dataset = _sample_imagefolder_by_class(dataset, selected_classes, cfg.imagenet_train_samples_per_class, cfg.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    features_all = []
    labels_all = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.cuda()
            features, _ = extract_features_and_logits(model, images, cfg)
            features_all.append(features)
            labels_all.append(labels.numpy())
    features = np.concatenate(features_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    mean = features.mean(axis=0)
    feat_path, label_path, mean_path = train_feature_paths(cfg)
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, features)
    np.save(label_path, labels)
    np.save(mean_path, mean)
    return features, labels, mean


def load_or_extract_train_features(cfg: ExperimentConfig, model, num_classes: int):
    feat_path, label_path, mean_path = train_feature_paths(cfg)
    if feat_path.exists() and label_path.exists() and mean_path.exists() and not cfg.force_refit:
        return np.load(feat_path, allow_pickle=True), np.load(label_path, allow_pickle=True), np.load(mean_path, allow_pickle=True)
    return extract_train_features(cfg, model, num_classes)


def classifier_params(model, cfg: ExperimentConfig):
    head = model.head if cfg.model_arch in {"swin", "efficientnet"} else model.fc
    bias_vector = None
    weight_matrix = None
    for param in head.parameters():
        arr = param.cpu().detach().numpy()
        if param.data.ndimension() == 1:
            bias_vector = arr
        elif param.data.ndimension() == 2:
            weight_matrix = arr
    if bias_vector is None or weight_matrix is None:
        raise RuntimeError("Could not read classifier weight/bias.")
    return weight_matrix, bias_vector


def class_order(num_classes: int, fraction: float, seed: int):
    rng = np.random.RandomState(seed)
    count = max(1, int(num_classes * fraction))
    return rng.choice(num_classes, size=count, replace=False)


def use_route_dice(cfg: ExperimentConfig) -> bool:
    return cfg.c_pp == "dice"


def _subspace_file(cache_dir: Path, cfg: ExperimentConfig, subspace: str, classnum: int) -> Path:
    return cache_dir / f"{subspace}_{cfg.in_dataset}_{cfg.model_arch}_dim{cfg.dim}_class{classnum}.pkl"


def fit_or_load_subspaces(cfg: ExperimentConfig, model, subspace: str, num_classes: int):
    if subspace not in {"kpca", "kpca_linear", "kpca_poly", "pca"}:
        raise ValueError(f"Unsupported subspace: {subspace}")

    cache_name = f"{subspace}_dim{cfg.dim}_seed{cfg.seed}"
    if subspace == "pca" and cfg.pca_components is not None:
        cache_name = f"{subspace}_dim{cfg.dim}_components{cfg.pca_components}_seed{cfg.seed}"
    cache_dir = Path(cfg.subspace_cache_dir) / cfg.in_dataset / cache_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    feature_id_train, target_id_train, feature_mean = load_or_extract_train_features(cfg, model, num_classes)
    weight_matrix, bias_vector = classifier_params(model, cfg)
    u = -np.matmul(pinv(weight_matrix), bias_vector)
    featuredim = feature_dim_for(cfg)
    n_components = featuredim - cfg.dim
    available_classes = np.asarray(sorted(np.unique(target_id_train)))
    selected_idx = class_order(len(available_classes), cfg.fit_class_fraction, cfg.seed)
    fit_classes = available_classes[selected_idx]
    score_count = max(1, int(num_classes * cfg.score_class_fraction))
    fit_classes = fit_classes[:score_count]

    estimators = []
    alpha = []
    logit_id_train = feature_id_train @ weight_matrix.T + bias_vector
    logit_scale = logit_id_train.max(axis=-1).mean()

    for classnum in fit_classes:
        save_path = _subspace_file(cache_dir, cfg, subspace, int(classnum))
        if save_path.exists() and not cfg.force_refit:
            with open(save_path, "rb") as f:
                estimator, alpha_c = pickle.load(f)
            estimators.append(estimator)
            alpha.append(alpha_c)
            continue

        feature_class = feature_id_train[target_id_train == classnum]
        x = feature_class - u
        if subspace.startswith("kpca"):
            n_components_eff = min(n_components, max(1, x.shape[0] - 1), x.shape[1])
            kernel = {"kpca": "rbf", "kpca_linear": "linear", "kpca_poly": "poly"}[subspace]
            estimator = KernelPCA(
                kernel=kernel,
                n_components=n_components_eff,
                fit_inverse_transform=True,
                remove_zero_eig=False,
                copy_X=False,
            )
        else:
            n_components_eff = min(n_components, max(1, x.shape[0] - 1), x.shape[1])
            if cfg.pca_components is not None:
                n_components_eff = min(n_components_eff, cfg.pca_components)
            estimator = PCA(n_components=n_components_eff)

        transformed = estimator.fit_transform(x)
        reconstructed = estimator.inverse_transform(transformed)
        residual = x - reconstructed
        residual_norm = norm(residual, axis=-1)
        alpha_c = logit_scale / residual_norm.mean()

        with open(save_path, "wb") as f:
            pickle.dump((estimator, alpha_c), f)
        estimators.append(estimator)
        alpha.append(alpha_c)
        print(f"fit {subspace}: class={int(classnum)}")

    return np.asarray(alpha), estimators, u


def _ash(features: torch.Tensor, percentile: int) -> torch.Tensor:
    x = features.view(features.size(0), -1, 1, 1)
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=[1, 2, 3])
    scale = s1 / s2
    return (x * torch.exp(scale[:, None, None, None])).view(features.size(0), -1)


def _scale(features: torch.Tensor, percentile: int) -> torch.Tensor:
    x = features.view(features.size(0), -1, 1, 1)
    original = x.clone()
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=[1, 2, 3])
    scale = s1 / s2
    return (original * torch.exp(scale[:, None, None, None])).view(features.size(0), -1)


def c_pp_features(features: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    if cfg.c_pp == "ash":
        return _ash(features, cfg.ash_percentile)
    if cfg.c_pp == "scale":
        return _scale(features, cfg.scale_percentile)
    return features


def extract_features_and_logits(model, inputs, cfg: ExperimentConfig):
    with torch.no_grad():
        if cfg.model_arch not in {"resnet18", "efficientnet", "swin"}:
            features = model.features(inputs)
            out = F.adaptive_avg_pool2d(features, 1)
            features = out.view(out.size(0), -1)
            outputs = model.fc(c_pp_features(features, cfg))
        elif cfg.model_arch == "resnet18":
            feature1 = F.relu(model.bn1(model.conv1(inputs)))
            feature2 = model.layer1(feature1)
            feature3 = model.layer2(feature2)
            feature4 = model.layer3(feature3)
            feature5 = model.layer4(feature4)
            feature5 = model.avgpool(feature5)
            features = feature5.view(feature5.size(0), -1)
            outputs = model.fc(c_pp_features(features, cfg))
        else:
            features = model.extract_feat(inputs)
            outputs = model.head(c_pp_features(features, cfg))

    return features.cpu().detach().numpy(), outputs.cpu().detach().numpy()


def score_numpy(features, outputs, alpha, estimators, u, mode: str):
    mode = normalize_mode(mode)
    c_pp = logsumexp(outputs, axis=-1)
    if mode == "cpp_dice":
        return c_pp

    vlogit = np.zeros((features.shape[0], len(estimators)))
    for i, estimator in enumerate(estimators):
        reconstructed = estimator.inverse_transform(estimator.transform(features - u))
        vlogit[:, i] = norm((features - u) - reconstructed, axis=-1) * alpha[i]
    c_sp = np.max(-vlogit, axis=-1)

    if mode in {"csp_kpca", "csp_pca"}:
        return c_sp
    return c_pp + c_sp


def _l2_normalize_np(features: np.ndarray) -> np.ndarray:
    denom = norm(features, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return features / denom


def prepare_distance_baseline_state(cfg: ExperimentConfig, model, num_classes: int):
    feature_id_train, _, _ = load_or_extract_train_features(cfg, model, num_classes)
    weight_matrix, bias_vector = classifier_params(model, cfg)
    train_logits = feature_id_train @ weight_matrix.T + bias_vector
    train_energy = logsumexp(train_logits, axis=-1)
    knn_features = torch.from_numpy(_l2_normalize_np(feature_id_train).astype(np.float32)).cuda()
    nnguide_features = torch.from_numpy((feature_id_train * train_energy[:, None]).astype(np.float32)).cuda()
    train_energy = torch.from_numpy(train_energy.astype(np.float32)).cuda()
    return {"knn_features": knn_features, "nnguide_features": nnguide_features, "energy": train_energy}


def score_distance_baseline(features, outputs, state, mode: str, cfg: ExperimentConfig):
    if mode == "knn":
        query = torch.from_numpy(_l2_normalize_np(features).astype(np.float32)).cuda()
        similarities = torch.mm(query, state["knn_features"].T)
        k = min(cfg.knn_k, similarities.shape[1])
        values = torch.topk(similarities, k, dim=1).values
        # KNN-OOD uses IndexFlatL2 on L2-normalized features and scores by -D[:, -1].
        return (2.0 * values[:, -1] - 2.0).cpu().numpy()
    if mode == "nnguide":
        query = torch.from_numpy(features.astype(np.float32)).cuda()
        guided = torch.mm(query, state["nnguide_features"].T)
        k = min(cfg.nnguide_k, guided.shape[1])
        values = torch.topk(guided, k, dim=1).values
        guidance = values.mean(dim=1).cpu().numpy()
        energy = logsumexp(outputs, axis=-1)
        return energy * guidance
    raise ValueError(f"Unsupported distance baseline mode: {mode}")


def write_scores(loader, model, cfg: ExperimentConfig, mode: str, alpha, estimators, u, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for batch_idx, data in enumerate(loader):
            images = data[0].cuda()
            features, outputs = extract_features_and_logits(model, images, cfg)
            scores = score_numpy(features, outputs, alpha, estimators, u, mode)
            for score in scores:
                f.write(f"{score}\n")
            if batch_idx % 10 == 0:
                print(f"{mode}: {batch_idx + 1}/{len(loader)}")


def write_distance_scores(loader, model, cfg: ExperimentConfig, mode: str, state, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for batch_idx, data in enumerate(loader):
            images = data[0].cuda()
            features, outputs = extract_features_and_logits(model, images, cfg)
            scores = score_distance_baseline(features, outputs, state, mode, cfg)
            for score in scores:
                f.write(f"{score}\n")
            if batch_idx % 10 == 0:
                print(f"{mode}: {batch_idx + 1}/{len(loader)}")


def run_mode(cfg: ExperimentConfig, mode: str):
    mode = normalize_mode(mode)

    args = config_as_legacy_args(cfg)
    loader_in, num_classes = get_dataloader(cfg.in_dataset, cfg.batch_size, args)
    backbone_args = config_as_legacy_args(cfg)
    backbone_args.p = None
    backbone_model = get_model(backbone_args, num_classes).eval().cuda()
    feat_path, label_path, mean_path = train_feature_paths(cfg)
    if not (feat_path.exists() and label_path.exists() and mean_path.exists()) or cfg.force_refit:
        extract_train_features(cfg, backbone_model, num_classes)

    if mode in {"knn", "nnguide"}:
        model = backbone_model
        state = prepare_distance_baseline_state(cfg, model, num_classes)
        score_dir = cfg.score_dir(mode)
        write_distance_scores(loader_in, model, cfg, mode, state, score_dir / "in_scores.txt")
        for out_dataset in cfg.out_datasets:
            loader_out = get_outdataset(out_dataset, cfg.in_dataset, cfg.batch_size)
            write_distance_scores(loader_out, model, cfg, mode, state, score_dir / out_dataset / "out_scores.txt")
        return

    if not use_route_dice(cfg):
        args.p = None
    model = get_model(args, num_classes).eval().cuda()

    if mode == "cpp_dice":
        alpha, estimators, u = None, None, None
    else:
        if mode in {"dfsd_main_pca", "csp_pca"}:
            subspace = "pca"
        elif mode == "dfsd_main_kpca_linear":
            subspace = "kpca_linear"
        elif mode == "dfsd_main_kpca_poly":
            subspace = "kpca_poly"
        else:
            subspace = "kpca"
        alpha, estimators, u = fit_or_load_subspaces(cfg, model, subspace, num_classes)

    score_dir = cfg.score_dir(mode)
    write_scores(loader_in, model, cfg, mode, alpha, estimators, u, score_dir / "in_scores.txt")

    for out_dataset in cfg.out_datasets:
        loader_out = get_outdataset(out_dataset, cfg.in_dataset, cfg.batch_size)
        write_scores(
            loader_out,
            model,
            cfg,
            mode,
            alpha,
            estimators,
            u,
            score_dir / out_dataset / "out_scores.txt",
        )
