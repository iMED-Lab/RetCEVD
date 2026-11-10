import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import OCTATabularDataset, load_split
from loss import (
    InfoNCEGroup,
    PairwiseCLUB,
    build_cross_class_pairs,
    fit_info_nce,
    fit_pairwise_club,
)
from metrics import classification_metrics
from model import RetCEVD
from utils import set_requires_grad, set_seed, update_class_memories


def parse_args():
    parser = argparse.ArgumentParser(description="Train RetCEVD on anonymized split files.")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone-weights", type=Path, default=None)
    parser.add_argument(
        "--tasks", nargs="+", default=["cevd_det", "cevd_sub", "cevd_ris"]
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--estimator-learning-rate", type=float, default=5e-5)
    parser.add_argument("--lambda-inter", type=float, default=0.1)
    parser.add_argument("--lambda-intra", type=float, default=0.1)
    parser.add_argument("--memory-minimum", type=int, default=64)
    parser.add_argument("--club-topk", type=int, default=16)
    parser.add_argument("--club-temperature", type=float, default=0.1)
    parser.add_argument("--club-max-reuse", type=int, default=1)
    parser.add_argument("--club-estimator-steps", type=int, default=1)
    parser.add_argument("--weighted-sampling", action="store_true")
    parser.add_argument("--gradient-check", action="store_true")
    return parser.parse_args()


def make_loader(dataset, batch_size, workers, weighted, shuffle):
    if weighted:
        counts = np.bincount(np.asarray(dataset.labels, dtype=np.int64), minlength=2)
        class_weights = 1.0 / np.maximum(counts, 1)
        sample_weights = torch.as_tensor(
            [class_weights[label] for label in dataset.labels], dtype=torch.double
        )
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )


def move_batch(batch, device):
    svc, dvc, cc, tabular, labels = batch
    images = torch.cat([svc, dvc, cc], dim=1).to(device, non_blocking=True)
    return images, tabular.to(device, non_blocking=True), labels.to(device, non_blocking=True)


def epoch_metrics(labels, predictions, scores, average_loss, sample_count):
    result = classification_metrics(labels, predictions, scores)
    result["loss"] = average_loss / max(sample_count, 1)
    return result


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    club,
    club_optimizer,
    info_nce,
    info_optimizer,
    positive_memory,
    negative_memory,
    epoch,
    args,
    device,
):
    model.train()
    labels_all, predictions_all, scores_all = [], [], []
    loss_sum, sample_count = 0.0, 0
    checked = False

    for batch in loader:
        images, tabular, labels = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        representation, logits = model(images, tabular)
        ce_loss = criterion(logits, labels)
        positive_features = representation[labels == 1]
        negative_features = representation[labels == 0]
        regularization = ce_loss.new_zeros(())

        ready = (
            epoch > 0
            and len(positive_features) > 0
            and len(negative_features) > 0
            and len(positive_memory) >= args.memory_minimum
            and len(negative_memory) >= args.memory_minimum
        )
        if ready:
            positive_bank = torch.stack(positive_memory).detach()
            negative_bank = torch.stack(negative_memory).detach()
            paired_positive, positive_stats = build_cross_class_pairs(
                positive_features,
                negative_bank,
                topk=args.club_topk,
                temperature=args.club_temperature,
                max_reuse=args.club_max_reuse,
            )
            paired_negative, negative_stats = build_cross_class_pairs(
                negative_features,
                positive_bank,
                topk=args.club_topk,
                temperature=args.club_temperature,
                max_reuse=args.club_max_reuse,
            )
            club_fit = fit_pairwise_club(
                club,
                club_optimizer,
                positive_features,
                negative_features,
                paired_positive,
                paired_negative,
                estimator_steps=args.club_estimator_steps,
            )
            fit_info_nce(info_nce, info_optimizer, positive_features, positive_bank)
            fit_info_nce(info_nce, info_optimizer, negative_features, negative_bank)

            set_requires_grad(club, False)
            set_requires_grad(info_nce, False)
            club_term = club(positive_features, paired_positive) + club(
                negative_features, paired_negative
            )
            info_term = info_nce(positive_features, positive_bank) + info_nce(
                negative_features, negative_bank
            )
            regularization = (
                args.lambda_inter * club_term
                - args.lambda_intra * F.softplus(info_term)
            )

            if args.gradient_check and not checked:
                club_gradient = torch.autograd.grad(
                    club_term, representation, retain_graph=True, allow_unused=True
                )[0]
                total_gradient = torch.autograd.grad(
                    regularization, representation, retain_graph=True, allow_unused=True
                )[0]
                club_norm = 0.0 if club_gradient is None else float(club_gradient.norm())
                total_norm = 0.0 if total_gradient is None else float(total_gradient.norm())
                if club_norm <= 1e-8 or total_norm <= 1e-8:
                    raise RuntimeError(
                        f"IM gradient check failed: club={club_norm}, total={total_norm}"
                    )
                print(
                    f"[IM CHECK] epoch={epoch} club={float(club_term):.6f} "
                    f"club_fit={float(club_fit):.6f} info={float(info_term):.6f} "
                    f"club_grad={club_norm:.8f} total_grad={total_norm:.8f} "
                    f"unique_pairs={positive_stats.unique_ratio:.3f}/"
                    f"{negative_stats.unique_ratio:.3f}",
                    flush=True,
                )
                checked = True

        total_loss = ce_loss + regularization
        total_loss.backward()
        optimizer.step()
        set_requires_grad(club, True)
        set_requires_grad(info_nce, True)

        if len(positive_features) and len(negative_features):
            update_class_memories(
                positive_features,
                negative_features,
                positive_memory,
                negative_memory,
                maximum_size=len(loader.dataset),
            )

        predictions = logits.argmax(dim=1)
        scores = logits.softmax(dim=1)[:, 1]
        batch_size = labels.shape[0]
        loss_sum += float(total_loss.detach()) * batch_size
        sample_count += batch_size
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())
        scores_all.extend(scores.detach().cpu().tolist())

    return epoch_metrics(labels_all, predictions_all, scores_all, loss_sum, sample_count)


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    labels_all, predictions_all, scores_all = [], [], []
    loss_sum, sample_count = 0.0, 0
    for batch in loader:
        images, tabular, labels = move_batch(batch, device)
        _, logits = model(images, tabular)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        scores = logits.softmax(dim=1)[:, 1]
        batch_size = labels.shape[0]
        loss_sum += float(loss) * batch_size
        sample_count += batch_size
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        scores_all.extend(scores.cpu().tolist())
    return epoch_metrics(labels_all, predictions_all, scores_all, loss_sum, sample_count)


def train_task(task, args, device):
    task_output = args.output_dir / task
    task_output.mkdir(parents=True, exist_ok=True)
    fold_results = []
    criterion = nn.CrossEntropyLoss()

    for fold in args.folds:
        train_file = args.split_root / task / f"train_fold{fold}.tsv"
        validation_file = args.split_root / task / f"test_fold{fold}.tsv"
        train_rows = load_split(train_file, args.image_root)
        validation_rows = load_split(validation_file, args.image_root)
        train_dataset = OCTATabularDataset(*train_rows, training=True)
        validation_dataset = OCTATabularDataset(
            *validation_rows,
            training=False,
            vmin=train_dataset.vmin.numpy(),
            value_range=train_dataset.value_range.numpy(),
        )
        train_loader = make_loader(
            train_dataset,
            args.batch_size,
            args.workers,
            args.weighted_sampling,
            shuffle=True,
        )
        validation_loader = make_loader(
            validation_dataset,
            args.batch_size,
            args.workers,
            weighted=False,
            shuffle=False,
        )

        model = RetCEVD(69, args.backbone_weights).to(device)
        club = PairwiseCLUB(model.representation_dim, model.representation_dim).to(device)
        info_nce = InfoNCEGroup(model.representation_dim, model.representation_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        club_optimizer = torch.optim.Adam(
            club.parameters(), lr=args.estimator_learning_rate
        )
        info_optimizer = torch.optim.Adam(
            info_nce.parameters(), lr=args.estimator_learning_rate
        )
        positive_memory, negative_memory = [], []
        best = {"auc": -np.inf, "epoch": -1}
        epochs_without_improvement = 0

        for epoch in range(args.epochs):
            train_result = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                club,
                club_optimizer,
                info_nce,
                info_optimizer,
                positive_memory,
                negative_memory,
                epoch,
                args,
                device,
            )
            validation_result = evaluate(model, validation_loader, criterion, device)
            print(
                f"task={task} fold={fold} epoch={epoch} "
                f"train_auc={train_result['auc']:.6f} "
                f"val_auc={validation_result['auc']:.6f}",
                flush=True,
            )

            if validation_result["auc"] > best["auc"] + 1e-8:
                best = {"epoch": epoch, **validation_result}
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vmin": train_dataset.vmin,
                        "value_range": train_dataset.value_range,
                        "task": task,
                        "fold": fold,
                        "feature_indices": list(range(0, 7)) + list(range(8, 70)),
                        "best": best,
                    },
                    task_output / f"retcevd_fold{fold}_best.pt",
                )
            else:
                epochs_without_improvement += 1
                if args.patience > 0 and epochs_without_improvement >= args.patience:
                    break
        fold_results.append(best)

    summary = {
        "task": task,
        "fold_results": fold_results,
        "mean_auc": float(np.mean([row["auc"] for row in fold_results])),
        "std_auc": float(np.std([row["auc"] for row in fold_results])),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (task_output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        train_task(task, args, device)


if __name__ == "__main__":
    main()
