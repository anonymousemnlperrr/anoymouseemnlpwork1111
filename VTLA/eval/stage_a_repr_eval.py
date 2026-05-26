"""
VTLA/eval/stage_a_repr_eval.py

Stage A held-out representation evaluation.

Outputs:
  - align_loss
  - align_score
  - phys_accuracy
  - cosine_gap
  - optional PCA / t-SNE visualization

Default behavior:
  - evaluate held-out test roots from VTLA/dataset/splits.py
  - use L2 instructions for stable semantic probing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from VTLA.dataset.splits import get_named_roots, get_selected_test_roots
from VTLA.dataset.vtla_dataset import build_multi_task_dataset
from VTLA.models.phys_classifier import get_phys_labels


def collate_fn(batch: list[dict]) -> dict:
    collated = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([b[key] for b in batch])
        elif isinstance(batch[0][key], str):
            collated[key] = [b[key] for b in batch]
        else:
            collated[key] = [b[key] for b in batch]
    return collated


def _scale_phys_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"phys_temperature must be > 0, got {temperature}")
    return logits / temperature


def _average_phys_logits_by_episode(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_names: list[str],
    episode_indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    sums: dict[tuple[str, int], torch.Tensor] = {}
    counts: dict[tuple[str, int], int] = {}
    label_map: dict[tuple[str, int], torch.Tensor] = {}
    ordered_keys: list[tuple[str, int]] = []

    for idx, (task_name, episode_index) in enumerate(zip(task_names, episode_indices)):
        key = (task_name, int(episode_index))
        if key not in sums:
            sums[key] = logits[idx].clone()
            counts[key] = 1
            label_map[key] = labels[idx].clone()
            ordered_keys.append(key)
        else:
            sums[key] += logits[idx]
            counts[key] += 1

    mean_logits = torch.stack([sums[key] / counts[key] for key in ordered_keys], dim=0)
    mean_labels = torch.stack([label_map[key] for key in ordered_keys], dim=0)
    mean_task_names = [key[0] for key in ordered_keys]
    mean_episode_indices = [key[1] for key in ordered_keys]
    return mean_logits, mean_labels, mean_task_names, mean_episode_indices


def _compute_phys_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_names: list[str],
    temperature: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    scaled_logits = _scale_phys_logits(logits, temperature)
    probs = torch.sigmoid(scaled_logits)
    preds = (probs >= 0.5).float()

    correct_matrix = (preds == labels).float()
    exact_match = (preds == labels).all(dim=1).float()
    overall = {
        "phys_accuracy": float(correct_matrix.mean().item()) if correct_matrix.numel() > 0 else 0.0,
        "phys_exact_match": float(exact_match.mean().item()) if exact_match.numel() > 0 else 0.0,
        "phys_num_units": int(labels.size(0)),
    }

    per_task: dict[str, dict[str, float]] = {}
    for task_name in sorted(set(task_names)):
        indices = [idx for idx, value in enumerate(task_names) if value == task_name]
        task_correct = correct_matrix[indices]
        task_exact = exact_match[indices]
        per_task[task_name] = {
            "phys_accuracy": float(task_correct.mean().item()) if task_correct.numel() > 0 else 0.0,
            "phys_exact_match": float(task_exact.mean().item()) if task_exact.numel() > 0 else 0.0,
            "phys_num_units": len(indices),
        }
    return overall, per_task


def _compute_embedding_projection(embeddings: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        return u[:, :2] * s[:2]
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise ImportError("t-SNE requires scikit-learn to be installed") from exc
        perplexity = min(30, max(5, embeddings.shape[0] // 10))
        tsne = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity)
        return tsne.fit_transform(embeddings)
    raise ValueError(f"Unknown projection method: {method}")


def save_embedding_visualization(
    embeddings: np.ndarray,
    task_names: list[str],
    output_path: str,
    method: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Embedding visualization requires matplotlib") from exc

    projected = _compute_embedding_projection(embeddings, method)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    labels = []
    colors = []
    for task_name in task_names:
        if "blueberry" in task_name:
            labels.append("fragile+compliant")
            colors.append("#c44e52")
        elif "egg" in task_name:
            labels.append("fragile")
            colors.append("#dd8452")
        elif "Sponge" in task_name or "sponge" in task_name:
            labels.append("compliant")
            colors.append("#55a868")
        elif "inboxpicking" in task_name:
            labels.append("compliant")
            colors.append("#4c72b0")
        else:
            labels.append("other")
            colors.append("#8172b2")

    plt.figure(figsize=(7, 6))
    unique_labels = sorted(set(labels))
    for label in unique_labels:
        indices = [i for i, value in enumerate(labels) if value == label]
        plt.scatter(
            projected[indices, 0],
            projected[indices, 1],
            s=14,
            alpha=0.8,
            label=label,
        )
    plt.title(f"Stage A representation {method.upper()} projection")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def evaluate_stage_a(
    checkpoint_path: str,
    config_path: str,
    data_root: str,
    device: str = "cuda",
    batch_size: int = 16,
    instruction_level: str = "L2",
    max_pad_length: Optional[int] = None,
    output_path: Optional[str] = None,
    save_embeddings_path: Optional[str] = None,
    vis_method: str = "none",
    vis_output: Optional[str] = None,
    task_names: Optional[list[str]] = None,
    phys_temperature: float = 1.25,
    phys_aggregation: str = "episode",
) -> dict:
    from VTLA.models.vtla_model import VTLAModel

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    eval_device = torch.device(device)
    model = VTLAModel(
        vlm_model_id=cfg.get("vlm_model_id", "Qwen/Qwen2-VL-2B-Instruct"),
        action_dim=cfg.get("action_dim", 6),
        n_queries=cfg.get("n_queries", 16),
        d_bottleneck=cfg.get("d_bottleneck", 512),
        n_bottleneck_layers=cfg.get("n_bottleneck_layers", 2),
        freeze_vlm=True,
        freeze_encoders=False,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(eval_device)
    model.eval()

    tokenizer = model.processor.tokenizer
    task_roots = get_named_roots(task_names, data_root) if task_names else get_selected_test_roots(task_names, data_root)
    dataset = build_multi_task_dataset(
        task_roots,
        tokenizer=tokenizer,
        target_fps=cfg.get("target_fps", 10),
        T=cfg.get("T", 16),
        action_dim=cfg.get("action_dim", 6),
        chunk_size=cfg.get("chunk_size", 1),
        instruction_level=instruction_level,
        max_pad_length=max_pad_length or cfg.get("max_pad_length", 128),
        load_rgb=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    batch_align_losses = []
    batch_align_scores = []
    positive_cosines = []
    negative_cosines = []
    num_samples = 0
    saved_bn_embeddings = []
    saved_lang_embeddings = []
    saved_task_names = []
    saved_episode_indices = []
    saved_phys_logits = []
    saved_phys_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(eval_device)
            attention_mask = batch["attention_mask"].to(eval_device)
            tactile_grid = batch["tactile_grid"].to(eval_device)
            task_names = batch["task_name"]
            episode_indices = [int(value) for value in batch["episode_index"]]

            dtype = next(model.bottleneck.parameters()).dtype
            z_tactile, _ = model.tactile_encoder(tactile_grid.to(dtype=dtype))
            vlm_hidden = model.encode_vision_language(input_ids, attention_mask)
            seq_lens = attention_mask.sum(dim=1).long() - 1
            lang_cls = vlm_hidden[torch.arange(vlm_hidden.size(0), device=eval_device), seq_lens]

            bottleneck_tokens = model.bottleneck(vlm_hidden, z_tactile)
            bn_cls = model.bottleneck.get_cls(bottleneck_tokens)

            align_loss = model.tla.compute_loss(bn_cls, lang_cls, task_names)
            projections = model.tla(bn_cls, lang_cls)
            proj_bn = projections["proj_bn"]
            proj_lang = projections["proj_lang"]
            cosine_matrix = proj_bn @ proj_lang.T
            pred_similarity = cosine_matrix.sigmoid()
            label_similarity = model.tla._compute_label_similarity(task_names, eval_device)

            positive_mask = label_similarity >= 0.999
            negative_mask = label_similarity <= 0.001
            diag_mask = torch.eye(label_similarity.size(0), device=eval_device, dtype=torch.bool)
            positive_mask = positive_mask & (~diag_mask)
            negative_mask = negative_mask & (~diag_mask)

            align_score = 1.0 - torch.abs(pred_similarity - label_similarity).mean().item()
            batch_align_losses.append(align_loss.item())
            batch_align_scores.append(align_score)

            positive_values = cosine_matrix[positive_mask]
            negative_values = cosine_matrix[negative_mask]
            if positive_values.numel() > 0:
                positive_cosines.extend(positive_values.detach().cpu().tolist())
            if negative_values.numel() > 0:
                negative_cosines.extend(negative_values.detach().cpu().tolist())

            phys_logits = model.phys_classifier(bn_cls)
            phys_labels = get_phys_labels(task_names, phys_logits.device)
            num_samples += int(phys_labels.size(0))

            saved_bn_embeddings.append(proj_bn.detach().cpu())
            saved_lang_embeddings.append(proj_lang.detach().cpu())
            saved_task_names.extend(task_names)
            saved_episode_indices.extend(episode_indices)
            saved_phys_logits.append(phys_logits.detach().cpu())
            saved_phys_labels.append(phys_labels.detach().cpu())

    overall_positive = float(np.mean(positive_cosines)) if positive_cosines else 0.0
    overall_negative = float(np.mean(negative_cosines)) if negative_cosines else 0.0

    results = {
        "checkpoint": checkpoint_path,
        "instruction_level": instruction_level,
        "num_samples": num_samples,
        "align_loss": float(np.mean(batch_align_losses)) if batch_align_losses else 0.0,
        "align_score": float(np.mean(batch_align_scores)) if batch_align_scores else 0.0,
        "cosine_positive": overall_positive,
        "cosine_negative": overall_negative,
        "cosine_gap": overall_positive - overall_negative,
        "phys_temperature": float(phys_temperature),
        "phys_aggregation": phys_aggregation,
        "per_task": {},
    }

    all_bn = torch.cat(saved_bn_embeddings, dim=0)
    all_lang = torch.cat(saved_lang_embeddings, dim=0)
    all_phys_logits = torch.cat(saved_phys_logits, dim=0)
    all_phys_labels = torch.cat(saved_phys_labels, dim=0)
    all_task_names = np.asarray(saved_task_names, dtype=object)

    if phys_aggregation == "episode":
        phys_logits_eval, phys_labels_eval, phys_task_names_eval, _ = _average_phys_logits_by_episode(
            all_phys_logits,
            all_phys_labels,
            saved_task_names,
            saved_episode_indices,
        )
    elif phys_aggregation == "none":
        phys_logits_eval = all_phys_logits
        phys_labels_eval = all_phys_labels
        phys_task_names_eval = list(saved_task_names)
    else:
        raise ValueError(f"Unknown phys_aggregation: {phys_aggregation}")

    overall_phys, per_task_phys = _compute_phys_metrics(
        phys_logits_eval,
        phys_labels_eval,
        phys_task_names_eval,
        temperature=phys_temperature,
    )
    results.update(overall_phys)

    for task_name in sorted(set(saved_task_names)):
        task_indices_np = np.where(all_task_names == task_name)[0]
        task_indices = torch.as_tensor(task_indices_np, dtype=torch.long)
        task_bn = all_bn[task_indices]
        task_labels = all_phys_labels[task_indices]

        sim = task_bn @ all_lang.T
        label_sim = (task_labels.unsqueeze(1) == all_phys_labels.unsqueeze(0)).float().mean(dim=-1)
        pred_similarity = sim.sigmoid()
        align_loss = F.mse_loss(pred_similarity, label_sim).item()
        align_score = 1.0 - torch.abs(pred_similarity - label_sim).mean().item()

        positive_mask = label_sim >= 0.999
        negative_mask = label_sim <= 0.001
        if task_indices.numel() > 0:
            positive_mask[torch.arange(task_indices.numel()), task_indices] = False

        positive_values = sim[positive_mask]
        negative_values = sim[negative_mask]
        task_positive = float(positive_values.mean().item()) if positive_values.numel() > 0 else 0.0
        task_negative = float(negative_values.mean().item()) if negative_values.numel() > 0 else 0.0
        results["per_task"][task_name] = {
            "align_loss": align_loss,
            "align_score": align_score,
            "cosine_positive": task_positive,
            "cosine_negative": task_negative,
            "cosine_gap": task_positive - task_negative,
            "num_samples": int(task_indices.numel()),
        }
        results["per_task"][task_name].update(per_task_phys.get(task_name, {}))

    if save_embeddings_path or vis_method != "none":
        embeddings = all_bn.float().cpu().numpy()
        phys_labels = all_phys_labels.float().cpu().numpy()
        if save_embeddings_path:
            save_path = Path(save_embeddings_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                save_path,
                embeddings=embeddings,
                task_names=np.asarray(saved_task_names, dtype=object),
                phys_labels=phys_labels,
            )
        if vis_method != "none":
            if not vis_output:
                raise ValueError("vis_output is required when vis_method is not 'none'")
            save_embedding_visualization(embeddings, saved_task_names, vis_output, vis_method)

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Stage A Representation Eval ===")
    print(f"align_loss      = {results['align_loss']:.6f}")
    print(f"align_score     = {results['align_score']:.6f}")
    print(f"phys_accuracy   = {results['phys_accuracy']:.6f}")
    print(f"phys_exact_match= {results['phys_exact_match']:.6f}")
    print(f"cosine_gap      = {results['cosine_gap']:.6f}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--instruction_level", default="L2")
    parser.add_argument("--max_pad_length", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save_embeddings", default=None)
    parser.add_argument("--vis_method", choices=["none", "pca", "tsne"], default="none")
    parser.add_argument("--vis_output", default=None)
    parser.add_argument("--task_names", nargs="+", default=None)
    parser.add_argument("--phys_temperature", type=float, default=1.25)
    parser.add_argument("--phys_aggregation", choices=["none", "episode"], default="episode")
    args = parser.parse_args()

    evaluate_stage_a(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        instruction_level=args.instruction_level,
        max_pad_length=args.max_pad_length,
        output_path=args.output,
        save_embeddings_path=args.save_embeddings,
        vis_method=args.vis_method,
        vis_output=args.vis_output,
        task_names=args.task_names,
        phys_temperature=args.phys_temperature,
        phys_aggregation=args.phys_aggregation,
    )


if __name__ == "__main__":
    main()