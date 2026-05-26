"""
VTLA/dataset/vtla_dataset.py

VTLA 三模态数据集 (3-Modality)

模态:
    1. RGB 多视角 (hand-eye + side)  → Qwen2-VL 原生 SigLIP ViT
    2. Tactile L+R                   → TactileEncoder
    3. Language (三级指令)            → Qwen2-VL tokenizer

输入/输出键:
  input_ids          [L]          tokenized instruction (with vision placeholders)
  attention_mask     [L]          token mask
  pixel_values       [N, C, pH, pW]  Qwen2-VL 格式 RGB patches
  image_grid_thw     [N_img, 3]      grid sizes for each image
  tactile_grid       [T, 2, 16, 16]
    state              [action_dim]    current robot state
  action             [chunk_size, action_dim]
  task_name          str
  instruction_level  str  ("L0"/"L1"/"L2")
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import av
except ImportError:
    av = None

from VTLA.dataset.instructions import (
    get_task_key,
    sample_material_probe_classification_prompt,
    sample_instruction,
)
from VTLA.dataset.material_labels import (
    UNKNOWN_MATERIAL_ID,
    get_material_label,
    get_material_name_from_task_name,
    normalize_material_name,
)

# ============================================================================
# Constants
# ============================================================================

SRC_FPS = 30
DEFAULT_TGT_FPS = 10
DEFAULT_T = 16
DEFAULT_ACTION_DIM = 6   # SO-101 6-DoF
DEFAULT_TAC_H = 16
DEFAULT_TAC_W = 16
PHASE_NAMES = ("approach", "contact", "lift", "retreat")


def _phase_from_progress(progress: float) -> str:
    if progress < 0.25:
        return "approach"
    if progress < 0.50:
        return "contact"
    if progress < 0.75:
        return "lift"
    return "retreat"


def _deterministic_repeat_count(key: str, weight: float) -> int:
    if weight <= 0:
        return 0
    base = int(math.floor(weight))
    frac = float(weight) - base
    if frac <= 0:
        return base
    digest = hashlib.md5(key.encode("utf-8"), usedforsecurity=False).digest()
    score = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return base + int(score < frac)


class VTLADataset(Dataset):
    """
    VTLA 三模态数据集 (3-Modality)

    Parameters
    ----------
    sidecar_root : str
        Sidecar 数据根目录 (tactile_raw_*, videos/)
    task_name : str
        任务名 (e.g., "inboxpicking-01")
    tokenizer : optional
        Qwen2-VL tokenizer (from AutoProcessor.tokenizer)
    processor : optional
        Qwen2-VL AutoProcessor (处理 text+images → pixel_values)
    target_fps : int
        降采样帧率 (default 10)
    T : int
        历史窗口长度 (default 16)
    action_dim : int
        动作维度 (default 6, SO-101)
    chunk_size : int
        Action chunk 大小 (default 1)
    instruction_level : str | None
        固定指令级别 (评估用), None=随机采样 (训练用)
    prompt_mode : str
        `action_instruction` 使用现有动作指令, `material_probe_material` 使用中性材料分类 prompt
    phase_filter : list[str] | None
        仅保留指定 phase 的样本 (e.g. ["contact", "lift"])
    max_pad_length : int
        Token 序列最大长度
    load_rgb : bool
        是否加载 RGB 图像 (False=text-only, True=3-modality)
    rgb_size : tuple
        RGB 缩放尺寸
    use_dummy : bool
        Dummy 模式
    n_dummy : int
        Dummy 样本数
    rgb_cache_dir : str | None
        预缓存 RGB 帧目录; 若为相对路径, 则相对于 sidecar_root
    """

    def __init__(
        self,
        sidecar_root: str,
        task_name: str = "inboxpicking-01",
        tokenizer=None,
        processor=None,
        target_fps: int = DEFAULT_TGT_FPS,
        T: int = DEFAULT_T,
        action_dim: int = DEFAULT_ACTION_DIM,
        chunk_size: int = 1,
        instruction_level: str | None = None,
        prompt_mode: str = "action_instruction",
        phase_filter: Optional[list[str]] = None,
        max_pad_length: int = 128,
        load_rgb: bool = False,
        rgb_size: tuple = (480, 640),
        rgb_cache_dir: str | None = None,
        phase_balance_weights: Optional[dict[str, float]] = None,
        use_dummy: bool = False,
        n_dummy: int = 50,
        episodes: Optional[list[int]] = None,
    ):
        super().__init__()
        self.sidecar_root = Path(sidecar_root) if sidecar_root else None
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.processor = processor
        self.T = T
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.instruction_level = instruction_level
        self.prompt_mode = str(prompt_mode)
        if self.prompt_mode not in {"action_instruction", "material_probe_material"}:
            raise ValueError(
                f"Unsupported prompt_mode: {prompt_mode!r}. Expected 'action_instruction' or 'material_probe_material'."
            )
        self.phase_filter = {str(phase).lower() for phase in (phase_filter or [])}
        self.max_pad_length = max_pad_length
        self.target_fps = target_fps
        self.stride = max(1, round(SRC_FPS / target_fps))
        self.use_dummy = use_dummy
        self.episodes = episodes
        self.phase_balance_weights = {
            phase: float(weight)
            for phase, weight in (phase_balance_weights or {}).items()
        }
        self.load_rgb = load_rgb
        self.rgb_size = rgb_size
        self.rgb_cache_dir = None
        self._task_material_name = get_material_name_from_task_name(task_name)
        self._episode_material_names: dict[int, str] = {}
        self._text_prompt_cache: dict[tuple[str, int], str] = {}
        self._text_only_token_cache: dict[str, dict[str, torch.Tensor]] = {}
        if rgb_cache_dir:
            cache_path = Path(rgb_cache_dir)
            if not cache_path.is_absolute() and self.sidecar_root is not None:
                cache_path = self.sidecar_root / cache_path
            self.rgb_cache_dir = cache_path

        # RGB video containers 缓存 {camera_name: {episode_idx: (path, total_frames)}}
        self._rgb_meta: dict[str, dict[int, tuple[Path, int]]] = {}
        if load_rgb and self.sidecar_root is not None:
            self._init_rgb_meta()
            if self.rgb_cache_dir is not None:
                print(f"[VTLADataset] RGB cache enabled: {self.rgb_cache_dir}")
        self.max_pad_length = max_pad_length
        self.target_fps = target_fps
        self.stride = max(1, round(SRC_FPS / target_fps))
        self.use_dummy = use_dummy
        self.episodes = episodes

        # ---- 加载 auto-annotate 元数据 (episode_metadata.json) ----
        self._episode_levels: dict[int, str] = {}
        if self.sidecar_root is not None:
            meta_path = self.sidecar_root / "episode_metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                task_material_name = normalize_material_name(
                    meta.get("task_attributes", {}).get("material_type")
                )
                if task_material_name is not None:
                    self._task_material_name = task_material_name
                for ep_key, ep_meta in meta.get("episodes", {}).items():
                    ep_idx = int(ep_key)
                    self._episode_levels[ep_idx] = ep_meta.get("level", None)
                    episode_material_name = normalize_material_name(
                        ep_meta.get("object_attributes", {}).get("material_type")
                    )
                    if episode_material_name is None:
                        episode_material_name = self._task_material_name
                    if episode_material_name is not None:
                        self._episode_material_names[ep_idx] = episode_material_name
                print(f"[VTLADataset] Loaded episode_metadata.json: "
                      f"{len(self._episode_levels)} episodes annotated")

        if use_dummy:
            self._n_dummy = n_dummy
            self._frames_df = None
            return

        # ---- 加载 parquet 元数据 ----
        dataset_root = Path(sidecar_root)
        data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
        if not data_files:
            # 尝试直接在 data/ 下找
            data_files = sorted((dataset_root / "data").glob("*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No parquet files in {dataset_root / 'data'}")

        self._frames_df = pd.concat(
            [pd.read_parquet(p) for p in data_files], ignore_index=True
        )
        if episodes is not None:
            self._frames_df = self._frames_df[
                self._frames_df["episode_index"].isin(episodes)
            ].reset_index(drop=True)

        total = len(self._frames_df)
        self._frame_bounds = self._build_frame_bounds()
        self._index_map = self._build_index_map()
        print(f"[VTLADataset] task={task_name} frames={total} stride={self.stride} samples={len(self._index_map)}")

    def __len__(self) -> int:
        if self.use_dummy:
            return self._n_dummy
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict:
        if self.use_dummy:
            return self._make_dummy(idx)
        return self._load_sample(idx)

    def _load_sample(self, idx: int) -> dict:
        row = self._frames_df.iloc[self._index_map[idx]]
        ep_idx = int(row["episode_index"])
        fr_idx = int(row["frame_index"])

        # ---- Tactile ----
        tactile_grid = self._load_tactile(ep_idx, fr_idx)  # [T, 2, 16, 16]

        # ---- Action ----
        action = self._extract_action(row)  # [chunk_size, action_dim]
        state = self._extract_state(row)  # [action_dim]

        # ---- Instruction ----
        # 优先级: explicit level > episode_metadata > random
        level = self.instruction_level  # None=random for training
        if level is None and ep_idx in self._episode_levels:
            level = self._episode_levels[ep_idx]
        if level is None:
            level = random.choice(["L0", "L1", "L2"])
        material_name = self._episode_material_names.get(ep_idx, self._task_material_name)
        material_label = get_material_label(material_name)
        is_material_probe = material_label != UNKNOWN_MATERIAL_ID
        if self.prompt_mode == "material_probe_material" and is_material_probe:
            instruction = sample_material_probe_classification_prompt()
        else:
            instruction = sample_instruction(self.task_name, level=level)
        phase_name = self._infer_phase(ep_idx, fr_idx)

        # ---- RGB (3-modality) ----
        # Hand-eye camera: 优先 realsense_rgb, 无则 fallback 到 depth (egg/sponge 数据命名问题)
        rgb_images = []
        if self.load_rgb:
            handeye_img = self._load_rgb_frame(ep_idx, fr_idx, "observation.images.realsense_rgb")
            if handeye_img is None:
                handeye_img = self._load_rgb_frame(ep_idx, fr_idx, "observation.images.depth")
            if handeye_img is not None:
                rgb_images.append(handeye_img)
            side_img = self._load_rgb_frame(ep_idx, fr_idx, "observation.images.side")
            if side_img is not None:
                rgb_images.append(side_img)

        # ---- Tokenize (with optional images) ----
        if self.processor is not None and rgb_images:
            # 3-modality: 使用 Qwen2-VL processor 处理 text + images
            # 构造 Qwen2-VL 格式: <|vision_start|><|image_pad|>...<|vision_end|> + text
            cache_key = (instruction, len(rgb_images))
            text_prompt = self._text_prompt_cache.get(cache_key)
            if text_prompt is None:
                messages = [{
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": f"<image_{idx}>"} for idx in range(len(rgb_images))],
                        {"type": "text", "text": instruction},
                    ],
                }]
                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                self._text_prompt_cache[cache_key] = text_prompt
            proc_out = self.processor(
                text=[text_prompt],
                images=rgb_images,
                padding="max_length",
                truncation=True,
                max_length=self.max_pad_length,
                return_tensors="pt",
            )
            input_ids = proc_out["input_ids"].squeeze(0)
            attention_mask = proc_out["attention_mask"].squeeze(0)
            pixel_values = proc_out.get("pixel_values", None)
            image_grid_thw = proc_out.get("image_grid_thw", None)
            if pixel_values is not None:
                pixel_values = pixel_values.squeeze(0)  # remove batch dim
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.squeeze(0)
        elif self.tokenizer is not None:
            cached_tokens = self._text_only_token_cache.get(instruction)
            if cached_tokens is None:
                tokens = self.tokenizer(
                    instruction,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_pad_length,
                    return_tensors="pt",
                )
                cached_tokens = {
                    "input_ids": tokens["input_ids"].squeeze(0),
                    "attention_mask": tokens["attention_mask"].squeeze(0),
                }
                self._text_only_token_cache[instruction] = cached_tokens
            input_ids = cached_tokens["input_ids"].clone()
            attention_mask = cached_tokens["attention_mask"].clone()
            pixel_values = None
            image_grid_thw = None
        else:
            input_ids = torch.zeros(self.max_pad_length, dtype=torch.long)
            attention_mask = torch.ones(self.max_pad_length, dtype=torch.long)
            pixel_values = None
            image_grid_thw = None

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tactile_grid": tactile_grid.float(),
            "state": state.float(),
            "action": action.float(),
            "task_name": self.task_name,
            "instruction": instruction,
            "instruction_level": level,
            "material_label": torch.tensor(material_label, dtype=torch.long),
            "material_name": material_name or "none",
            "is_material_probe": torch.tensor(is_material_probe, dtype=torch.bool),
            "episode_index": ep_idx,
            "episode_id": f"{self.task_name}:{ep_idx}",
            "frame_index": fr_idx,
            "phase_name": phase_name,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
            result["image_grid_thw"] = image_grid_thw
        return result

    def _build_frame_bounds(self) -> dict[int, tuple[int, int]]:
        grouped = self._frames_df.groupby("episode_index")["frame_index"]
        return {
            int(ep_idx): (int(frame_min), int(frame_max))
            for ep_idx, frame_min, frame_max in zip(grouped.min().index, grouped.min().values, grouped.max().values)
        }

    def _infer_phase(self, episode_index: int, frame_index: int) -> str:
        frame_min, frame_max = self._frame_bounds.get(episode_index, (frame_index, frame_index))
        denom = max(frame_max - frame_min, 1)
        progress = (frame_index - frame_min) / denom
        return _phase_from_progress(progress)

    def _build_index_map(self) -> list[int]:
        sampled_positions = list(range(0, len(self._frames_df), self.stride))
        if not self.phase_balance_weights:
            return sampled_positions

        index_map: list[int] = []
        phase_stats = {
            phase: {"base": 0, "sampled": 0}
            for phase in PHASE_NAMES
        }
        for pos in sampled_positions:
            row = self._frames_df.iloc[pos]
            ep_idx = int(row["episode_index"])
            fr_idx = int(row["frame_index"])
            phase = self._infer_phase(ep_idx, fr_idx)
            if self.phase_filter and phase not in self.phase_filter:
                continue
            phase_stats[phase]["base"] += 1
            weight = self.phase_balance_weights.get(phase, 1.0)
            repeat = _deterministic_repeat_count(f"{self.task_name}:{ep_idx}:{fr_idx}", weight)
            if repeat <= 0:
                continue
            index_map.extend([pos] * repeat)
            phase_stats[phase]["sampled"] += repeat

        stats_str = ", ".join(
            f"{phase}:{phase_stats[phase]['base']}->{phase_stats[phase]['sampled']}"
            for phase in PHASE_NAMES
        )
        print(
            f"[VTLADataset] phase_balance task={self.task_name} weights={self.phase_balance_weights} {stats_str}"
        )
        return index_map

    # ====================================================================== #
    # RGB loading (3-modality)
    # ====================================================================== #

    def _init_rgb_meta(self):
        """扫描 RGB mp4 文件路径 (不打开视频)"""
        for cam in ["observation.images.realsense_rgb", "observation.images.depth", "observation.images.side"]:
            cam_dir = self.sidecar_root / "videos" / cam
            if not cam_dir.exists():
                continue
            mp4_files = sorted(cam_dir.rglob("*.mp4"))
            for mp4 in mp4_files:
                # 每个 episode 对应一个 mp4 文件
                # 从 parquet 元数据中推断 episode 映射
                self._rgb_meta.setdefault(cam, {})
                # file-000.mp4 → episode 0, etc.
                stem = mp4.stem  # "file-000"
                try:
                    ep_offset = int(stem.split("-")[-1])
                except ValueError:
                    ep_offset = 0
                self._rgb_meta[cam][ep_offset] = (mp4, -1)  # frames unknown until opened

    def _prepare_rgb_image(self, image: Image.Image) -> Image.Image:
        """统一 RGB 缩放逻辑, 保持在线 decode 与缓存读取一致。"""
        if self.rgb_size and image.size != (self.rgb_size[1], self.rgb_size[0]):
            image = image.resize((self.rgb_size[1], self.rgb_size[0]), Image.BILINEAR)
        return image

    def _rgb_cache_path(self, ep_idx: int, fr_idx: int, camera_name: str) -> Path:
        return self.rgb_cache_dir / camera_name / f"episode-{ep_idx:06d}" / f"frame-{fr_idx:06d}.jpg"

    def _load_cached_rgb_frame(
        self, ep_idx: int, fr_idx: int, camera_name: str
    ) -> Optional[Image.Image]:
        """优先从预缓存帧读取, 避免每个 sample 都打开 mp4。"""
        if self.rgb_cache_dir is None:
            return None

        cache_path = self._rgb_cache_path(ep_idx, fr_idx, camera_name)
        if not cache_path.exists():
            return None

        try:
            with Image.open(cache_path) as img:
                return self._prepare_rgb_image(img.convert("RGB"))
        except Exception:
            return None

    def _save_cached_rgb_frame(
        self, image: Image.Image, ep_idx: int, fr_idx: int, camera_name: str
    ) -> None:
        """将在线 decode 到的帧回填到缓存, 后续 epoch 直接命中。"""
        if self.rgb_cache_dir is None:
            return

        cache_path = self._rgb_cache_path(ep_idx, fr_idx, camera_name)
        if cache_path.exists():
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prepared = self._prepare_rgb_image(image.convert("RGB"))
        prepared.save(cache_path, format="JPEG", quality=90)

    def _load_rgb_frame(
        self, ep_idx: int, fr_idx: int, camera_name: str
    ) -> Optional[Image.Image]:
        """从 mp4 中提取单帧 RGB (PIL Image)"""
        cached = self._load_cached_rgb_frame(ep_idx, fr_idx, camera_name)
        if cached is not None:
            return cached

        if av is None:
            return None
        cam_meta = self._rgb_meta.get(camera_name, {})
        if not cam_meta:
            return None

        # 查找对应 episode 的 mp4
        # episodes 可能跨 chunk, 但通常 file-000.mp4 包含所有帧
        # 使用 episode 0 作为默认 (单 chunk 场景)
        mp4_path = None
        if ep_idx in cam_meta:
            mp4_path = cam_meta[ep_idx][0]
        elif 0 in cam_meta:
            mp4_path = cam_meta[0][0]
        else:
            return None

        try:
            container = av.open(str(mp4_path))
            stream = container.streams.video[0]
            # 直接 seek 到目标帧附近
            target_pts = int(fr_idx * stream.time_base.denominator / SRC_FPS)
            container.seek(target_pts, stream=stream)
            for frame in container.decode(video=0):
                img = self._prepare_rgb_image(frame.to_image())  # PIL Image
                self._save_cached_rgb_frame(img, ep_idx, fr_idx, camera_name)
                container.close()
                return img
            container.close()
        except Exception:
            pass
        return None

    # ====================================================================== #
    # Sidecar loading
    # ====================================================================== #

    def _load_tactile(self, ep_idx: int, fr_idx: int) -> torch.Tensor:
        """加载双触觉序列 [T, 2, 16, 16]

        支持两种存储格式:
          A) 单文件: tactile_raw_left/episode_XXXXXX.npy  (shape=[N_frames, 16, 16])
          B) 目录:   tactile_raw_left/episode-XXXXXX/frame-XXXXXX.npy  (per-frame 16×16)
        """
        if self.sidecar_root is None:
            return torch.zeros(self.T, 2, DEFAULT_TAC_H, DEFAULT_TAC_W)

        left = self._load_tactile_channel(
            self.sidecar_root / "tactile_raw_left", ep_idx, fr_idx
        )
        right = self._load_tactile_channel(
            self.sidecar_root / "tactile_raw_right", ep_idx, fr_idx
        )
        return torch.stack([left, right], dim=1)  # [T, 2, 16, 16]

    def _load_tactile_channel(
        self, channel_dir: Path, ep_idx: int, fr_idx: int
    ) -> torch.Tensor:
        """加载单通道触觉数据 (自动检测格式 A 或 B)"""
        # 格式 B: episode-XXXXXX/ 目录 (per-frame npy)
        ep_dir = channel_dir / f"episode-{ep_idx:06d}"
        if ep_dir.is_dir():
            return self._load_npy_window_from_dir(ep_dir, fr_idx)

        # 格式 A: episode_XXXXXX.npy 单文件
        ep_file = channel_dir / f"episode_{ep_idx:06d}.npy"
        if ep_file.exists():
            arr = np.load(str(ep_file), mmap_mode="r")
            window = self._build_window(arr, fr_idx)
            return torch.from_numpy(window.copy()).float()

        return torch.zeros(self.T, DEFAULT_TAC_H, DEFAULT_TAC_W)

    def _load_npy_window_from_dir(self, ep_dir: Path, fr_idx: int) -> torch.Tensor:
        """从 per-frame npy 目录加载 T 帧窗口"""
        frame_ids = [fr_idx - (self.T - 1 - i) * self.stride for i in range(self.T)]
        frames = []
        for fid in frame_ids:
            fid_clamped = max(0, fid)
            npy_path = ep_dir / f"frame-{fid_clamped:06d}.npy"
            if npy_path.exists():
                frames.append(np.load(str(npy_path)))
            else:
                # 超出范围: 用最近的有效帧或零填充
                all_frames = sorted(ep_dir.glob("frame-*.npy"))
                if all_frames:
                    if fid < 0:
                        frames.append(np.load(str(all_frames[0])))
                    else:
                        frames.append(np.load(str(all_frames[-1])))
                else:
                    frames.append(np.zeros((DEFAULT_TAC_H, DEFAULT_TAC_W), dtype=np.float32))
        return torch.from_numpy(np.stack(frames, axis=0)).float()

    def _build_window(self, arr: np.ndarray, fr_idx: int) -> np.ndarray:
        """从全帧数组提取 T 帧历史窗口 (30FPS → stride 降采样)"""
        N = len(arr)
        frame_ids = [fr_idx - (self.T - 1 - i) * self.stride for i in range(self.T)]
        frames = []
        for fid in frame_ids:
            if fid < 0:
                frames.append(arr[0])
            elif fid >= N:
                frames.append(arr[-1])
            else:
                frames.append(arr[fid])
        return np.stack(frames, axis=0)

    def _extract_action(self, row) -> torch.Tensor:
        """提取 action 并构造 chunk"""
        if "action" in row and row["action"] is not None:
            action = np.asarray(row["action"], dtype=np.float32)
        else:
            action = np.zeros(self.action_dim, dtype=np.float32)

        # 截断或填充到 action_dim
        if len(action) > self.action_dim:
            action = action[:self.action_dim]
        elif len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)))

        action_t = torch.from_numpy(action)
        # 复制为 chunk
        return action_t.unsqueeze(0).expand(self.chunk_size, -1).contiguous()

    def _extract_state(self, row) -> torch.Tensor:
        """提取当前 robot state 并对齐到 action_dim。"""
        if "observation.state" in row and row["observation.state"] is not None:
            state = np.asarray(row["observation.state"], dtype=np.float32)
        else:
            state = np.zeros(self.action_dim, dtype=np.float32)

        if len(state) > self.action_dim:
            state = state[:self.action_dim]
        elif len(state) < self.action_dim:
            state = np.pad(state, (0, self.action_dim - len(state)))

        return torch.from_numpy(state)

    # ====================================================================== #
    # Dummy
    # ====================================================================== #

    def _make_dummy(self, idx: int) -> dict:
        level = self.instruction_level
        if level is None:
            level = random.choice(["L0", "L1", "L2"])
        material_name = self._episode_material_names.get(0, self._task_material_name)
        material_label = get_material_label(material_name)
        is_material_probe = material_label != UNKNOWN_MATERIAL_ID
        if self.prompt_mode == "material_probe_material" and is_material_probe:
            instruction = sample_material_probe_classification_prompt()
        else:
            instruction = sample_instruction(self.task_name, level=level)
        input_ids = torch.zeros(self.max_pad_length, dtype=torch.long)
        attention_mask = torch.ones(self.max_pad_length, dtype=torch.long)
        pixel_values = None
        image_grid_thw = None

        if self.load_rgb and self.processor is not None:
            dummy_mm_cache = getattr(self, "_dummy_mm_cache", None)
            if dummy_mm_cache is None:
                blank_images = [
                    Image.new("RGB", (self.rgb_size[1], self.rgb_size[0]), color=(0, 0, 0)),
                    Image.new("RGB", (self.rgb_size[1], self.rgb_size[0]), color=(0, 0, 0)),
                ]
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "<image_0>"},
                        {"type": "image", "image": "<image_1>"},
                        {"type": "text", "text": "dummy instruction"},
                    ],
                }]
                text_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                proc_out = self.processor(
                    text=[text_prompt],
                    images=blank_images,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_pad_length,
                    return_tensors="pt",
                )
                dummy_mm_cache = {
                    "input_ids": proc_out["input_ids"].squeeze(0),
                    "attention_mask": proc_out["attention_mask"].squeeze(0),
                    "pixel_values": proc_out.get("pixel_values"),
                    "image_grid_thw": proc_out.get("image_grid_thw"),
                }
                if dummy_mm_cache["pixel_values"] is not None:
                    dummy_mm_cache["pixel_values"] = dummy_mm_cache["pixel_values"].squeeze(0)
                if dummy_mm_cache["image_grid_thw"] is not None:
                    dummy_mm_cache["image_grid_thw"] = dummy_mm_cache["image_grid_thw"].squeeze(0)
                self._dummy_mm_cache = dummy_mm_cache

            input_ids = self._dummy_mm_cache["input_ids"].clone()
            attention_mask = self._dummy_mm_cache["attention_mask"].clone()
            pixel_values = self._dummy_mm_cache["pixel_values"]
            image_grid_thw = self._dummy_mm_cache["image_grid_thw"]
            if pixel_values is not None:
                pixel_values = pixel_values.clone()
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.clone()
        elif self.tokenizer is not None:
            tokens = self.tokenizer(
                instruction,
                padding="max_length",
                truncation=True,
                max_length=self.max_pad_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tactile_grid": torch.randn(self.T, 2, DEFAULT_TAC_H, DEFAULT_TAC_W),
            "state": torch.randn(self.action_dim),
            "action": torch.randn(self.chunk_size, self.action_dim),
            "task_name": self.task_name,
            "instruction": instruction,
            "instruction_level": level,
            "material_label": torch.tensor(material_label, dtype=torch.long),
            "material_name": material_name or "none",
            "is_material_probe": torch.tensor(is_material_probe, dtype=torch.bool),
            "episode_index": 0,
            "episode_id": f"{self.task_name}:0",
            "frame_index": idx,
            "phase_name": self._infer_phase(0, idx),
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values
            result["image_grid_thw"] = image_grid_thw
        return result


def build_multi_task_dataset(
    data_roots: dict[str, str | dict[str, object]],
    tokenizer=None,
    processor=None,
    target_fps: int = DEFAULT_TGT_FPS,
    T: int = DEFAULT_T,
    instruction_level: str | None = None,
    load_rgb: bool = False,
    task_repeat_factors: Optional[dict[str, float]] = None,
    family_repeat_factors: Optional[dict[str, float]] = None,
    phase_balance_weights: Optional[dict[str, float]] = None,
    phase_balance_families: Optional[list[str]] = None,
    **kwargs,
) -> torch.utils.data.ConcatDataset:
    """
    构建多任务拼接数据集

    data_roots: {"inboxpicking-01": "/data/inboxpicking-01", "grasp-blueberry-01": "/data/grasp-blueberry-01"}
        也支持 {"task-alias": {"root": "/data/task", "episodes": [0, 1, ...]}}
    """
    datasets = []
    task_repeat_factors = task_repeat_factors or {}
    family_repeat_factors = family_repeat_factors or {}
    phase_balance_weights = phase_balance_weights or {}
    phase_balance_family_set = set(phase_balance_families or [])

    for task_name, root_spec in data_roots.items():
        sidecar_root = root_spec
        episodes = None
        if isinstance(root_spec, dict):
            sidecar_root = root_spec["root"]
            episodes = root_spec.get("episodes")

        task_phase_weights = None
        if phase_balance_weights:
            task_family = get_task_key(task_name)
            if not phase_balance_family_set or task_family in phase_balance_family_set:
                task_phase_weights = phase_balance_weights

        ds = VTLADataset(
            sidecar_root=sidecar_root,
            task_name=task_name,
            tokenizer=tokenizer,
            processor=processor,
            target_fps=target_fps,
            T=T,
            instruction_level=instruction_level,
            load_rgb=load_rgb,
            episodes=episodes,
            phase_balance_weights=task_phase_weights,
            **kwargs,
        )
        repeat_factor = task_repeat_factors.get(task_name)
        if repeat_factor is None:
            repeat_factor = family_repeat_factors.get(get_task_key(task_name), 1.0)
        repeat_count = max(1, math.ceil(float(repeat_factor)))
        datasets.extend([ds] * repeat_count)
        if repeat_count > 1:
            print(
                f"[VTLADataset] oversampling task={task_name} family={get_task_key(task_name)} "
                f"repeat_factor={repeat_factor} repeat_count={repeat_count} base_samples={len(ds)}"
            )
    return torch.utils.data.ConcatDataset(datasets)
