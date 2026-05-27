from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

from torch.utils.data import Dataset

from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import Scene, SceneMetadata
from trajdata.dataset_specific.raw_dataset import RawDataset
from trajdata.utils import agent_utils, env_utils


# Per-worker cache of loaded RawDataset objects, keyed by env_name. Lives in
# the worker process and persists across __getitem__ calls. With fork-based
# workers (the default on Linux) each worker gets its own copy, populated
# lazily on first use and freed when the worker exits. This avoids paying
# RawDataset.load_dataset_obj() per scene — for nuPlan that constructor
# globs every log .db and runs a count(*) subquery per scene in each, which
# dominates wall-clock time on large splits.
_RAW_DATASET_CACHE: Dict[str, RawDataset] = {}


def scene_paths_collate_fn(filled_scenes: List) -> List[Optional[str]]:
    # Each item is a list of scene path strings (one per scene in a log
    # group). Flatten so the downstream consumer can treat the result as a
    # flat sequence of paths, matching the previous one-path-per-item shape.
    flat: List[Optional[str]] = []
    for item in filled_scenes:
        if item is None:
            continue
        flat.extend(item)
    return flat


def _log_key(scene_name: str) -> str:
    # nuPlan encodes the parent log as "<log_filename>=<scene_token>"; for
    # datasets that don't share data across scenes the whole name is the key,
    # so each scene becomes its own group (no behavior change).
    return scene_name.split("=", 1)[0]


class ParallelDatasetPreprocessor(Dataset):
    def __init__(
        self,
        scene_info_list: List[SceneMetadata],
        envs_dir_dict: Dict[str, str],
        env_cache_path: str,
        desired_dt: Optional[float],
        cache_class: Type[SceneCache],
        rebuild_cache: bool,
    ) -> None:
        self.env_cache_path: str = str(env_cache_path)
        self.desired_dt = desired_dt
        self.cache_class = cache_class
        self.rebuild_cache = rebuild_cache
        self.envs_dir_dict: Dict[str, str] = {
            k: str(v) for k, v in envs_dir_dict.items()
        }

        # Group scenes by (env_name, log_key) so each __getitem__ processes
        # all scenes from a single underlying source together. Preserves the
        # input order of groups, which lets callers control shuffling /
        # load-balancing upstream.
        groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
        order: List[Tuple[str, str]] = []
        for s in scene_info_list:
            key = (s.env_name, _log_key(s.name))
            bucket = groups.get(key)
            if bucket is None:
                groups[key] = bucket = []
                order.append(key)
            bucket.append((s.raw_data_idx, s.name))

        self._groups: List[Tuple[str, List[Tuple[int, str]]]] = [
            (env_name, groups[(env_name, log_key)]) for env_name, log_key in order
        ]

    def __len__(self) -> int:
        return len(self._groups)

    def _get_raw_dataset(self, env_name: str) -> RawDataset:
        cached = _RAW_DATASET_CACHE.get(env_name)
        if cached is not None:
            return cached

        raw_dataset = env_utils.get_raw_dataset(env_name, self.envs_dir_dict[env_name])
        # Leaving verbose False here so that we don't spam stdout with
        # loading messages from every worker.
        raw_dataset.load_dataset_obj(verbose=False)
        _RAW_DATASET_CACHE[env_name] = raw_dataset
        return raw_dataset

    def __getitem__(self, idx: int) -> List[Optional[str]]:
        env_cache: EnvCache = EnvCache(Path(self.env_cache_path))
        env_name, scenes = self._groups[idx]

        raw_dataset = self._get_raw_dataset(env_name)
        env_dt: float = raw_dataset.metadata.dt

        out: List[Optional[str]] = []
        for raw_idx, scene_name in scenes:
            scene_info = SceneMetadata(env_name, scene_name, env_dt, raw_idx)

            scene: Optional[Scene] = agent_utils.get_agent_data(
                scene_info,
                raw_dataset,
                env_cache,
                self.rebuild_cache,
                self.cache_class,
                self.desired_dt,
            )

            if scene is None:
                # Escape hatch — e.g. nuPlan single-frame scenes that can't
                # be used for prediction/planning.
                out.append(None)
                continue

            scene_path: Path = EnvCache.scene_metadata_path(
                env_cache.path, scene.env_name, scene.name, scene.dt
            )
            out.append(str(scene_path))

        return out
