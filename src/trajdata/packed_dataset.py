"""Pre-packed LMDB Dataset for fast training.

PackedDataset wraps an LMDB file created by UnifiedDataset.pack_to_lmdb().
__getitem__ is a single LMDB read + pickle deserialize, eliminating per-sample
filesystem I/O from cache lookups and extras computation.
"""
import pickle
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import numpy as np
from torch.utils.data import Dataset

from trajdata.data_structures.collation import agent_collate_fn
from trajdata.data_structures.state import NP_STATE_TYPES, TORCH_STATE_TYPES
from trajdata.data_structures.agent import AgentType


class _ArrayWithFormat(np.ndarray):
    """numpy ndarray subclass with a ._format instance attribute.

    collation.py line 308 reads `curr_agent_state_np._format` for the state
    format string; line 374 calls `torch.as_tensor(curr_agent_state_np, ...)`.
    Both require a real ndarray.

    Unlike the dynamic StateArray subclasses (StateArrayXYZXdYdXddYddH...),
    this class has a fixed name in a fixed module, so pickle can always find it
    in every DDP rank subprocess without triggering AttributeError.
    """

    def __new__(cls, arr: np.ndarray, fmt: str):
        obj = np.asarray(arr).view(cls)
        obj._format = fmt
        return obj

    def __array_finalize__(self, obj):
        # Called whenever a new view/copy is made; propagate _format.
        self._format = getattr(obj, "_format", "")


class _ObsTypeStub:
    def __init__(self, obs_format: str):
        self._format = obs_format


class _CacheStub:
    def __init__(self, obs_format: str):
        self.obs_type = _ObsTypeStub(obs_format)


class PackedBatchElement:
    """Lightweight batch element compatible with agent_collate_fn."""

    __slots__ = (
        "data_index",
        "scene_ts",
        "dt",
        "agent_name",
        "agent_type",
        "scene_id",
        "agent_history_len",
        "agent_future_len",
        "num_neighbors",
        "history_sec",
        "future_sec",
        "curr_agent_state_np",
        "agent_history_np",
        "agent_history_extent_np",
        "agent_future_np",
        "agent_future_extent_np",
        "agent_from_world_tf",
        "neighbor_types_np",
        "neighbor_histories",
        "neighbor_history_extents",
        "neighbor_history_lens_np",
        "neighbor_futures",
        "neighbor_future_extents",
        "neighbor_future_lens_np",
        "robot_future_np",
        "extras",
        "cache",
        "map_name",
        "map_patch",
        "vec_map",
    )

    def __init__(
        self,
        data: dict,
        state_format: str,
        obs_format: str,
        history_sec: Tuple,
        future_sec: Tuple,
    ):
        self.data_index = data["data_index"]
        self.scene_ts = data["scene_ts"]
        self.dt = data["dt"]
        self.agent_name = data["agent_name"]
        self.agent_type = AgentType(data["agent_type"])
        self.scene_id = data["scene_id"]
        self.agent_history_len = data["agent_history_len"]
        self.agent_future_len = data["agent_future_len"]
        self.num_neighbors = data["num_neighbors"]
        self.history_sec = history_sec
        self.future_sec = future_sec

        # _ArrayWithFormat gives collation the ._format it needs (line 308)
        # without using dynamic StateArray subclasses that break IPC pickling
        # in DDP rank subprocesses.
        self.curr_agent_state_np = _ArrayWithFormat(
            data["curr_agent_state_np"], state_format
        )
        self.agent_history_np = data["agent_history_np"]
        self.agent_history_extent_np = data["agent_history_extent_np"]
        self.agent_future_np = data["agent_future_np"]
        self.agent_future_extent_np = data["agent_future_extent_np"]
        self.agent_from_world_tf = data["agent_from_world_tf"]

        self.neighbor_types_np = data["neighbor_types_np"]
        self.neighbor_histories = data["neighbor_histories"]
        self.neighbor_history_extents = data["neighbor_history_extents"]
        self.neighbor_history_lens_np = data["neighbor_history_lens_np"]
        self.neighbor_futures = data["neighbor_futures"]
        self.neighbor_future_extents = data["neighbor_future_extents"]
        self.neighbor_future_lens_np = data["neighbor_future_lens_np"]
        self.robot_future_np = data["robot_future_np"]
        self.extras = data["extras"]

        # Stubs required by agent_collate_fn
        self.cache = _CacheStub(obs_format)  # line 309: cache.obs_type._format
        self.map_name = None   # line 49: map_name
        self.map_patch = None  # line 46: map_patch is None check
        self.vec_map = None    # line 678: vec_map is not None check


class PackedDataset(Dataset):
    """torch.utils.data.Dataset backed by a pre-packed LMDB file.

    Created by UnifiedDataset.pack_to_lmdb(). Each __getitem__ is a single
    LMDB read + pickle deserialize — no filesystem I/O beyond that.

    LMDB handles are opened lazily per-worker (required after DataLoader fork).
    """

    def __init__(self, lmdb_path: "str | Path", *, lock: bool = False):
        """
        Args:
            lmdb_path: Path to the LMDB directory created by pack_to_lmdb().
            lock: Whether to use LMDB file locking. Set False on NFS/Lustre.
        """
        self._lmdb_path = Path(lmdb_path)
        self._lock = lock
        self._env = None  # opened lazily in each worker

        # Read metadata once at init (temp env, then close)
        env = lmdb.open(
            str(self._lmdb_path),
            readonly=True,
            lock=lock,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            metadata = pickle.loads(txn.get(b"__metadata__"))
        env.close()

        self._length = metadata["length"]
        self._state_format = metadata["state_format"]
        self._obs_format = metadata["obs_format"]
        self._history_sec = metadata["history_sec"]
        self._future_sec = metadata["future_sec"]

        # Pre-register dynamic state/obs tensor types in this process.
        # UnifiedDataset does this in __init__ (self.torch_state_type = ...).
        # Without it, agent_collate_fn running in DataLoader workers creates
        # StateTensorXYZ... via createStateType → globals()[name] = cls, but
        # the parent DDP rank process never runs createStateType for that format
        # and can't deserialize the collated AgentBatch from the worker queue.
        NP_STATE_TYPES[self._state_format]
        NP_STATE_TYPES[self._obs_format]
        TORCH_STATE_TYPES[self._state_format]
        TORCH_STATE_TYPES[self._obs_format]

    def _open_lmdb(self) -> None:
        self._env = lmdb.open(
            str(self._lmdb_path),
            readonly=True,
            lock=self._lock,
            readahead=False,
            meminit=False,
        )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> PackedBatchElement:
        if self._env is None:
            self._open_lmdb()
        with self._env.begin(write=False) as txn:
            buf = txn.get(str(idx).encode())
        data = pickle.loads(buf)
        return PackedBatchElement(
            data,
            state_format=self._state_format,
            obs_format=self._obs_format,
            history_sec=self._history_sec,
            future_sec=self._future_sec,
        )

    def get_collate_fn(self, return_dict: bool = False, pad_format: str = "outside"):
        """Return collate function compatible with agent_collate_fn.

        Signature matches UnifiedDataset.get_collate_fn so build_dataloader()
        works without modification.
        """
        return partial(
            agent_collate_fn,
            return_dict=return_dict,
            pad_format=pad_format,
        )
