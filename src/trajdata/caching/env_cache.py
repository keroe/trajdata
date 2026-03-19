from pathlib import Path
from typing import List, NamedTuple, Union

from trajdata.data_structures.scene_metadata import Scene
from trajdata.utils.cache_utils import safe_dill_dump, safe_dill_load


class EnvCache:
    def __init__(self, cache_location: Path) -> None:
        self.path = cache_location

    def env_is_cached(self, env_name: str) -> bool:
        return (self.path / env_name / "scenes_list.dill").exists()

    def scene_is_cached(self, env_name: str, scene_name: str, scene_dt: float) -> bool:
        return EnvCache.scene_metadata_path(
            self.path, env_name, scene_name, scene_dt
        ).is_file()

    @staticmethod
    def scene_metadata_path(
        base_path: Path, env_name: str, scene_name: str, scene_dt: float
    ) -> Path:
        return (
            base_path / env_name / scene_name / f"scene_metadata_dt{scene_dt:.2f}.dill"
        )

    def load_scene(self, env_name: str, scene_name: str, scene_dt: float) -> Scene:
        scene_file: Path = EnvCache.scene_metadata_path(
            self.path, env_name, scene_name, scene_dt
        )
        scene: Scene = safe_dill_load(scene_file)
        return scene

    def save_scene(self, scene: Scene) -> Path:
        scene_file: Path = EnvCache.scene_metadata_path(
            self.path, scene.env_name, scene.name, scene.dt
        )

        scene_cache_dir: Path = scene_file.parent
        scene_cache_dir.mkdir(parents=True, exist_ok=True)

        safe_dill_dump(scene, scene_file)

        return scene_file

    @staticmethod
    def save_scene_with_path(base_path: Path, scene: Scene) -> Path:
        scene_file: Path = EnvCache.scene_metadata_path(
            base_path, scene.env_name, scene.name, scene.dt
        )

        scene_cache_dir: Path = scene_file.parent
        scene_cache_dir.mkdir(parents=True, exist_ok=True)

        safe_dill_dump(scene, scene_file)

        return scene_file

    def load_env_scenes_list(self, env_name: str) -> List[NamedTuple]:
        env_cache_dir: Path = self.path / env_name
        scenes_list: List[NamedTuple] = safe_dill_load(
            env_cache_dir / "scenes_list.dill"
        )
        return scenes_list

    def save_env_scenes_list(
        self, env_name: str, scenes_list: List[NamedTuple]
    ) -> None:
        env_cache_dir: Path = self.path / env_name
        env_cache_dir.mkdir(parents=True, exist_ok=True)
        safe_dill_dump(scenes_list, env_cache_dir / "scenes_list.dill")

    @staticmethod
    def load(scene_info_path: Union[Path, str]) -> Scene:
        scene: Scene = safe_dill_load(Path(scene_info_path))
        return scene
