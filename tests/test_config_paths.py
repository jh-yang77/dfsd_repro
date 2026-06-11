from pathlib import Path

from dfsd_repro.config import load_config


def test_configs_use_repo_relative_outputs():
    for path in Path("dfsd_repro/configs").glob("*.json"):
        if path.name == "ablation_eta.json":
            continue
        cfg = load_config(path)
        assert str(cfg.output_root).startswith("dfsd_repro/results")
        assert cfg.subspace_cache_dir.startswith("dfsd_repro/cache/subspaces")

