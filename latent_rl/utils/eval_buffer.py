import json
from pathlib import Path
from typing import Any

import numpy as np


class EvalBuffer:
    def __init__(self, root: Path) -> None:
        missing = [
            p for p in (root / "buffer.npz", root / "meta.json") if not p.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"eval buffer files {[str(p) for p in missing]} not found; "
                "run `just eval-buffer` first"
            )
        self.meta: dict[str, Any] = json.loads((root / "meta.json").read_text())
        self._npz = np.load(root / "buffer.npz")
        # NpzFile decompresses on every access; obs alone is ~500 MB
        self._cache: dict[str, np.ndarray] = {}

    @property
    def algos(self) -> list[str]:
        return self.meta["algos"]

    def __len__(self) -> int:
        return self.meta["fields"]["obs"]["shape"][0]

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._npz.files:
            raise KeyError(
                f"no field {key!r} in the eval buffer; have {self._npz.files}"
            )
        if key not in self._cache:
            self._cache[key] = self._npz[key]
        return self._cache[key]

    def latents(self, algo: str, next_step: bool = False) -> np.ndarray:
        if algo not in self.algos:
            raise KeyError(f"no {algo!r} latents in the eval buffer; have {self.algos}")
        return self[f"z_next_{algo}" if next_step else f"z_{algo}"]

    def resample_episodes(self, rng: np.random.Generator) -> np.ndarray:
        # frames within an episode are correlated, so resample whole episodes;
        # per source, so every policy keeps its share of the frames
        episode, source = self["episode"], self["source"]
        starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
        ends = np.r_[starts[1:], len(episode)]
        picks = []
        for s in np.unique(source):
            own = np.flatnonzero(source[starts] == s)
            picks.append(rng.choice(own, size=len(own)))
        return np.concatenate(
            [np.arange(starts[e], ends[e]) for e in np.concatenate(picks)]
        )
