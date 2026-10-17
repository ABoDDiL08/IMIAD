"""Create a disjoint normal-image validation set for threshold calibration."""

from __future__ import annotations

import random
from math import ceil
from typing import TypeVar


T = TypeVar("T")


def split_normal_train_validation(records: list[T], max_train: int | None,
                                  fraction: float, min_images: int,
                                  seed: int) -> tuple[list[T], list[T]]:
    """Keep max_train training records when unused normal records are available."""
    train = list(records[:max_train] if max_train is not None else records)
    if not train:
        raise ValueError("No normal training images are available")
    wanted = max(1, ceil(len(train) * fraction))
    unused = records[len(train):]
    rng = random.Random(seed)
    if unused:
        count = min(max(wanted, min_images), len(unused))
        positions = set(rng.sample(range(len(unused)), count))
        validation = [record for index, record in enumerate(unused) if index in positions]
    else:
        if len(train) < 2:
            raise ValueError("Automatic image threshold needs at least two normal training images")
        count = min(wanted, len(train) - 1)
        positions = set(rng.sample(range(len(train)), count))
        validation = [record for index, record in enumerate(train) if index in positions]
        train = [record for index, record in enumerate(train) if index not in positions]
    return train, validation
