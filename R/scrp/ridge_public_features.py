"""The public feature functions used by frozen Ridge-R.

Extracted without logic changes from the shared PLAN-045 C1/C2 modules. Other
C1/C2 selectors, fitters, bank code, and actor adapters are intentionally absent.
"""

from __future__ import annotations

from math import isfinite
import numpy as np

HORIZONS = (0, 2, 5, 10, 20)
UNKNOWN_RANK = 22
Q_PRIOR = 5183.0 / 9920.0


def _public_state(frame: object) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], int, int, float]:
    observation = getattr(frame, "observation", None)
    if observation is None:
        raise ValueError("C1 requires a detached public decision frame")
    raw_ranks = np.asarray(getattr(observation, "rank_tensor", None))
    if raw_ranks.ndim != 2 or not np.issubdtype(raw_ranks.dtype, np.integer):
        raise ValueError("public rank tensor must be a two-dimensional integer array")
    ranks = raw_ranks.astype(np.int64, copy=False)
    occupied = np.asarray(getattr(observation, "occupied_mask", None), dtype=np.bool_)
    legal_mask = np.asarray(getattr(observation, "legal_destination_mask", None), dtype=np.bool_)
    tiers = int(getattr(frame, "max_tiers", -1))
    if ranks.shape != occupied.shape or ranks.ndim != 2:
        raise ValueError("public rank and occupancy shapes differ")
    if legal_mask.ndim != 1 or legal_mask.shape[0] != ranks.shape[0]:
        raise ValueError("public legal-action mask must match the stack dimension")
    if ranks.shape != (legal_mask.shape[0], tiers) or tiers <= 0:
        raise ValueError("public frame dimensions differ from its masks")
    if np.any(ranks[~occupied] != 0):
        raise ValueError("empty padding must have rank zero")
    if np.any(occupied & ((ranks < 1) | (ranks > UNKNOWN_RANK))):
        raise ValueError("occupied ranks must be known ranks or sentinel 22")

    heights = occupied.sum(axis=1).astype(np.int64)
    expected_occupied = np.arange(tiers)[None, :] < heights[:, None]
    if not np.array_equal(occupied, expected_occupied):
        raise ValueError("occupied cells must form bottom-filled stack prefixes")
    known = ranks[occupied & (ranks < UNKNOWN_RANK)]
    if np.count_nonzero(known == 1) != 1 or len(np.unique(known)) != len(known):
        raise ValueError("public frame must have one current target and unique known ranks")

    source = int(np.argwhere(occupied & (ranks == 1))[0, 0])
    source_height = int(heights[source])
    if source_height < 2:
        raise ValueError("C1 features require a relocation decision with a source blocker")
    blocker = int(ranks[source, source_height - 1])
    if blocker == 1:
        raise ValueError("current target cannot be the moving blocker at a relocation decision")

    legal = tuple(int(value) for value in getattr(frame, "legal_destinations", ()))
    if legal != tuple(int(value) for value in np.flatnonzero(legal_mask)):
        raise ValueError("public legal destinations differ from the action mask")
    if not legal or len(set(legal)) != len(legal):
        raise ValueError("C1 requires distinct legal destinations")
    if source in legal or any(heights[index] >= tiers for index in legal):
        raise ValueError("legal destinations must exclude source and full stacks")

    q_prior = float(getattr(observation, "q_prior", float("nan")))
    q_t = float(getattr(observation, "q_t", float("nan")))
    if not isfinite(q_prior) or abs(q_prior - Q_PRIOR) > 1e-15:
        raise ValueError("public q_prior differs from the frozen 5183/9920 value")
    if not isfinite(q_t) or not 0.0 <= q_t <= 1.0:
        raise ValueError("public q_t must be a finite probability")
    return ranks, occupied, legal, tiers, blocker, q_t


def c1_destination_relations(frame: object) -> dict[int, tuple[float, ...]]:
    """Return the six public relation features used in Ridge's 37D base map."""
    ranks, occupied, legal, tiers, blocker, _q_t = _public_state(frame)
    blocker_known = 2 <= blocker <= 21
    relations: dict[int, tuple[float, ...]] = {}
    for destination in legal:
        mask = occupied[destination]
        values = ranks[destination, mask]
        height = int(mask.sum())
        known_values = values[(values >= 2) & (values <= 21)]
        unknown_count = int(np.count_nonzero(values == UNKNOWN_RANK))
        minimum_known = int(known_values.min()) if len(known_values) else UNKNOWN_RANK
        safe_known = blocker_known and bool(
            np.all((values >= 2) & (values <= 21) & (values > blocker))
        )
        earlier_count = (
            int(np.count_nonzero((values >= 2) & (values < blocker)))
            if blocker_known
            else 0
        )
        denominator = max(height, 1)
        relations[destination] = (
            float(tiers - height) / tiers,
            float(height == 0),
            float(unknown_count) / denominator,
            float(minimum_known) / UNKNOWN_RANK,
            float(safe_known),
            float(earlier_count) / denominator,
        )
    return relations


def c2_global_features(frame: object) -> np.ndarray:
    """Build the six public global features used in Ridge's 37D base map."""
    observation = frame.observation
    raw_ranks = np.asarray(observation.rank_tensor)
    if raw_ranks.ndim != 2 or not np.issubdtype(raw_ranks.dtype, np.integer):
        raise ValueError("C2 requires a two-dimensional integer public rank tensor")
    ranks = raw_ranks.astype(np.int64, copy=False)
    occupied = np.asarray(observation.occupied_mask, dtype=np.bool_)
    legal_mask = np.asarray(observation.legal_destination_mask, dtype=np.bool_)
    if ranks.shape != occupied.shape:
        raise ValueError("C2 public ranks and occupancy masks differ")
    if legal_mask.ndim != 1 or ranks.shape != (legal_mask.shape[0], int(frame.max_tiers)):
        raise ValueError("C2 public frame dimensions differ from the legal mask")
    if int(frame.max_tiers) <= 0 or ranks.shape[0] <= 0:
        raise ValueError("C2 public frame must have positive stack and tier dimensions")
    if np.any(ranks[~occupied] != 0):
        raise ValueError("C2 empty padding must have rank zero")
    if np.any(occupied & ((ranks < 1) | (ranks > UNKNOWN_RANK))):
        raise ValueError("C2 occupied ranks must be known or the public unknown sentinel")
    heights = occupied.sum(axis=1).astype(np.int64)
    expected_occupancy = np.arange(ranks.shape[1])[None, :] < heights[:, None]
    if not np.array_equal(occupied, expected_occupancy):
        raise ValueError("C2 occupied cells must form bottom-filled stack prefixes")
    legal = tuple(int(value) for value in np.flatnonzero(legal_mask))
    if legal != tuple(int(value) for value in frame.legal_destinations):
        raise ValueError("C2 legal destinations differ from the public mask")
    if not legal:
        raise ValueError("C2 requires at least one legal destination")

    target = np.argwhere(occupied & (ranks == 1))
    known_future = occupied & (ranks >= 2) & (ranks <= 21)
    if target.shape != (1, 2) or len(np.unique(ranks[known_future])) != int(known_future.sum()):
        raise ValueError("C2 frame must contain one current target and unique known ranks")
    source_stack, target_tier = (int(value) for value in target[0])
    source_height = int(heights[source_stack])
    if source_height <= target_tier + 1:
        raise ValueError("C2 horizon gate requires an active relocation decision")
    if source_stack in tuple(int(value) for value in np.flatnonzero(legal_mask)):
        raise ValueError("C2 legal destinations must exclude the source stack")
    stack_count, tier_count = ranks.shape
    occupied_count = int(occupied.sum())
    unknown_count = int(np.count_nonzero(occupied & (ranks == UNKNOWN_RANK)))
    known_future_count = int(known_future.sum())
    empty_count = int(np.count_nonzero(heights == 0))
    values = np.asarray(
        [
            1.0,
            occupied_count / float(stack_count * tier_count),
            (source_height - target_tier - 1) / float(tier_count),
            unknown_count / float(max(occupied_count, 1)),
            known_future_count / float(max(occupied_count, 1)),
            empty_count / float(stack_count),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("C2 global public features must be finite")
    return values

