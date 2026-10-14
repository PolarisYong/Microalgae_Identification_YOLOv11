from __future__ import annotations

import math
from functools import lru_cache
from itertools import combinations
from typing import Iterable, Sequence

from .models import ChamberResult, GroupStats


def metrics(rows: Sequence[ChamberResult]) -> GroupStats:
    values = [row.umax for row in rows]
    n = len(values)
    if n < 2:
        raise ValueError("At least two values are required for sample statistics")

    mean = sum(values) / n
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (n - 1))
    m2 = sum((value - mean) ** 2 for value in values) / n
    if mean == 0 or m2 <= 0:
        return GroupStats(mean, sample_sd, math.inf, math.inf, 0.0)

    m3 = sum((value - mean) ** 3 for value in values) / n
    m4 = sum((value - mean) ** 4 for value in values) / n
    skew = m3 / (m2**1.5)
    kurtosis = m4 / (m2**2)
    jb = n / 6 * (skew**2 + (kurtosis - 3) ** 2 / 4)
    return GroupStats(mean, sample_sd, sample_sd / mean, jb, math.exp(-jb / 2))


def qualifies(
    rows: Sequence[ChamberResult],
    cv_limit: float = 0.15,
    normality_p_limit: float = 0.05,
) -> bool:
    if len(rows) < 2:
        return False
    result = metrics(rows)
    return result.cv < cv_limit and result.p_value > normality_p_limit


def sort_by_umax(rows: Iterable[ChamberResult]) -> list[ChamberResult]:
    return sorted(rows, key=lambda row: (row.umax, row.flow_no, row.chamber_no, row.source_file))


def sort_for_output(rows: Iterable[ChamberResult]) -> list[ChamberResult]:
    return sorted(rows, key=lambda row: (row.flow_no, row.chamber_no, row.source_file))


def valid_windows(
    sorted_rows: Sequence[ChamberResult],
    size: int,
    cv_limit: float,
    normality_p_limit: float,
) -> dict[int, tuple[list[ChamberResult], GroupStats]]:
    windows: dict[int, tuple[list[ChamberResult], GroupStats]] = {}
    for start in range(0, len(sorted_rows) - size + 1):
        rows = list(sorted_rows[start : start + size])
        if qualifies(rows, cv_limit, normality_p_limit):
            windows[start] = (rows, metrics(rows))
    return windows


def choose_full_groups(
    sorted_rows: Sequence[ChamberResult],
    group_size: int,
    cv_limit: float,
    normality_p_limit: float,
) -> tuple[list[list[ChamberResult]], int]:
    """Choose the maximum number of non-overlapping valid contiguous windows.

    Rows are sorted by umax. For ties in group count, the partition with the
    lowest sum of CV values wins. This generalizes the earlier three-group
    search while retaining its behavior for the L210‑N160‑IC5.25% and B datasets.
    """

    windows = valid_windows(sorted_rows, group_size, cv_limit, normality_p_limit)

    @lru_cache(maxsize=None)
    def solve(index: int) -> tuple[int, float, tuple[int, ...]]:
        if index > len(sorted_rows) - group_size:
            return 0, 0.0, ()

        best = solve(index + 1)
        if index in windows:
            tail_count, tail_cv, tail_starts = solve(index + group_size)
            candidate = (
                tail_count + 1,
                tail_cv + windows[index][1].cv,
                (index,) + tail_starts,
            )
            if candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] < best[1] - 1e-15
            ):
                best = candidate
        return best

    _, _, starts = solve(0)
    return [windows[start][0] for start in starts], len(windows)


def largest_qualifying_subset(
    remaining_rows: Sequence[ChamberResult],
    min_size: int,
    max_size: int,
    cv_limit: float,
    normality_p_limit: float,
    exhaustive_limit: int = 24,
) -> list[ChamberResult] | None:
    upper = min(max_size, len(remaining_rows))
    for size in range(upper, min_size - 1, -1):
        best_rows: list[ChamberResult] | None = None
        best_cv = math.inf

        for start in range(0, len(remaining_rows) - size + 1):
            candidate = list(remaining_rows[start : start + size])
            if qualifies(candidate, cv_limit, normality_p_limit):
                candidate_cv = metrics(candidate).cv
                if candidate_cv < best_cv:
                    best_rows, best_cv = candidate, candidate_cv
        if best_rows is not None:
            return best_rows

        if len(remaining_rows) <= exhaustive_limit:
            for candidate_tuple in combinations(remaining_rows, size):
                candidate = list(candidate_tuple)
                if qualifies(candidate, cv_limit, normality_p_limit):
                    candidate_cv = metrics(candidate).cv
                    if candidate_cv < best_cv:
                        best_rows, best_cv = candidate, candidate_cv
            if best_rows is not None:
                return best_rows
    return None


def _legacy_allocate_groups(
    rows: Sequence[ChamberResult],
    min_group_size: int = 8,
    max_group_size: int = 18,
    cv_limit: float = 0.15,
    normality_p_limit: float = 0.05,
) -> tuple[list[list[ChamberResult]], list[ChamberResult], int]:
    """Original DP-based splitter: spreads groups across the entire μmax range.

    Kept for one-shot rollback / debugging. The new ``allocate_groups`` supersedes
    it for "same culture" batches where all data should cluster tightly.
    """
    sorted_rows = sort_by_umax(rows)
    groups, window_count = choose_full_groups(
        sorted_rows,
        max_group_size,
        cv_limit,
        normality_p_limit,
    )

    used = {row.key for group in groups for row in group}
    remaining = [row for row in sorted_rows if row.key not in used]
    while True:
        next_group = largest_qualifying_subset(
            remaining,
            min_group_size,
            max_group_size - 1,
            cv_limit,
            normality_p_limit,
        )
        if next_group is None:
            break
        groups.append(next_group)
        chosen = {row.key for row in next_group}
        remaining = [row for row in remaining if row.key not in chosen]

    return groups, remaining, window_count


def _greedy_split_core(
    core: Sequence[ChamberResult],
    max_size: int,
    min_size: int,
    cv_limit: float,
    normality_p_limit: float,
) -> list[list[ChamberResult]]:
    """Carve a μmax-sorted core into as many groups of up to ``max_size`` as possible.

    On each iteration the largest ``size`` in ``[min_size, max_size]`` is tried
    first. For each ``size`` the contiguous window with the smallest CV that
    independently qualifies is taken. The last group may be shorter than
    ``max_size`` but must be at least ``min_size`` to be committed; otherwise the
    splitter stops.

    Used as the workhorse inside :func:`select_compact_groups` so the caller can
    explore many candidate core ranges cheaply.
    """
    groups: list[list[ChamberResult]] = []
    remaining = list(core)
    while len(remaining) >= min_size:
        taken: list[ChamberResult] | None = None
        for size in range(max_size, min_size - 1, -1):
            if len(remaining) < size:
                continue
            best_window: list[ChamberResult] | None = None
            best_cv = math.inf
            for s in range(0, len(remaining) - size + 1):
                win = list(remaining[s : s + size])
                if qualifies(win, cv_limit, normality_p_limit):
                    cv = metrics(win).cv
                    if cv < best_cv:
                        best_window, best_cv = win, cv
            if best_window is not None:
                taken = best_window
                break
        if taken is None:
            break
        groups.append(taken)
        taken_keys = {row.key for row in taken}
        remaining = [row for row in remaining if row.key not in taken_keys]
    return groups


def select_compact_groups(
    rows: Sequence[ChamberResult],
    max_size: int = 18,
    min_size: int = 12,
    cv_limit: float = 0.15,
    normality_p_limit: float = 0.05,
    max_mean_spread_ratio: float | None = None,
) -> tuple[list[list[ChamberResult]], list[ChamberResult]]:
    """Pick the largest compact set of groups for "same culture" batches.

    Rows are sorted by μmax ascending and packed into contiguous chunks:
    ``k`` full groups of exactly ``max_size`` members followed by at most one
    terminal partial group of ``min_size .. max_size - 1`` members (the
    trailing residual). Every chunk must *independently* satisfy
    ``CV < cv_limit`` and ``JB p > normality_p_limit``. Rows may be skipped
    between chunks, so boundary outliers never contaminate a group — and the
    whole candidate set is never required to qualify as one mega-group.

    (The earlier implementation required each candidate core range to pass
    CV+JB as a single mega-group before splitting it. On real data where a
    handful of extreme μmax rows push the overall CV above the limit, only
    tiny ranges ever qualified, collapsing the output to one small group.
    This DP removes that intermediate condition entirely.)

    The best packing maximises, in order:
      1. ``n_groups`` — as many groups as possible;
      2. small ``mean_spread`` — ``max(group means) - min(group means)``,
         keeping the μmax gap *between* groups tight;
      3. ``last_group_size`` — a larger trailing group beats a smaller one.

    ``max_mean_spread_ratio`` optionally lets "tight inter-group means" trump
    group count: the acceptable spread budget is ``max_mean_spread_ratio ×``
    (the overall mean μmax of the strict candidates). A packing whose
    inter-group mean spread fits within that budget beats any packing that
    exceeds it, even if the latter has more groups; among solutions on the
    same side of the budget the usual ``(count, spread, last_size)`` order
    applies. A single group always has spread 0, so the constraint never
    degenerates below one group. ``None`` (default) restores the
    unconstrained count-first behaviour.

    Computed with a right-to-left dynamic program over the sorted candidates.
    Undersized groups are only ever produced in the trailing position, so
    the compact-mode contract enforced by ``validation.py`` holds by
    construction.
    """
    sorted_rows = sort_by_umax(rows)
    n = len(sorted_rows)
    if n < min_size:
        return [], list(sorted_rows)

    mean_spread_limit: float | None = None
    if max_mean_spread_ratio is not None:
        overall_mean = sum(row.umax for row in sorted_rows) / n
        mean_spread_limit = max_mean_spread_ratio * overall_mean

    # Candidate tuple: (count, mean_spread, min_mean, max_mean, last_size, groups).
    # Each dp[pos] holds a Pareto set of non-dominated candidates. Dominance is
    # over (count ↑, max_mean ↓, mean_spread ↓, last_size ↑): if A dominates B,
    # any solution built on B is matched or beaten by the same solution built
    # on A — both as a standalone answer and when prefixed by an earlier chunk
    # (combined spread = sub.max_mean - chunk.mean, so a smaller max_mean also
    # keeps combinations within the optional spread budget). Retaining the full
    # frontier instead of a single "best" is what makes the budget constraint
    # exact: the sub that combines best with a given chunk is not always the
    # standalone-best sub.

    def dominates(a: tuple, b: tuple) -> bool:
        return (
            a[0] >= b[0]
            and a[3] <= b[3]
            and a[1] <= b[1]
            and a[4] >= b[4]
        )

    def insert(cands: list, cand: tuple) -> None:
        for existing in cands:
            if dominates(existing, cand):
                return
        cands[:] = [c for c in cands if not dominates(cand, c)]
        cands.append(cand)

    def final_better(candidate: tuple, incumbent: tuple) -> bool:
        if mean_spread_limit is not None:
            # Feasibility flag first: within-budget beats over-budget.
            if (candidate[1] <= mean_spread_limit) != (incumbent[1] <= mean_spread_limit):
                return candidate[1] <= mean_spread_limit
        if candidate[0] != incumbent[0]:
            return candidate[0] > incumbent[0]
        if candidate[1] != incumbent[1]:
            return candidate[1] < incumbent[1]
        return candidate[4] > incumbent[4]

    dp: list[list[tuple]] = [[] for _ in range(n + 1)]
    for pos in range(n - 1, -1, -1):
        cands: list[tuple] = list(dp[pos + 1])  # option: skip row ``pos``

        # Option A: a full group of exactly ``max_size`` starting at ``pos``.
        if max_size >= 2 and pos + max_size <= n:
            chunk = sorted_rows[pos : pos + max_size]
            stats = metrics(chunk)
            if stats.cv < cv_limit and stats.p_value > normality_p_limit:
                subs = dp[pos + max_size]
                if not subs:
                    # This chunk alone is a complete solution.
                    insert(cands, (1, 0.0, stats.mean, stats.mean, max_size, [chunk]))
                for sub in subs:
                    # The chunk mean is <= every sub row, hence <= sub's group
                    # means, so the combined spread is sub.max_mean - chunk mean.
                    insert(
                        cands,
                        (
                            sub[0] + 1,
                            sub[3] - stats.mean,
                            stats.mean,
                            sub[3],
                            sub[4],
                            [chunk] + sub[5],
                        ),
                    )

        # Option B: terminal partial groups of ``min_size .. max_size - 1``.
        # Terminal means: once taken, no further group may follow — this keeps
        # undersized groups in the trailing position (validation contract).
        # All qualifying sizes are inserted: a smaller partial trades a lower
        # max_mean against a smaller last_size, and the frontier needs both.
        for size in range(max_size - 1, min_size - 1, -1):
            if size < 2 or pos + size > n:
                continue
            chunk = sorted_rows[pos : pos + size]
            stats = metrics(chunk)
            if stats.cv < cv_limit and stats.p_value > normality_p_limit:
                insert(cands, (1, 0.0, stats.mean, stats.mean, size, [chunk]))

        dp[pos] = cands

    if not dp[0]:
        return [], list(sorted_rows)
    best = dp[0][0]
    for cand in dp[0][1:]:
        if final_better(cand, best):
            best = cand
    groups = best[5]
    used = {row.key for group in groups for row in group}
    remaining = [row for row in sorted_rows if row.key not in used]
    return groups, remaining


def allocate_groups(
    rows: Sequence[ChamberResult],
    min_group_size: int = 8,
    max_group_size: int = 18,
    cv_limit: float = 0.15,
    normality_p_limit: float = 0.05,
    max_mean_spread_ratio: float | None = None,
) -> tuple[list[list[ChamberResult]], list[ChamberResult], int]:
    """Compact-group selection entry point used by the pipeline.

    Returns ``(groups, remaining, full_group_count)``. The trailing integer is
    preserved in the public signature for downstream diagnostics; for the new
    algorithm it counts how many of the returned groups reached
    ``max_group_size``. The legacy DP-era window count is no longer meaningful
    and is recorded as ``full_group_count`` instead.

    ``max_mean_spread_ratio`` (optional) caps the inter-group mean spread at
    ``ratio × overall strict-candidate mean μmax`` — see
    :func:`select_compact_groups`.

    Roll back to the previous behaviour by importing ``_legacy_allocate_groups``
    from this module and routing the call accordingly.
    """
    groups, remaining = select_compact_groups(
        rows,
        max_size=max_group_size,
        min_size=min_group_size,
        cv_limit=cv_limit,
        normality_p_limit=normality_p_limit,
        max_mean_spread_ratio=max_mean_spread_ratio,
    )
    full_group_count = sum(1 for group in groups if len(group) == max_group_size)
    return groups, remaining, full_group_count
