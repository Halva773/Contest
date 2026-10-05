"""Split a pandas DataFrame without dividing equal timestamps between chunks."""

from collections.abc import Iterator
from numbers import Integral

import numpy as np
import pandas as pd


def chunk_by_datetime(
    frame: pd.DataFrame, datetime_column: str, chunk_size: int
) -> Iterator[pd.DataFrame]:
    """Yield chunks of at least ``chunk_size`` rows, except for the last one.

    Rows with the same timestamp always stay together. Chunks are ordered by
    timestamp; the original order of rows with equal timestamps is preserved.
    An empty input produces no chunks. Missing timestamps are rejected because
    they do not represent a date that can be assigned to a chunk.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if datetime_column not in frame.columns:
        raise KeyError(f"Column {datetime_column!r} does not exist")

    dates = frame[datetime_column]
    if not pd.api.types.is_datetime64_any_dtype(dates.dtype):
        raise TypeError(f"Column {datetime_column!r} must have a datetime dtype")
    if dates.hasnans:
        raise ValueError(f"Column {datetime_column!r} must not contain NaT")

    row_count = len(frame)
    if row_count == 0:
        return
    if row_count <= chunk_size:
        yield frame
        return

    if dates.is_monotonic_increasing:
        # The timestamp array and iloc slices do not copy the entire frame.
        timestamps = dates.array.asi8
        start = 0
        while start < row_count:
            if row_count - start <= chunk_size:
                end = row_count
            else:
                boundary = timestamps[start + chunk_size - 1]
                end = int(np.searchsorted(timestamps, boundary, side="right"))
            yield frame.iloc[start:end]
            start = end
        return

    # Group only row positions: sorting/copying every column is unnecessary.
    # Pandas sorts the distinct timestamps and keeps positions within each
    # timestamp in their original order.
    positions_by_date = dates.groupby(dates, sort=True).indices
    pending: list[np.ndarray] = []
    pending_size = 0
    for positions in positions_by_date.values():
        pending.append(positions)
        pending_size += len(positions)
        if pending_size >= chunk_size:
            indices = pending[0] if len(pending) == 1 else np.concatenate(pending)
            yield frame.take(indices)
            pending.clear()
            pending_size = 0

    if pending:
        indices = pending[0] if len(pending) == 1 else np.concatenate(pending)
        yield frame.take(indices)
