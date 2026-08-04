'''
Provide functionality to interactively disply progress, e.g. iteration with a progress bar.
'''

import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

# Attempt to use tqdm when it is available.
try:
    # Use tqdm's standard text renderer so progress updates do not depend on
    # notebook-widget support in the surrounding frontend.
    from tqdm import tqdm as _tqdm
# Fall back to a dependency-free progress display when tqdm is unavailable.
except ImportError:
    # Record the unavailable optional dependency without preventing execution.
    _tqdm = None


# Define the item type yielded by the progress-wrapper generator.
_Item = TypeVar('_Item')


def iterate_with_progress(
    items: Iterable[_Item],
    *,
    total: int,
    description: str,
    units: str,
    enabled: bool,
) -> Iterator[_Item]:
    '''
    Yield iterable items while optionally displaying completion progress.

    tqdm is used when installed. Otherwise, a small standard-library progress
    bar is written to standard error.

    Parameters
    ----------
    items
        Source iterable whose items are yielded unchanged.
    total
        Expected number of items.
    description
        Label shown beside the progress bar.
    units
        Units in which items are counted in the progress bar.
    enabled
        Whether progress output is displayed.

    Yields
    ------
    _Item
        Each source item in its original order.
    '''

    # Bypass all display work when progress output is disabled.
    if not enabled:
        # Forward the original iterable without adding output.
        yield from items
        # Stop after the source iterable has been exhausted.
        return

    # Prefer tqdm's standard dynamically updating text display when available.
    if _tqdm is not None:
        # Ensure that tqdm closes its display even when iteration raises.
        with _tqdm(
            total=total,
            desc=description,
            unit=units,
            leave=True,
            miniters=1,
            dynamic_ncols=True,
            bar_format=('{l_bar}{bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}, {rate_fmt}]'),
        ) as progress:
            # Consume the source in its original order.
            for item in items:
                # Yield the completed item to the caller.
                yield item
                # Advance the bar after the caller receives the item.
                progress.update(1)
                # Force the frontend to repaint after every completed item.
                progress.refresh()
        # Stop before entering the fallback implementation.
        return

    # Display the initial state of the dependency-free fallback.
    _write_basic_progress(
        description,
        units,
        0,
        total,
        final=(total == 0),
    )

    # Consume the source iterable sequentially.
    for completed, item in enumerate(items):
        # Yield the completed item to the caller.
        yield item
        # Render the updated fallback bar.
        _write_basic_progress(
            description,
            units,
            completed + 1,
            total,
            final=(completed + 1 == total),
        )


def _write_basic_progress(
    description: str,
    units: str,
    completed: int,
    total: int,
    *,
    final: bool,
) -> None:
    '''
    Write one compact progress-bar update to standard error.

    This renderer is used only when progress display is enabled and tqdm is not
    installed.

    Parameters
    ----------
    description
        Label identifying the active phase.
    units
        Units in which items are counted in the progress bar.
    completed
        Number of completed items.
    total
        Total number of items.
    final
        Whether this is the final update for the active phase.
    '''

    # Use a fixed width to keep the fallback display compact.
    bar_width = 30
    # Avoid division by zero for defensive handling of an empty iterable.
    fraction = completed / total if total else 1.0
    # Convert the completion fraction to filled character positions.
    filled_width = min(bar_width, int(round(bar_width * fraction)))
    # Construct the visual progress bar.
    bar = '#' * filled_width + '-' * (bar_width - filled_width)
    # End the completed bar with a newline and intermediate updates with return.
    ending = '\n' if final else '\r'
    # Render the update without interfering with the returned dataset.
    print(
        f'{description}: [{bar}] {completed}/{total} {units}',
        end=ending,
        file=sys.stderr,
        flush=True,
    )
