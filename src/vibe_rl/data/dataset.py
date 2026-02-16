"""Dataset protocol for offline RL / imitation learning.

Defines the minimal interface that any dataset adapter must implement.
Both random-access (__getitem__) and iterable (__iter__) access are
supported so datasets work with both torch DataLoaders and direct
iteration.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vibe_rl.dataprotocol.transition import Transition


@runtime_checkable
class Dataset(Protocol):
    """Random-access + iterable dataset returning Transitions.

    Implementations must provide:
      - ``__len__``  : total number of transitions
      - ``__getitem__``: return a single ``Transition`` by integer index
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Transition: ...
