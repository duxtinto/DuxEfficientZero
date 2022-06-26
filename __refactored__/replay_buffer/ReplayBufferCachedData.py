from typing import List, Tuple

from numpy import ndarray

from core.game import GameHistory


class ReplayBufferCachedData:
    _buffer: List[GameHistory]
    _priorities: ndarray
    _game_look_up: List[Tuple[int, int]]
    _eps_collected: int
    _base_idx: int
    _clear_time: int

    def __init__(
            self,
            buffer: List[GameHistory],
            priorities: ndarray,
            game_look_up: List[Tuple[int, int]],
            eps_collected: int,
            base_idx: int,
            clear_time: int,
    ):
        self._buffer = buffer
        self._priorities = priorities
        self._game_look_up = game_look_up

        self._eps_collected = eps_collected
        self._base_idx = base_idx
        self._clear_time = clear_time

    @property
    def buffer(self):
        return self._buffer

    @property
    def priorities(self):
        return self._priorities

    @property
    def game_look_up(self):
        return self._game_look_up

    @property
    def eps_collected(self):
        return self._eps_collected

    @property
    def base_idx(self):
        return self._base_idx

    @property
    def clear_time(self):
        return self._clear_time
