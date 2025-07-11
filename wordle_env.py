from __future__ import annotations
import datetime as dt
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np

def evaluate_guess(guess: str, answer: str) -> np.ndarray:
    '''
    2 = green, 1 = yellow, 0 = grey
    '''
    feedback = np.zeros(5, dtype=np.int8)
    answer_chars = list(answer)

    for i, g in enumerate(guess):
        if g == answer[i]:
            feedback[i] = 2
            answer_chars[i] = None

    for i, g in enumerate(guess):
        if feedback[i] == 0 and g in answer_chars:
            feedback[i] = 1
            answer_chars[answer_chars.index(g)] = None

    return feedback

class WordleEnv(gym.Env):
    '''
    Observation:    shape = (6, 5): int matrix where rows are guesses so far
                    cell values: 2 = green, 1 = yellow, 0 = grey, -1 = empty

    Action:         Discrete(N) where N = len(allowed guesses)
                    Accesses full ~10k dictionary

    Reward:         6 - (guess #) if it wins
                    -1 if it loses
                    0 otherwise
    '''
    metadata = {"render_modes": ["ansi"]}

    def __init__(
            self,
            target: str | None = None,
            calendar: Dict[dt.date, str] | None = None,
            seed: int = 100,
            eval_mode: bool = False,
    ):
        super().__init__()
        self.rng = random.Random(seed)

        # dictionaries
        base = Path(__file__).parent

        self.allowed: List[str] = [w.strip().lower() for w in open(base / "data/valid_guesses.txt")]
        self.solutions = [w.strip().lower() for w in open(base / "data/valid_answers.txt")]

        if calendar is not None:
            self.calendar = calendar
            self.solutions = [calendar[d] for d in sorted(calendar)]
        else:
            self.calendar = {}

        self._candidate_set = set(self.solutions)  # start full
        self._potential = -np.log2(len(self._candidate_set))


        self.action_space = spaces.Discrete(len(self.allowed))
        self.observation_space = spaces.Box(
            low = -1,
            high = 2,
            shape = (6, 5),
            dtype = np.int8
        )

        # state
        self._answer: str | None = target
        self._eval_mode = eval_mode
        self._board = np.full((6,5), -1, dtype=np.int8)
        self._guess_index = 0
    
    def _update_candidates(self, guess, fb):
        def match(word):
            for i, (g, colour) in enumerate(zip(guess, fb)):
                if   colour == 2 and word[i] != g:             return False
                elif colour == 1 and (g not in word or word[i] == g): return False
                elif colour == 0 and g in word:                return False
            return True
        self._candidate_set = {w for w in self._candidate_set if match(w)}

    def reset(
            self,
            *,
            seed: int = 100,
            options = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._board[:] = -1
        self._guess_index = 0

        # if it's eval mode we choose today's word
        # otherwise, pick a training word
        if self._eval_mode:                       # daily live Wordle
            if self._answer is None:              # allow constructor override
                today = dt.date.today()
                if today not in self.calendar:
                    raise ValueError(f"No answer for {today}")
                self._answer = self.calendar[today]
        else:                                     # training ⇒ ALWAYS resample
            pool = self.solutions or self.allowed
            self._answer = self.rng.choice(pool)

        full_set = self.solutions
        self._candidate_set = set(full_set)
        self._potential     = -math.log2(len(self._candidate_set))

        return self._board.copy(), {}
    
    def step(self, action: int, alpha: float = 0.3):
        assert self._answer is not None, "call reset() first"
        guess = self.allowed[action]



        # colour feedback & update board
        fb = evaluate_guess(guess, self._answer)
        self._board[self._guess_index] = fb
        self._guess_index += 1

        terminated = bool((fb == 2).all() or self._guess_index == 6)
        reward = 0
        if terminated:
            if (fb == 2).all():     # win
                reward = 7 - self._guess_index     # 6,5,…,1
            else:                   # fail
                reward = -1
        else:
            prev_H = math.log2(max(len(self._candidate_set), 1))
            self._update_candidates(guess, fb)
            new_H  = math.log2(max(len(self._candidate_set), 1))

            info_gain = prev_H - new_H              # bits won this turn
            reward += alpha * info_gain

        return self._board.copy(), reward, terminated, False, {
            "guess": guess,
            "answer": self._answer,
            "step": self._guess_index,
        }
    
    def render(self, mode="ansi"):
        if mode != "ansi":
            raise NotImplementedError
        symbols = { -1: "·", 0: "⬜", 1: "🟨", 2: "🟩" }
        lines = ["".join(symbols[v] for v in row) for row in self._board]
        return "\n".join(lines)
    
    def close(self):
        pass