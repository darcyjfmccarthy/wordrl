from __future__ import annotations
import datetime as dt
import random
import torch
import math
import string
from pathlib import Path
from typing import List, Tuple, Dict
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch.nn as nn

LETTER2IDX = {c: i for i, c in enumerate(string.ascii_lowercase)}
PAD_TOKEN   = 26

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
            alpha: float = 1.0,
            green_bonus: float = 0.1
    ):
        super().__init__()
        self.rng = random.Random(seed)

        # dictionaries
        base = Path(__file__).parent

        self.allowed: List[str] = [w.strip().lower() for w in open(base / "data/valid_guesses.txt")]
        self.solutions = [w.strip().lower() for w in open(base / "data/valid_answers.txt")]

        small_solutions = self.solutions[:200]
        self.solutions = small_solutions
        self.allowed = small_solutions

        if calendar is not None:
            self.calendar = calendar
            self.solutions = [calendar[d] for d in sorted(calendar)]
        else:
            self.calendar = {}

        self._candidate_set = set(self.solutions)  # start full
        self._potential = -np.log2(len(self._candidate_set))

        self._colours = np.full((6, 5), -1,  dtype=np.int8)
        self._letters = np.full((6, 5), PAD_TOKEN, dtype=np.int8)

        self.action_space = spaces.Discrete(len(self.allowed))
        self.observation_space = spaces.Box(
            low  = -1,
            high = 26,
            shape = (6, 5, 2),
            dtype = np.int8,
        )

        # state
        self._answer: str | None = target
        self._eval_mode = eval_mode
        self._guess_index = 0
        self.alpha = alpha
        self.green_bonus = green_bonus

    def _obs(self) -> np.ndarray:            # helper
        return np.stack((self._letters, self._colours), axis=-1)     
    
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
        self._colours.fill(-1)
        self._letters.fill(PAD_TOKEN)
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
        self._already_guessed = set()

        return self._obs(), {}
    
    def step(self, action: int, alpha: float = 0.3):
        assert self._answer is not None, "call reset() first"
        guess = self.allowed[action]
        
        for j, ch in enumerate(guess):
            self._letters[self._guess_index, j] = LETTER2IDX[ch]

        # colour feedback & update board
        fb = evaluate_guess(guess, self._answer)
        self._colours[self._guess_index] = fb
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
            reward += self.alpha * (prev_H - new_H)

            # extra shaping – reward each green tile immediately
            reward += self.green_bonus * np.sum(fb == 2)

        if guess in self._already_guessed:
            reward -= 1.0
        else:
            self._already_guessed.add(guess)

        return self._obs(), reward, terminated, False, {
            "guess": guess,
            "answer": self._answer,
            "step": self._guess_index,
        }
    
    def render(self, mode="ansi"):
        if mode != "ansi":
            raise NotImplementedError
        symbols = { -1: "·", 0: "⬜", 1: "🟨", 2: "🟩" }
        lines = ["".join(symbols[v] for v in row) for row in self._colours]
        return "\n".join(lines)
    
    def close(self):
        pass

class WordleTokens(BaseFeaturesExtractor):
    def __init__(self, space: gym.spaces.Box, emb=16):
        super().__init__(space, features_dim=256)
        self.let_emb  = nn.Embedding(27, emb)      # 0-25 + <pad>
        self.col_emb  = nn.Embedding(4,  4)
        self.pos_emb  = nn.Parameter(torch.randn(30, emb+4))
        self.enc      = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=emb+4, nhead=4, dim_feedforward=128,
                batch_first=True, activation="gelu"
            ), num_layers=4
        )
        self.fc = nn.Sequential(
            nn.Linear(emb+4, 256), nn.ReLU(),
        )

    def forward(self, obs):
        letters  = obs[..., 0].long()          # (B,6,5) → LongTensor
        colours  = obs[..., 1].long().clamp(min=0)  # mask -1 → 0 for PAD

        x = torch.cat(
            (self.let_emb(letters), self.col_emb(colours)), dim=-1
        ).view(obs.size(0), 30, -1) + self.pos_emb         # (B,30,20)
        
        x = self.enc(x).mean(dim=1)                        # (B,20)
        return self.fc(x)