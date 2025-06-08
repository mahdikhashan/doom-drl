from typing import Dict, Tuple

import numpy as np


class VizDoomReward:
    def __init__(self, num_players: int):
        self.num_players = num_players
        self.reset()

    def reset(self):
        self._step = 0
        self.cumulated_frag = np.zeros(self.num_players)

    def __call__(
        self,
        vizdoom_reward: float,
        game_var: Dict[str, float],
        game_var_old: Dict[str, float],
        player_id: int,
    ) -> Tuple:
        self._step += 1
        _ = player_id
        rwd_hit = 2.0 * (game_var["HITCOUNT"] - game_var_old["HITCOUNT"])
        rwd_hit_taken = -0.1 * (game_var["HITS_TAKEN"] - game_var_old["HITS_TAKEN"])
        rwd_frag = 100.0 * (game_var["FRAGCOUNT"] - game_var_old["FRAGCOUNT"])
        # rwd_health = (game_var["HEALTH"] - game_var_old["HEALTH"])
        # rwd_dead = -10.0 if (game_var["DEAD"] - game_var_old["DEAD"]) > 0 else 0.0

        return (
            # vizdoom_reward,
            rwd_hit,
            rwd_hit_taken,
            rwd_frag,
            # rwd_health,
            # rwd_dead,
        )