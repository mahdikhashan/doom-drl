import torch
import torch.nn as nn
from gym import Env



@torch.no_grad
def epsilon_greedy(
    env: Env,
    model: nn.Module,
    obs: torch.Tensor,
    epsilon: float,
    device: torch.device,
    dtype: torch.dtype,
):
    import random

    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        obs = obs.to(device, dtype=dtype).unsqueeze(0)
        return model(obs).argmax().item()


@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.995):
    from collections import OrderedDict

    """Exponential moving average model updates."""
    ema_params = OrderedDict(ema_model.named_parameters())
    if hasattr(model, "module"):
        model_params = OrderedDict(model.module.named_parameters())
    else:
        model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def onnx_dump(env, model, config, filename: str):
    import onnx
    import json
    # dummy state
    init_state = env.reset()[0].unsqueeze(0)

    # Export to ONNX
    torch.onnx.export(
        model.cpu(),
        args=init_state,
        f=filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    onnx_model = onnx.load(filename)

    meta = onnx_model.metadata_props.add()
    meta.key = "config"
    meta.value = json.dumps(config)

    onnx.save(onnx_model, filename)


if __name__ == "__main__":

    USE_GRAYSCALE = False  # ← flip to False for RGB

    PLAYER_CONFIG = {
        # NOTE: "algo_type" defaults to POLICY in evaluation script!
        "algo_type": "QVALUE",  # OPTIONAL, change to POLICY if using policy-based (eg PPO)
        "n_stack_frames": 1,
        "extra_state": ["depth"],
        "hud": "none",
        "crosshair": True,
        "screen_format": 8 if USE_GRAYSCALE else 0,
    }

    N_STACK_FRAMES = 1
    NUM_BOTS = 4
    EPISODE_TIMEOUT = 1000
    # TODO: model hyperparams
    GAMMA = 0.95
    EPISODES = 100
    BATCH_SIZE = 32
    REPLAY_BUFFER_SIZE = 10_000
    LEARNING_RATE = 1e-4
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.995
    N_EPOCHS = 50

    device = "cpu"
    DTYPE = torch.float32

    from doom_arena.reward import VizDoomReward
    from typing import Dict, Tuple

    class YourReward(VizDoomReward):
        def __init__(self, num_players: int):
            super().__init__(num_players)

        def __call__(
            self,
            vizdoom_reward: float,
            game_var: Dict[str, float],
            game_var_old: Dict[str, float],
            player_id: int,
        ) -> Tuple[float, float, float]:
            """
            Custom reward function used by both training and evaluation.
            *  +100  for every new frag
            *  +2    for every hit landed
            *  -0.1  for every hit taken
            """
            self._step += 1
            _ = vizdoom_reward, player_id  # unused

            rwd_hit = 2.0 * (game_var["HITCOUNT"] - game_var_old["HITCOUNT"])
            rwd_hit_taken = -0.1 * (game_var["HITS_TAKEN"] - game_var_old["HITS_TAKEN"])
            rwd_frag = 100.0 * (game_var["FRAGCOUNT"] - game_var_old["FRAGCOUNT"])

            return rwd_hit, rwd_hit_taken, rwd_frag


    class DQN(nn.Module):
        """
        Deep-Q Network template.

        Expected behaviour
        ------------------
        forward(frame)      # frame: (B, C, H, W)  →  Q-values: (B, num_actions)

        What to add / change
        --------------------
        • Replace the two `NotImplementedError` lines.
        • Build an encoder (Conv2d / Conv3d) + a head (MLP or duelling).
        • Feel free to use residual blocks from `agents/utils.py` or any design you like.
        """

        def __init__(self, input_dim: int, action_space: int, hidden: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),       nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),       nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 12 * 12, hidden), nn.ReLU(),
                nn.Linear(hidden, action_space),
            )

        def forward(self, frame: torch.Tensor) -> torch.Tensor:
            x = self.encoder(frame)
            x = self.head(x)
            return x


    from doom_arena import VizdoomMPEnv

    reward_fn = YourReward(num_players=1)

    env = VizdoomMPEnv(
        num_players=1,
        num_bots=NUM_BOTS,
        bot_skill=0,
        doom_map="ROOM",  # NOTE simple, small map; other options: TRNM, TRNMBIG
        extra_state=PLAYER_CONFIG[
            "extra_state"
        ],  # see info about states at the beginning of 'Environment configuration' above
        episode_timeout=EPISODE_TIMEOUT,
        n_stack_frames=PLAYER_CONFIG["n_stack_frames"],
        crosshair=PLAYER_CONFIG["crosshair"],
        hud=PLAYER_CONFIG["hud"],
        screen_format=PLAYER_CONFIG["screen_format"],
        reward_fn=reward_fn,
    )

    in_channels = env.observation_space.shape[0]  # 1 if grayscale, else 3/4
    model = DQN(
        input_dim=in_channels,
        action_space=env.action_space.n,
        hidden=64,  # change or ignore
    ).to(device, dtype=DTYPE)

    from copy import deepcopy
    model_tgt  = deepcopy(model).to(device)
    
    optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler  = torch.optim.lr_scheduler.ExponentialLR(optimizer, GAMMA)

    epsilon = 1.0

    import collections
    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER_SIZE)

    import random
    
    import torch.nn.functional as F
    import torch.optim as optim

    for episode in range(EPISODES):
        # TODO(mahdi): rename obs, what does it mean?
        obs = env.reset()[0]
        done, ep_return = False, 0.0
        model.eval()

        # roll-out phase
        while not done:
            act = epsilon_greedy(env, model, obs, epsilon, device, DTYPE)
            next_obs, rwd_raw, done, _ = env.step(act)

            # reward selection and scaling
            # TODO(mahdi): ...
            gv, gv_pre = env.envs[0].unwrapped._game_vars, env.envs[0].unwrapped._game_vars_pre
            a, b, c = reward_fn(gv, gv_pre)
            custom_rwd = a + b + c

            # work on replay buffer
            # TODO(mahdi): there could be a problem here
            # TODO(mahdi): how to do it better
            replay_buffer.append((obs, act, custom_rwd, next_obs[0], done))
            obs, ep_return = next_obs[0], ep_return + custom_rwd

        if len(replay_buffer) >= BATCH_SIZE:
            model.train()
            for _ in range(N_EPOCHS):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                s, a, r, s2, d = zip(*batch)

                s = torch.stack(s).to(device, dtype=DTYPE)
                s2 = torch.stack(s2).to(device, dtype=DTYPE)
                a = torch.tensor(a, device=device)
                r = torch.tensor(r, device=device, dtype=torch.float32)
                d = torch.tensor(d, device=device, dtype=torch.float32)

                q = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q2 = model_tgt(s2).max(1).values
                    tgt = r + GAMMA * q2 * (1 - d)
                loss = F.mse_loss(q, tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            update_ema(model_tgt, model)

        scheduler.step()
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        print(f"Ep {episode+1:03}: return {ep_return:6.1f}  |  ε {epsilon:.3f}")
