if __name__ == "__main__":
    import onnx
    import json

    import torch

    PLAYER_CONFIG = {
        # NOTE: "algo_type" defaults to POLICY in evaluation script!
        "algo_type": "QVALUE",
        "n_stack_frames": 1,
        "extra_state": ["depth"],
        "hud": "none",
        "crosshair": True,
        "screen_format": 0,
    }
    from doom_arena import VizdoomMPEnv
    
    env = VizdoomMPEnv(
        num_players=1,
        num_bots=4,
        bot_skill=0,
        doom_map="ROOM",  # NOTE simple, small map; other options: TRNM, TRNMBIG
        extra_state=PLAYER_CONFIG[
            "extra_state"
        ],  # see info about states at the beginning of 'Environment configuration' above
        episode_timeout=2000,
        n_stack_frames=PLAYER_CONFIG["n_stack_frames"],
        crosshair=PLAYER_CONFIG["crosshair"],
        hud=PLAYER_CONFIG["hud"],
        screen_format=PLAYER_CONFIG["screen_format"],
    )

    def onnx_dump(env, model, config, filename: str):
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

    import torch.nn as nn

    class DuelingDQN(nn.Module):
        def __init__(self, input_dim: int, action_space: int, hidden: int = 128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, 32, 8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),       nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),       nn.ReLU(),
                nn.Flatten(),
            )
            flattened_size = 9216
            self.value_stream = nn.Sequential(
                nn.Linear(flattened_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(flattened_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_space)
            )

        def forward(self, frame: torch.Tensor) -> torch.Tensor:
            features = self.encoder(frame)

            values = self.value_stream(features)
            advantages = self.advantage_stream(features)

            q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
            
            return q_values

    device = "cpu"
    DTYPE = torch.float32

    model = DuelingDQN(
        input_dim=4,
        action_space=env.action_space.n,
        hidden=128,
    ).to(device, dtype=DTYPE)
    model.load_state_dict(torch.load("model_ep100_2_bot_fine_tuned_2000_steps_DQN_Dueling.pt", map_location=device))

    onnx_dump(env, model, PLAYER_CONFIG, filename="ss.onnx")
