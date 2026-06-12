from pathlib import Path

from prime_rl._device import current_device
from prime_rl.configs.trainer import LoRAConfig, WeightBroadcastConfig
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.filesystem import FileSystemWeightBroadcast
from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast


def setup_weight_broadcast(
    output_dir: Path, config: WeightBroadcastConfig, lora_config: LoRAConfig | None = None
) -> WeightBroadcast:
    if config.type == "nccl":
        return NCCLWeightBroadcast(output_dir, config, current_device())
    elif config.type == "filesystem":
        return FileSystemWeightBroadcast(output_dir, config, lora_config)
    else:
        raise ValueError(f"Invalid weight broadcast type: {config.type}")
