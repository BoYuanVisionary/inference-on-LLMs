import os
import yaml
from typing import Dict, Any
import wandb
from reasoning.tools.utils import seed_everything

def load_config(config_path: str) -> Dict[str, Any]:
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    # Load configuration from YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def apply_config(config: Dict[str, Any]):

    wandb.init(project=config["wandb_project"])
    # set the run name to the config name
    wandb.run.name = config["config_name"]
    seed_everything(config["seed"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in config["cuda_device_ids"])
    # save config to wandb
    wandb.config.update(config)

if __name__ == "__main__":
    config = load_config("../../configs/development.yaml")
    print(config)



