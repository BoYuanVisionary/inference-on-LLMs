#!/usr/bin/env python3
import yaml
import itertools
from pathlib import Path
import copy

def generate_ablation_configs(base_config_path, ablation_params, output_dir):
    # Read base configuration
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations of ablation parameters
    param_names = list(ablation_params.keys())
    param_values = [ablation_params[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    
    # Generate config for each combination
    configs = []
    for combo in combinations:
        config = copy.deepcopy(base_config)
        config_name_parts = []
        
        # Update config with ablation values
        for param_name, value in zip(param_names, combo):
            # Handle nested parameters (e.g., "model.learning_rate")
            current = config
            parts = param_name.split('.')
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
            config_name_parts.append(f"{parts[-1]}_{value}")
        
        # Update config name
        config['config_name'] = f"{base_config['config_name']}_{'_'.join(config_name_parts)}"
        configs.append(config)
        
        # Save config
        output_path = Path(output_dir) / f"{config['config_name']}.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    return configs

if __name__ == "__main__":
    # Example usage
    ablation_params = {
        'samplingTree_temperature': [0.1, 0.3, 0.5],
        'beam_width': [2, 4, 8],
        'max_steps': [1, 2, 3],
        'threshold': [0.8, 0.9]
    }
    
    base_config_path = "configs/experiments_417/samplingTree_llama_4_1.yaml"
    output_dir = "configs/ablation_study"
    
    configs = generate_ablation_configs(base_config_path, ablation_params, output_dir) 