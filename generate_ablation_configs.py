#!/usr/bin/env python3
import yaml
import json
import itertools
from pathlib import Path
import copy
import os

def generate_ablation_configs(base_config_path, ablation_params, output_dir, use_json=False):
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
            
            # Set the value (now supporting nested lists)
            current[parts[-1]] = value
                
            # Format the value for the config name
            if isinstance(value, list):
                # Handle nested lists by flattening them for the filename
                flat_values = []
                def flatten(lst):
                    for item in lst:
                        if isinstance(item, list):
                            flatten(item)
                        else:
                            flat_values.append(str(item))
                flatten(value)
                formatted_value = '_'.join(flat_values)
            else:
                formatted_value = str(value)
            config_name_parts.append(f"{parts[-1]}_{formatted_value}")
        
        # Update config name
        config['config_name'] = f"{base_config['config_name']}_{'_'.join(config_name_parts)}"
        configs.append(config)
        
        # Save config
        if use_json:
            output_path = Path(output_dir) / f"{config['config_name']}.json"
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            output_path = Path(output_dir) / f"{config['config_name']}.yaml"
            with open(output_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    return configs

if __name__ == "__main__":
    # Example usage with nested lists
    ablation_params = {
        'cuda_device_ids': [[0,1], [2,3], [0,1,2,3]],  # Example of list parameters
        'max_steps': [1, 2, 3],
        'beam_search_params': [
            [[4, 0.1], [8, 0.2]],  # Example of nested list parameters
            [[2, 0.3], [4, 0.4]],
        ]
    }
    
    base_config_path = "configs/experiments_417/Wmajority_llama_16_0.yaml"
    output_dir = "configs/ablation_nested"
    
    # Generate both YAML and JSON versions
    configs_yaml = generate_ablation_configs(base_config_path, ablation_params, output_dir, use_json=False)
    configs_json = generate_ablation_configs(base_config_path, ablation_params, output_dir + "_json", use_json=True) 