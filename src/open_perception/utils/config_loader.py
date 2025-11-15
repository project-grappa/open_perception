"""
config_loader.py

Provides functionality to load and merge YAML configuration files,
as well as optionally override values from environment variables.
"""

import os
import yaml

from datetime import datetime
import re
from easydict import EasyDict
from pprint import pprint


class Loader(yaml.SafeLoader):

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)
Loader.add_constructor('!include', Loader.include)

def load_yaml_file(filepath):
    """
    Loads a YAML file and returns it as a dict.
    Raises FileNotFoundError if the file does not exist.
    """
    with open(filepath, 'r') as f:
        data = yaml.load(f, Loader) or {}
    return data

def override_with_env(config):
    """
    Optional: Override config values with environment variables if needed.
    For example, if you have config like:
        config["communication"]["redis"]["host"] = "localhost"
    and you have an environment variable like REDIS_HOST=remotehost,
    you can map that variable to override config here.

    This is just a placeholder example. Adjust the logic to your naming scheme.
    """
    # Example environment variable overrides
    # Suppose you define that "REDIS_HOST" overrides "communication.redis.host"
    env_map = {
        "REDIS_HOST": ("communication", "redis", "host"),
        "REDIS_PORT": ("communication", "redis", "port"),
        "API_HOST": ("communication", "api_server", "host"),
        "API_PORT": ("communication", "api_server", "port"),
    }

    for env_var, path in env_map.items():
        if env_var in os.environ:
            # Navigate the config dict according to the path
            current = config
            for key in path[:-1]:
                if key not in current:
                    # Create nested dict if not present
                    current[key] = {}
                current = current[key]

            # Set the final value from the environment variable
            final_key = path[-1]
            env_value = os.environ[env_var]

            # Attempt to convert numeric strings to int, if relevant
            if env_value.isdigit():
                env_value = int(env_value)

            current[final_key] = env_value

    return config

def check_nested_attrs(obj, attrs):
    tmp = obj
    for attr in attrs:
        if not hasattr(tmp, attr):
            return False, None
        tmp = getattr(tmp, attr)
    return True, tmp

def parse_config_reference(entry, global_config, config):
    """
    Parse a configuration reference and return the resolved value.

    Args:
        entry (str): The configuration reference string (e.g., "key1.key2").
        global_config (dict): The global configuration dictionary.
        config (dict): The local configuration dictionary.

    Returns:
        The resolved value from the configuration or None if not found.
    """
    # get the keys to look up in the configs selecting the contents in between the curly braces and percent signs {%key%}
    keys = re.findall(r'\{%(.*?)%\}', entry)
    retrieved_config = {}
    for key in list(set(keys)):
        if key == "timestamp":
            # If the key is 'timestamp', we can directly use it
            retrieved_config[key] = datetime.now().strftime("%Y_%m_%d-%H.%M.%S")
            continue
        res, value = check_nested_attrs(config, key.split('.'))
        if res:
            # If the key exists in the global config, retrieve it
            retrieved_config[key] = value
            continue
        res, value = check_nested_attrs(config, key.split('.'))
        if res:
            # If the key exists in the local config, retrieve it
            retrieved_config[key] = value
            continue
        else:
            print(f"\033[93mWarning: Key '{key}' not found in config or global_config. Using original reference.\033[0m")
            retrieved_config[key] = f"{{%{key}%}}"  # If not found, keep the original reference

    for key, value in retrieved_config.items():
        entry = entry.replace("{%"+key+"%}", value)
    # Format the entry with the retrieved configuration values
    return entry

def resolve_linked_parameters(config, global_config=None, parent_key=""):
    """
    Recursively resolve linked parameters in the configuration.

    Args:
        config (dict or list): Configuration dictionary or list.
        global_config (dict): Global configuration for resolving parameters.
        parent_key (str): Parent key for nested dictionaries or index for lists.

    Returns:
        dict or list: Configuration with resolved linked parameters.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d-%H.%M.%S")
    if global_config is None:
        global_config = config
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                resolve_linked_parameters(value, global_config, parent_key=f"{parent_key}.{key}" if parent_key else key)
            elif isinstance(value, str) and "{%" in value and "%}" in value:
                config[key] = parse_config_reference(value, global_config, config)

    elif isinstance(config, list):
        for index, value in enumerate(config):
            if isinstance(value, (dict, list)):
                resolve_linked_parameters(value, global_config, parent_key=f"{parent_key}[{index}]")
            elif isinstance(value, str) and "{%" in value and "%}" in value:
                config[index] = parse_config_reference(value, global_config, config)

def load_config(config_path=None):
    """
    Load a YAML configuration file and optionally merge with defaults,
    environment variables, etc.
    """
    script_path = os.path.dirname(os.path.realpath(__file__))
    default_config_path = str(os.path.join(script_path, "../../../config/default.yaml"))
    # 1. Load the main config file
    default_config = load_yaml_file(default_config_path)

    if config_path is None:
        config_path = default_config_path

    #check if config path is the same as default config path using os.path.samefile
    if os.path.samefile(default_config_path, config_path):
        print("Using default config file")
        config = default_config
    else:
        print("Using custom config file")
        
        # 2. (Optional) Load a default.yaml first and merge if needed
        # For instance, if you always want to start from default.yaml
        # and override with user-specified config:
        config_overrides = load_yaml_file(config_path) if config_path else {}
        config = merge_dicts(default_config, config_overrides)

    # 3. Override with environment variables
    config = override_with_env(config)
    config = EasyDict(config)
    # Resolve linked parameters
    resolve_linked_parameters(config)

    return config

def merge_dicts(base, override):
    """
    Recursively merges 'override' dict into 'base' dict.
    Values from 'override' take precedence.
    """
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


if __name__ == "__main__":
    # Example usage
    config = load_config()
    pprint(config)
    # You can also specify a custom config path
    # custom_config = load_config("path/to/your/config.yaml")
    # print(custom_config)