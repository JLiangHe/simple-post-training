
import yaml
from pathlib import Path
import re

class AttrDict(dict):
    """A dictionary that allows for attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, attr):
        # Avoid recursion error with special attributes
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError
        try:
            value = self[attr]
            if isinstance(value, dict):
                return AttrDict(value)
            return value
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def from_dict(cls, d):
        """Recursively convert a dictionary to an AttrDict."""
        if not isinstance(d, dict):
            return d
        return cls({k: cls.from_dict(v) for k, v in d.items()})

def load_config():
    """
    Loads all YAML configuration files from the 'configs' directory,
    and makes them accessible as attributes on a single config object.
    """
    config_root = AttrDict()
    config_dir = Path(__file__).parent.parent / 'configs'

    if not config_dir.is_dir():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

    for config_file in config_dir.glob('*.yaml'):
        # Derive the attribute name from the filename.
        base_name = config_file.stem
        attr_name = re.sub(r'_configs?$', '', base_name)

        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
            if data:
                config_root[attr_name] = AttrDict.from_dict(data)

    return config_root
