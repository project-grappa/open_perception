"""
This module sets up logging for the open_perception pipeline.
"""

import logging
import yaml
from pathlib import Path

class ConsoleHandler(logging.StreamHandler):
    def __init__(self, level):
        super().__init__()
        self.setLevel(level)

class CustomFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[90m',  # Darker gray
        'INFO': '\033[37m',   # Light gray
        'WARNING': '\033[33m', # Yellow
        'ERROR': '\033[31m',  # Red
        'RESET': '\033[0m'    # Reset color
    }

    def __init__(self, fmt):
        super().__init__(fmt)

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        record.msg = f"{log_color}{record.msg}{reset_color}"
        return super().format(record)

class FileHandler(logging.FileHandler):
    def __init__(self, filename, level):
        super().__init__(filename)
        self.setLevel(level)

DEFAULT_CONFIG = {
    'logging': {
        'level': 'ERROR',
        'handlers': {
            'console': {
                # 'level': 'ERROR',
                'formatter': 'default'
            },
            'file': {
                'filename': 'app.log',
                # 'level': 'ERROR',
                'formatter': 'default'
            }
        },
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    }
}
from open_perception.utils.config_loader import merge_dicts
class Logger:
    @staticmethod
    def get_logger(name: str, config: dict = None) -> logging.Logger:
        
        # Merge with default config
        if config is None:
            config = {}
        config = merge_dicts(DEFAULT_CONFIG, config)

        logger = logging.getLogger(name)
        logger.setLevel(config['logging']['level'])

        # Console handler
        console_config = config['logging']['handlers']['console']
        console_handler_level = console_config.get('level', config['logging']['level'])
        console_handler = ConsoleHandler(console_handler_level)
        console_formatter = CustomFormatter(config['logging']['formatters'][console_config['formatter']]['format'])
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        # File handler
        file_config = config['logging']['handlers']['file']
        file_handler_level = file_config.get('level', config['logging']['level'])
        file_handler = FileHandler(file_config['filename'], file_handler_level)
        file_formatter = CustomFormatter(config['logging']['formatters'][file_config['formatter']]['format'])
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger
