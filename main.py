import argparse
import importlib
import json
import logging
import os
import pathlib
import sys
from datetime import datetime

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class Config:
    """Class to hold training configurations."""

    def __init__(self):
        parser = argparse.ArgumentParser(description='Train stock sequence model')

        parser.add_argument('--config_file', type=str, help='Path of the configuration file for trainer.')
        parser.add_argument('--IEX_token', type=str, help='IEX authentication token')

        args = parser.parse_args()

        with open(args.config_file) as config_file:
            self.config = json.load(config_file)

        if args.IEX_token is not None:
            os.environ['IEX_TOKEN'] = args.IEX_token

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value
        return self.config

    def make_module(self, config_module, other_args=None):
        """Make a module in the format we have in config."""
        full_class_name = self.config[config_module]['name']

        assert '.' in full_class_name, 'Passed int class name is not fully qualified'
        names = full_class_name.split('.')
        class_name = names[-1]
        module_name = '.'.join(names[0:-1])

        module = importlib.import_module(module_name)
        class_obj = getattr(module, class_name)

        args = self.config[config_module]['args']
        if other_args:
            args.update(other_args)
        return class_obj(**args)

    def log_self(self):
        with open(os.path.join(self.config['logging_dir'], 'config.json'), 'w') as file_:
            json.dump(self.config, file_, indent=2)


def make_logger(file_path: str):
    logging.basicConfig(filename=file_path,
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def main():
    """Main trainer function."""

    now_str = datetime.now().strftime('%b-%d-%H-%M-%S')

    config = Config()

    config['logging_dir'] = os.path.join(config['logging_dir'], now_str)
    pathlib.Path(config['logging_dir']).mkdir(parents=True, exist_ok=True)
    make_logger(os.path.join(config['logging_dir'], config['exp_name'] + '.log'))

    config.log_self()

    dataset_common = config['dataset_common']['args']
    train_dataset = config.make_module('train_dataset', dataset_common)
    valdn_dataset = config.make_module('valdn_dataset', dataset_common)

    model = config.make_module('model', {'input_n_features': len(dataset_common['input_features']),
                                         'output_n_features' : len(dataset_common['output_features'])})

    optimizer = config.make_module('optimizer', {'params': model.parameters()})

    lr_scheduler = config.make_module('lr_scheduler', {'optimizer': optimizer})

    loss = config.make_module('loss')

    trainer = config.make_module('trainer', {'train_dataset': train_dataset,
                                             'valdn_dataset': valdn_dataset,
                                             'model': model,
                                             'optimizer': optimizer,
                                             'lr_scheduler': lr_scheduler,
                                             'loss' : loss,
                                             'logging_dir': config['logging_dir'],
                                             **dataset_common})

    trainer.train()

if __name__ == '__main__':
    main()
