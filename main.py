import argparse
import json
import logging
import os
import pathlib
from datetime import datetime

from misc.reflections import make_class_from_module

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class Config:
    """Class to hold training configurations."""

    def __init__(self):
        parser = argparse.ArgumentParser(description='Train stock sequence model')

        parser.add_argument('--config_file', type=str, help='Path of the configuration file for trainer.')
        parser.add_argument('--IEX_token', type=str, help='IEX authentication token')

        args, extra_args = parser.parse_known_args()

        with open(args.config_file) as config_file:
            self.config = json.load(config_file)

        try:
            for arg in extra_args:
                assert '=' in arg, 'need key=value type args'
                keys, value = arg.split('=')
                keys = keys.split('.')
                module = self.config
                for k in keys[:-1]:
                    module = module[k]
                module[keys[-1]] = type(module[keys[-1]])(value)
        except KeyError as key_err:
            logging.warn('Only existing keys can be replaced with additional args, ignoring: ' + str(key_err))

        if args.IEX_token is not None:
            os.environ['IEX_TOKEN'] = args.IEX_token

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value
        return self.config

    def make_module(self, config_module, other_args=None, module_index=None):
        """Make a module in the format we have in config, if not return None."""
        if config_module not in self.config:
            return None

        if module_index is None:
            full_class_name = self.config[config_module]['name']
            args = self.config[config_module]['args']
        else:
            full_class_name = self.config[config_module][module_index]['name']
            args = self.config[config_module][module_index]['args']

        class_type = make_class_from_module(full_class_name)

        if other_args:
            args = dict(list(other_args.items()) + list(args.items()))
        return class_type(**args)

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


def check_date_overlap(t_range, v_range):
    t_start, t_end = t_range
    v_start, v_end = v_range

    start, end = max(t_start, v_start), min(t_end, v_end)
    delta = (end - start).total_seconds() + 1
    if delta > 0:
        raise ValueError("There's an overlap in training and validation dataset")


def get_exp_detail_str(config):
    """Get a summary for config using dataset_common and model params."""
    common = config['dataset_common']['args']
    return config['model']['args']['recurrent_type'] + '-' + common['output_features'][0] + '-' + \
        str(common['input_seq_len']) + '-' + str(common['output_seq_len']) + '-' + common['frequency']


def main():
    """Main trainer function."""

    now_str = datetime.now().strftime('%b-%d-%H-%M-%S')

    config = Config()

    config['logging_dir'] = os.path.join(config['logging_dir'], now_str + '-' + get_exp_detail_str(config))
    pathlib.Path(config['logging_dir']).mkdir(parents=True, exist_ok=True)
    make_logger(os.path.join(config['logging_dir'], config['exp_name'] + '.log'))

    config.log_self()

    dataset_common = config['dataset_common']['args']

    if isinstance(config.config['train_dataset'], list):
        assert 'dataset_mixer' in config.config.keys(), '"dataset_mixer" needed in config for list of train_datasets'
        n = len(config['train_dataset'])
        for dataset in config.config['train_dataset']:
            for attr in ["input_features", "output_features", "input_seq_len", "output_seq_len"]:
                assert attr not in dataset['args'], f'attribute {attr} can only be dfined in dataset_common'

        datasets = [config.make_module('train_dataset', dataset_common, i) for i in range(n)]
        train_dataset = config.make_module('dataset_mixer', {'datasets': datasets})
    else:
        train_dataset = config.make_module('train_dataset', dataset_common)

    valdn_dataset = config.make_module('valdn_dataset', dataset_common)

    model = config.make_module('model', {'input_n_features': len(dataset_common['input_features']),
                                         'output_n_features': len(dataset_common['output_features'])})

    optimizer = config.make_module('optimizer', {'params': model.parameters()})

    lr_scheduler = config.make_module('lr_scheduler', {'optimizer': optimizer})

    loss = config.make_module('loss')

    trainer = config.make_module('trainer', {'train_dataset': train_dataset,
                                             'valdn_dataset': valdn_dataset,
                                             'model': model,
                                             'optimizer': optimizer,
                                             'lr_scheduler': lr_scheduler,
                                             'loss': loss,
                                             'logging_dir': config['logging_dir'],
                                             **dataset_common})

    try:
        trainer.train()
    except KeyboardInterrupt:
        logging.debug('Keyboard interrupt detected, validating and exiting.')
        trainer.run_validation(True, True)


if __name__ == '__main__':
    main()
