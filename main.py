import argparse
import json
import sys

from trainer import SequencePredictorTrainer

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description='Train stock sequence model')
        parser.add_argument('--config_file', type=str,
                            help='path of the configuration file for trainer.')
        parser.add_argument('--IEX_token', type=str,
                            help='IEX authentication token')

        args = parser.parse_args()

        with open(args.config_file) as config_file:
            config = json.load(config_file)['trainer_config']

        trainer = SequencePredictorTrainer(config, args)

        trainer.train()

    except KeyboardInterrupt:
        print('Keyboard Interrupt, validating, saving checkpoint and exiting')
        trainer.run_validation(True)
        trainer.save_checkpoint()
        print('Done')
        sys.exit(0)
