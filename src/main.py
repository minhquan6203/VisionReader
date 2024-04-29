import yaml
import argparse
import logging
from typing import Text
import transformers
from task.train import Training
from task.inference import Predict

def main(config_path: Text) -> None:
    transformers.logging.set_verbosity_error()
    logging.basicConfig(level=logging.INFO)
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    logging.info("Training started...")
    task_train=Training(config)
    task_train.training()
    logging.info("Training complete")
    
    logging.info('now evaluate on test data...')
    task_infer=Predict(config)
    task_infer.predict_submission()
    logging.info('task done!!!')
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    main(args.config)