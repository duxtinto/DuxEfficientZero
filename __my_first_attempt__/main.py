import os
import random
import shutil
import logging.config

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter

from ArgumentParser import ArgumentParser
from config.AtariConfig import AtariConfig
from test import test
from train import train


def set_seed(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_results_dir(exp_path, args):
    # make the result directory
    os.makedirs(exp_path, exist_ok=True)
    if args.opr == 'train' and os.path.exists(exp_path) and os.listdir(exp_path):
        if not args.force:
            raise FileExistsError('{} is not empty. Please use --force to overwrite it'.format(exp_path))
        else:
            print('Warning, path exists! Rewriting...')
            shutil.rmtree(exp_path)
            os.makedirs(exp_path)
    log_path = os.path.join(exp_path, 'logs')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(exp_path, 'model'), exist_ok=True)
    return exp_path, log_path


def init_logger(base_path):
    # initialize the logger
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['train', 'test', 'train_test', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    args = ArgumentParser().parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'
    assert args.revisit_policy_search_rate is None or 0 <= args.revisit_policy_search_rate <= 1, \
        ' Revisit policy search rate should be in [0,1]'

    if args.opr == 'train':
        ray.init(
            num_gpus=args.num_gpus,
            num_cpus=args.num_cpus,
            object_store_memory=args.object_store_memory,
            local_mode=True
        )
    else:
        ray.init()

    assert ray.is_initialized() is True

    # seeding random iterators
    set_seed(args.seed)

    game_config = AtariConfig()

    # set config as per arguments
    exp_path = game_config.set_config(args)
    exp_path, log_base_path = make_results_dir(exp_path, args)

    # set-up logger
    init_logger(log_base_path)
    logging.getLogger('train').info('Path: {}'.format(exp_path))
    logging.getLogger('train').info('Param: {}'.format(game_config.get_hparams()))

    device = game_config.device

    try:
        # Train
        summary_writer = SummaryWriter(exp_path, flush_secs=10)
        if args.load_model and os.path.exists(args.model_path):
            model_path = args.model_path
        else:
            model_path = None
        model, weights = train(game_config, summary_writer, model_path)
        model.set_weights(weights)
        total_steps = game_config.training_steps + game_config.last_steps
        test_score, test_path = test(game_config, model.to(device), total_steps, game_config.test_episodes,
                                     device, render=False, save_video=args.save_video, final_test=True, use_pb=True)
        mean_score = test_score.mean()
        std_score = test_score.std()

        test_log = {
            'mean_score': mean_score,
            'std_score': std_score,
        }
        for key, val in test_log.items():
            summary_writer.add_scalar('train/{}'.format(key), np.mean(val), total_steps)

        test_msg = '#{:<10} Test Mean Score of {}: {:<10} (max: {:<10}, min:{:<10}, std: {:<10})' \
                   ''.format(total_steps, game_config.env_name, mean_score, test_score.max(), test_score.min(), std_score)
        logging.getLogger('train_test').info(test_msg)
        if args.save_video:
            logging.getLogger('train_test').info('Saving video in path: {}'.format(test_path))

        ray.shutdown()
    except Exception as e:
        logging.getLogger('root').error(e, exc_info=True)