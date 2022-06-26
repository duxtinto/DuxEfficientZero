import argparse
import logging.config
import os
import sys

import numpy as np
import ray
import torch
from opentelemetry import trace
from torch.utils.tensorboard import SummaryWriter

from __refactored__.ArgumentParser import ArgumentParser
from __refactored__.deploy.RayActors import RayActors
from __refactored__.tracing.opentelemetry import make_tracer_provider
from core.test import test
from core.train import train
from core.utils import init_logger, make_results_dir, set_seed
if __name__ == '__main__':
    tracer_provider = make_tracer_provider("smartfighters-efficientZero")
    trace.set_tracer_provider(tracer_provider)

    module_tracer = trace.get_tracer(__name__)
    with module_tracer.start_as_current_span("main") as main_span:
        with module_tracer.start_as_current_span("initialization"):
            # Lets gather arguments
            parser = ArgumentParser()

            # Process arguments
            args = parser.parse_args()
            args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'
            assert args.revisit_policy_search_rate is None or 0 <= args.revisit_policy_search_rate <= 1, \
                ' Revisit policy search rate should be in [0,1]'

            runtime_env = {
                'working_dir': '.',
                'pip': "./requirements.txt",
            }

            ray.init(
                runtime_env=runtime_env,
                namespace="dux-efficient-zero",
                logging_level=logging.DEBUG,
                _tracing_startup_hook='__refactored__.tracing.opentelemetry:setup_ray_tracing',
            )

            # seeding random iterators
            set_seed(args.seed)

            # import corresponding configuration , neural networks and envs
            if args.case == 'atari':
                from config.atari import game_config
            else:
                raise Exception('Invalid --case option')

            # set config as per arguments
            exp_path = game_config.set_config(args)
            exp_path, log_base_path = make_results_dir(exp_path, args)

            # set-up logger
            init_logger(log_base_path)
            logging.getLogger('train').info('Path: {}'.format(exp_path))
            logging.getLogger('train').info('Param: {}'.format(game_config.get_hparams()))

            device = game_config.device

        try:
            if args.opr == 'train':
                with module_tracer.start_as_current_span("train operation"):
                    summary_writer = SummaryWriter(exp_path, flush_secs=10)
                    if args.load_model and os.path.exists(args.model_path):
                        model_path = args.model_path
                    else:
                        model_path = None
                    model = game_config.get_uniform_network()
                    target_model = game_config.get_uniform_network()

                    with module_tracer.start_as_current_span("deploying actors to the cluster"):
                        ray_actors = RayActors(game_config, model, target_model)
                    with module_tracer.start_as_current_span("run ray remote workers"):
                        ray_actors.run_remote_workers()

                    model, weights = train(game_config, summary_writer, ray_actors, model, target_model, model_path)
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
            elif args.opr == 'test':
                with module_tracer.start_as_current_span("test operation"):
                    assert args.load_model
                    if args.model_path is None:
                        model_path = game_config.model_path
                    else:
                        model_path = args.model_path
                    assert os.path.exists(model_path), 'model not found at {}'.format(model_path)

                    model = game_config.get_uniform_network().to(device)
                    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
                    test_score, test_path = test(game_config, model, 0, args.test_episodes, device=device, render=args.render,
                                                 save_video=args.save_video, final_test=True, use_pb=True)
                    mean_score = test_score.mean()
                    std_score = test_score.std()
                    logging.getLogger('test').info('Test Mean Score: {} (max: {}, min: {})'.format(mean_score, test_score.max(), test_score.min()))
                    logging.getLogger('test').info('Test Std Score: {}'.format(std_score))
                    if args.save_video:
                        logging.getLogger('test').info('Saving video in path: {}'.format(test_path))
            else:
                raise Exception('Please select a valid operation(--opr) to be performed')
            ray.shutdown()
        except Exception as e:
            logging.getLogger('root').error(e, exc_info=True)
        except KeyboardInterrupt:
            print('Interrupted')
            ray_actors.kill_actors()
            tracer_provider.force_flush()
            ray.shutdown()
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)