import argparse
import logging.config
import os

import numpy as np
import ray
import torch
from opentelemetry.propagate import inject
from torch.utils.tensorboard import SummaryWriter
from opentelemetry import trace

from __refactored__.ArgumentParser import ArgumentParser
from __refactored__.tracing.opentelemetry import make_trace_provider
from core.test import test
from core.train import train
from core.utils import init_logger, make_results_dir, set_seed
if __name__ == '__main__':
    global_tracer = make_trace_provider("smartfighters-efficientZero")
    trace.set_tracer_provider(global_tracer)

    module_tracer = trace.get_tracer(__name__)
    with module_tracer.start_as_current_span("main") as main_span:
        with module_tracer.start_as_current_span("initialization") as initialization_span:
            with module_tracer.start_as_current_span("parsing arguments") as parsing_arguments_span:
                # Lets gather arguments
                parser = ArgumentParser()
                args = parser.parse_args()
                args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'
                assert args.revisit_policy_search_rate is None or 0 <= args.revisit_policy_search_rate <= 1, \
                    ' Revisit policy search rate should be in [0,1]'
                parsing_arguments_span.set_attribute('arguments', str(args))

            with module_tracer.start_as_current_span("initalizing ray") as ray_span:
                runtime_env = {
                    'working_dir': '.',
                }

                ray.init(
                    runtime_env=runtime_env,
                    namespace="dux-efficient-zero",
                    logging_level=logging.DEBUG)
                # if args.opr == 'train':
                #     ray.init(
                #         # 'ray://127.0.0.1:10001',
                #         # num_gpus=args.num_gpus,
                #         # num_cpus=args.num_cpus,
                #         # object_store_memory=args.object_store_memory,
                #         logging_level=logging.DEBUG,
                #         # include_dashboard=False,
                #         # local_mode=True
                #         # _tracing_startup_hook="__refactored__.tracing.opentelemetry:setup_ray_tracing",
                #     )
                # else:
                #     ray.init()

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
                with module_tracer.start_as_current_span("training") as training_span:
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
            elif args.opr == 'test':
                with module_tracer.start_as_current_span("training") as testing_span:
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
