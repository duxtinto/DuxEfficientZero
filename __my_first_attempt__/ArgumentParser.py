import argparse
import os


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ArgumentParser, self).__init__(*args, **kwargs)

        self.description = 'EfficientZero'

        self.add_argument('--env', required=True, help='Name of the environment')
        self.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                            help="Directory Path to store results (default: %(default)s)")
        self.add_argument('--case', required=True, choices=['atari'],
                            help="It's used for switching between different domains(default: %(default)s)")
        self.add_argument('--opr', required=True, choices=['train', 'test'])
        self.add_argument('--amp_type', required=True, choices=['torch_amp', 'none'],
                            help='choose automated mixed precision type')
        self.add_argument('--no_cuda', action='store_true', default=False,
                            help='no cuda usage (default: %(default)s)')
        self.add_argument('--debug', action='store_true', default=False,
                            help='If enabled, logs additional values  '
                                 '(gradients, target value, reward distribution, etc.) (default: %(default)s)')
        self.add_argument('--render', action='store_true', default=False,
                            help='Renders the environment (default: %(default)s)')
        self.add_argument('--save_video', action='store_true', default=False, help='save video in test.')
        self.add_argument('--force', action='store_true', default=False,
                            help='Overrides past results (default: %(default)s)')
        self.add_argument('--cpu_actor', type=int, default=14, help='batch cpu actor')
        self.add_argument('--gpu_actor', type=int, default=20, help='batch bpu actor')
        self.add_argument('--p_mcts_num', type=int, default=8, help='number of parallel mcts')
        self.add_argument('--seed', type=int, default=0, help='seed (default: %(default)s)')
        self.add_argument('--num_gpus', type=int, default=4, help='gpus available')
        self.add_argument('--num_cpus', type=int, default=80, help='cpus available')
        self.add_argument('--revisit_policy_search_rate', type=float, default=0.99,
                            help='Rate at which target policy is re-estimated (default: %(default)s)')
        self.add_argument('--use_root_value', action='store_true', default=False,
                            help='choose to use root value in reanalyzing')
        self.add_argument('--use_priority', action='store_true', default=False,
                            help='Uses priority for data sampling in replay buffer. '
                                 'Also, priority for new data is calculated based on loss (default: False)')
        self.add_argument('--use_max_priority', action='store_true', default=False, help='max priority')
        self.add_argument('--test_episodes', type=int, default=10,
                            help='Evaluation episode count (default: %(default)s)')
        self.add_argument('--use_augmentation', action='store_true', default=True, help='use augmentation')
        self.add_argument('--augmentation', type=str, default=['shift', 'intensity'], nargs='+',
                            choices=['none', 'rrc', 'affine', 'crop', 'blur', 'shift', 'intensity'],
                            help='Style of augmentation')
        self.add_argument('--info', type=str, default='none', help='debug string')
        self.add_argument('--load_model', action='store_true', default=False, help='choose to load model')
        self.add_argument('--model_path', type=str, default='./results/test_model.p', help='load model path')
        self.add_argument('--object_store_memory', type=int, default=150 * 1024 * 1024 * 1024,
                            help='object store memory')
