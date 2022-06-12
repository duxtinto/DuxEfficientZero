import os
import time

import ray
import torch
import torch.optim as optim

from ReplayBuffer import ReplayBuffer
from storage.QueueStorage import QueueStorage
from storage.SharedStorage import SharedStorage
from workers.BatchWorker_CPU import BatchWorker_CPU
from workers.DataWorker import DataWorker
from test import _test


def _train(model, target_model, replay_buffer, shared_storage, batch_storage, config, summary_writer):
    """training loop
    Parameters
    ----------
    model: Any
        EfficientZero models
    target_model: Any
        EfficientZero models for reanalyzing
    replay_buffer: Any
        replay buffer
    shared_storage: Any
        model storage
    batch_storage: Any
        batch storage (queue)
    summary_writer: Any
        logging for tensorboard
    """
    # ----------------------------------------------------------------------------------
    model = model.to(config.device)
    target_model = target_model.to(config.device)

    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                          weight_decay=config.weight_decay)

    scaler = GradScaler()

    model.train()
    target_model.eval()
    # ----------------------------------------------------------------------------------
    # set augmentation tools
    if config.use_augmentation:
        config.set_transforms()

    # wait until collecting enough data to start
    while not (ray.get(replay_buffer.get_total_len.remote()) >= config.start_transitions):
        time.sleep(1)
        pass
    print('Begin training...')
    # set signals for other workers
    shared_storage.set_start_signal.remote()

    step_count = 0
    # Note: the interval of the current model and the target model is between x and 2x. (x = target_model_interval)
    # recent_weights is the param of the target model
    recent_weights = model.get_weights()

    # while loop
    while step_count < config.training_steps + config.last_steps:
        # remove data if the replay buffer is full. (more data settings)
        if step_count % 1000 == 0:
            replay_buffer.remove_to_fit.remote()

        # obtain a batch
        batch = batch_storage.pop()
        if batch is None:
            time.sleep(0.3)
            continue
        shared_storage.incr_counter.remote()
        lr = adjust_lr(config, optimizer, step_count)

        # update model for self-play
        if step_count % config.checkpoint_interval == 0:
            shared_storage.set_weights.remote(model.get_weights())

        # update model for reanalyzing
        if step_count % config.target_model_interval == 0:
            shared_storage.set_target_weights.remote(recent_weights)
            recent_weights = model.get_weights()

        if step_count % config.vis_interval == 0:
            vis_result = True
        else:
            vis_result = False

        if config.amp_type == 'torch_amp':
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result)
            scaler = log_data[3]
        else:
            log_data = update_weights(model, batch, optimizer, replay_buffer, config, scaler, vis_result)

        if step_count % config.log_interval == 0:
            _log(config, step_count, log_data[0:3], model, replay_buffer, lr, shared_storage, summary_writer,
                 vis_result)

        # The queue is empty.
        if step_count >= 100 and step_count % 50 == 0 and batch_storage.get_len() == 0:
            print('Warning: Batch Queue is empty (Require more batch actors Or batch actor fails).')

        step_count += 1

        # save models
        if step_count % config.save_ckpt_interval == 0:
            model_path = os.path.join(config.model_dir, 'model_{}.p'.format(step_count))
            torch.save(model.state_dict(), model_path)

    shared_storage.set_weights.remote(model.get_weights())
    time.sleep(30)
    return model.get_weights()


def train(config, summary_writer, model_path=None):
    """training process
    Parameters
    ----------
    summary_writer: Any
        logging for tensorboard
    model_path: str
        model path for resuming
        default: train from scratch
    """
    model = config.get_uniform_network()
    target_model = config.get_uniform_network()
    if model_path:
        print('resume model from path: ', model_path)
        weights = torch.load(model_path)

        model.load_state_dict(weights)
        target_model.load_state_dict(weights)

    storage = SharedStorage.remote(model, target_model)

    # prepare the batch and mctc context storage
    batch_storage = QueueStorage(15, 20)
    mcts_storage = QueueStorage(18, 25)
    replay_buffer = ReplayBuffer.remote(config=config)

    # other workers
    workers = []

    # reanalyze workers
    cpu_workers = [BatchWorker_CPU.remote(idx, replay_buffer, storage, batch_storage, mcts_storage, config) for idx in
                   range(config.cpu_actor)]
    workers += [cpu_worker.run.remote() for cpu_worker in cpu_workers]
    # gpu_workers = [BatchWorker_GPU.remote(idx, replay_buffer, storage, batch_storage, mcts_storage, config) for idx in range(config.gpu_actor)]
    # workers += [gpu_worker.run.remote() for gpu_worker in gpu_workers]

    # self-play workers
    data_workers = [DataWorker.remote(rank, replay_buffer, storage, config) for rank in range(0, config.num_actors)]
    workers += [worker.run.remote() for worker in data_workers]

    # test workers
    workers += [_test.remote(config, storage)]

    # training loop
    final_weights = _train(model, target_model, replay_buffer, storage, batch_storage, config, summary_writer)

    ray.wait(workers)
    print('Training over...')

    return model, final_weights
