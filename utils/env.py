from utils.registry import SimpleEventRegistry
from utils.logger import get_logger
import torch
import os
import time
import tqdm

logger = get_logger(__file__)

class TrainEnv(object):
    def __init__(self, model, optimizer, config, device):
        self._model = model.to(device)
        self._optimizer = optimizer
        self._config = config
        self._device = device

        self._train_event_manager = SimpleEventRegistry({
            'epoch:before', 'epoch:after',
            'step:before', 'step:after',
            'forward:before', 'forward:after',
            'backward:before', 'backward:after',
        })

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def config(self):
        return self._config

    def event_register(self, name, callback):
        logger.info('Register trainer event: name={}, callback={}.'.format(name, callback.__module__ + '.' + callback.__name__))
        self._train_event_manager.register(name, callback)

    def trigger_event(self, name, *args, **kwargs):
        self._train_event_manager.trigger(name, *args, **kwargs)

    def save_checkpoint(self, filename, extra=None):

        checkpoint = {
            'net': self._model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'config': self.config,
            'extra': extra
        }

        try:
            torch.save(checkpoint, filename)
            logger.info('Checkpoint saved: "{}".'.format(filename))
        except Exception:
            logger.exception('Error occurred when dump checkpoint "{}".'.format(filename))

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            try:
                checkpoint = torch.load(filename)
                self._model.load_state_dict(checkpoint['net'])
                self._optimizer.load_state_dict(checkpoint['optimizer'])
                self._config = checkpoint['config']
                logger.critical('Checkpoint loaded: {}.'.format(filename))
                return checkpoint['extra']
            except Exception:
                logger.exception('Error occurred when load checkpoint "{}".'.format(filename))
        else:
            logger.exception('No checkpoint found at specified position: "{}".'.format(filename))

    def set_learning_rate(self, lr):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def decay_learning_rate(self, decay):
        for param_group in self._optimizer.param_groups:
            param_group['lr'] *= decay

    def step(self, feed_dict, grad_clip=0., measure_time=False):

        assert self._model.training, 'Step a evaluation-mode model.'
        extra = dict()

        self.trigger_event('step:before', self)

        if measure_time:
            end_time = time.time()

        self.trigger_event('forward:before', self, feed_dict)
        loss, output_dict, monitors = self._model(feed_dict)
        self.trigger_event('forward:after', self, feed_dict, loss, monitors, output_dict)

        if measure_time:
            extra['time/forward'] = time.time() - end_time
            end_time = time.time()

        self._optimizer.zero_grad()
        self.trigger_event('backward:before', self, feed_dict, loss, monitors, output_dict)
        if loss.requires_grad:
            loss.backward()
            if grad_clip > 0:
                from torch.nn.utils.clip_grad import clip_grad_norm_
                clip_grad_norm_(self.model.parameters(), grad_clip)

        if measure_time:
            extra['time/backward'] = time.time() - end_time
            end_time = time.time()

        self.trigger_event('backward:after', self, feed_dict, loss, monitors, output_dict)
        if loss.requires_grad:
            self._optimizer.step()

        if measure_time:
            extra['time/optimize'] = time.time() - end_time

        self.trigger_event('step:after', self)

        return loss, output_dict, monitors, extra

    def evaluate(self, feed_dict):
        assert not self._model.training, 'Evaluating a training-mode model.'
        with torch.no_grad():
            loss, output_dict, monitors = self._model(feed_dict)
        return loss, output_dict, monitors

    def train_epoch(self, train_loader, epoch, meters):
        self._model.train()
        pbar = tqdm.tqdm(train_loader)
        for feed_dict in pbar:
            self.optimizer.zero_grad()
            feed_dict = {k: v.to(self._device) if 'to' in v.__dir__() else v for k, v in feed_dict.items()}
            loss, output_dict, monitors, extra = self.step(feed_dict)
            self.optimizer.step()
            n = list(feed_dict.values())[0].size(0)
            meters.update(loss=loss, n=n)
            meters.update(monitors, n=n)
            meters.update(extra)
            pbar.set_description(desc=f'Training {meters.epoch_common_format(epoch)}', refresh=True)

    def valid_epoch(self, valid_loader, epoch, evaluate_meter):
        self._model.eval()
        pbar = tqdm.tqdm(valid_loader)
        for feed_dict in pbar:
            feed_dict = {k: v.to(self._device) if 'to' in v.__dir__() else v for k, v in feed_dict.items()}
            loss, output_dict, monitors = self.evaluate(feed_dict)
            evaluate_meter.update(loss, feed_dict, output_dict)
            pbar.set_description(desc=f'Validating {epoch}', refresh=True)
        evaluate_meter.evaluate()