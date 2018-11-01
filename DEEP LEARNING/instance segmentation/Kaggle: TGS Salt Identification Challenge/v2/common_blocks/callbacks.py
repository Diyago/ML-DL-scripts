from functools import partial
import os
from datetime import datetime, timedelta

import numpy as np
import torch
import neptune
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from tempfile import TemporaryDirectory

from steppy.base import Step, IdentityOperation
from steppy.adapter import Adapter, E
from toolkit.pytorch_transformers.utils import Averager, persist_torch_model
from toolkit.pytorch_transformers.validation import score_model

from .utils import get_logger, sigmoid, softmax, make_apply_transformer, read_masks, get_list_of_image_predictions
from .metrics import intersection_over_union, intersection_over_union_thresholds
from .postprocessing import crop_image, resize_image, binarize

logger = get_logger()

Y_COLUMN = 'file_path_mask'
ORIGINAL_SIZE = (101, 101)
THRESHOLD = 0.5


class Callback:
    def __init__(self):
        self.epoch_id = None
        self.batch_id = None

        self.model = None
        self.optimizer = None
        self.loss_function = None
        self.output_names = None
        self.validation_datagen = None
        self.lr_scheduler = None

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.output_names = transformer.output_names
        self.validation_datagen = validation_datagen
        self.transformer = transformer

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0

    def on_train_end(self, *args, **kwargs):
        pass

    def on_epoch_begin(self, *args, **kwargs):
        pass

    def on_epoch_end(self, *args, **kwargs):
        self.epoch_id += 1

    def training_break(self, *args, **kwargs):
        return False

    def on_batch_begin(self, *args, **kwargs):
        pass

    def on_batch_end(self, *args, **kwargs):
        self.batch_id += 1

    def get_validation_loss(self):
        if self.epoch_id not in self.transformer.validation_loss.keys():
            self.transformer.validation_loss[self.epoch_id] = score_model(self.model, self.loss_function,
                                                                          self.validation_datagen)
        return self.transformer.validation_loss[self.epoch_id]


class CallbackList:
    def __init__(self, callbacks=None):
        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, Callback):
            self.callbacks = [callbacks]
        else:
            self.callbacks = callbacks

    def __len__(self):
        return len(self.callbacks)

    def set_params(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.set_params(*args, **kwargs)

    def on_train_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_begin(*args, **kwargs)

    def on_train_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_train_end(*args, **kwargs)

    def on_epoch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_begin(*args, **kwargs)

    def on_epoch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(*args, **kwargs)

    def training_break(self, *args, **kwargs):
        callback_out = [callback.training_break(*args, **kwargs) for callback in self.callbacks]
        return any(callback_out)

    def on_batch_begin(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_begin(*args, **kwargs)

    def on_batch_end(self, *args, **kwargs):
        for callback in self.callbacks:
            callback.on_batch_end(*args, **kwargs)


class TrainingMonitor(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        self.epoch_loss_averagers = {}
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averagers = {}
        self.epoch_id = 0
        self.batch_id = 0

    def on_epoch_end(self, *args, **kwargs):
        for name, averager in self.epoch_loss_averagers.items():
            epoch_avg_loss = averager.value
            averager.reset()
            if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
                logger.info('epoch {0} {1}:     {2:.5f}'.format(self.epoch_id, name, epoch_avg_loss))
        self.epoch_id += 1

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]
            if name in self.epoch_loss_averagers.keys():
                self.epoch_loss_averagers[name].send(loss)
            else:
                self.epoch_loss_averagers[name] = Averager()
                self.epoch_loss_averagers[name].send(loss)

            if self.batch_every and ((self.batch_id % self.batch_every) == 0):
                logger.info('epoch {0} batch {1} {2}:     {3:.5f}'.format(self.epoch_id, self.batch_id, name, loss))
        self.batch_id += 1


class ExponentialLRScheduler(Callback):
    def __init__(self, gamma, epoch_every=1, batch_every=None):
        super().__init__()
        self.gamma = gamma
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

    def set_params(self, transformer, validation_datagen, *args, **kwargs):
        self.validation_datagen = validation_datagen
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.lr_scheduler = ExponentialLR(self.optimizer, self.gamma, last_epoch=-1)

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        logger.info('initial lr: {0}'.format(self.optimizer.state_dict()['param_groups'][0]['initial_lr']))

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and (((self.epoch_id + 1) % self.epoch_every) == 0):
            self.lr_scheduler.step()
            logger.info('epoch {0} current lr: {1}'.format(self.epoch_id + 1,
                                                           self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.epoch_id += 1

    def on_batch_end(self, *args, **kwargs):
        if self.batch_every and ((self.batch_id % self.batch_every) == 0):
            self.lr_scheduler.step()
            logger.info('epoch {0} batch {1} current lr: {2}'.format(
                self.epoch_id + 1, self.batch_id + 1, self.optimizer.state_dict()['param_groups'][0]['lr']))
        self.batch_id += 1


class ExperimentTiming(Callback):
    def __init__(self, epoch_every=None, batch_every=None):
        super().__init__()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every
        self.batch_start = None
        self.epoch_start = None
        self.current_sum = None
        self.current_mean = None

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        logger.info('starting training...')

    def on_train_end(self, *args, **kwargs):
        logger.info('training finished')

    def on_epoch_begin(self, *args, **kwargs):
        if self.epoch_id > 0:
            epoch_time = datetime.now() - self.epoch_start
            if self.epoch_every:
                if (self.epoch_id % self.epoch_every) == 0:
                    logger.info('epoch {0} time {1}'.format(self.epoch_id - 1, str(epoch_time)[:-7]))
        self.epoch_start = datetime.now()
        self.current_sum = timedelta()
        self.current_mean = timedelta()
        logger.info('epoch {0} ...'.format(self.epoch_id))

    def on_batch_begin(self, *args, **kwargs):
        if self.batch_id > 0:
            current_delta = datetime.now() - self.batch_start
            self.current_sum += current_delta
            self.current_mean = self.current_sum / self.batch_id
        if self.batch_every:
            if self.batch_id > 0 and (((self.batch_id - 1) % self.batch_every) == 0):
                logger.info('epoch {0} average batch time: {1}'.format(self.epoch_id, str(self.current_mean)[:-5]))
        if self.batch_every:
            if self.batch_id == 0 or self.batch_id % self.batch_every == 0:
                logger.info('epoch {0} batch {1} ...'.format(self.epoch_id, self.batch_id))
        self.batch_start = datetime.now()


class NeptuneMonitor(Callback):
    def __init__(self, image_nr, image_resize, model_name):
        super().__init__()
        self.model_name = model_name
        self.ctx = neptune.Context()
        self.epoch_loss_averager = Averager()
        self.image_nr = image_nr
        self.image_resize = image_resize

    def on_train_begin(self, *args, **kwargs):
        self.epoch_loss_averagers = {}
        self.epoch_id = 0
        self.batch_id = 0

    def on_batch_end(self, metrics, *args, **kwargs):
        for name, loss in metrics.items():
            loss = loss.data.cpu().numpy()[0]

            if name in self.epoch_loss_averagers.keys():
                self.epoch_loss_averagers[name].send(loss)
            else:
                self.epoch_loss_averagers[name] = Averager()
                self.epoch_loss_averagers[name].send(loss)

            self.ctx.channel_send('{} batch {} loss'.format(self.model_name, name), x=self.batch_id, y=loss)

        self.batch_id += 1

    def on_epoch_end(self, *args, **kwargs):
        self._send_numeric_channels()
        self.epoch_id += 1

    def _send_numeric_channels(self, *args, **kwargs):
        for name, averager in self.epoch_loss_averagers.items():
            epoch_avg_loss = averager.value
            averager.reset()
            self.ctx.channel_send('{} epoch {} loss'.format(self.model_name, name), x=self.epoch_id, y=epoch_avg_loss)

        self.model.eval()
        val_loss = self.get_validation_loss()
        self.model.train()
        for name, loss in val_loss.items():
            loss = loss.data.cpu().numpy()[0]
            self.ctx.channel_send('{} epoch_val {} loss'.format(self.model_name, name), x=self.epoch_id, y=loss)


class ValidationMonitor(Callback):
    def __init__(self, data_dir, loader_mode, epoch_every=None, batch_every=None):
        super().__init__()
        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every
        if batch_every == 0:
            self.batch_every = False
        else:
            self.batch_every = batch_every

        self.data_dir = data_dir
        self.validation_pipeline = postprocessing_pipeline_simplified
        self.loader_mode = loader_mode
        self.meta_valid = None
        self.y_true = None
        self.activation_func = None

    def set_params(self, transformer, validation_datagen, meta_valid=None, *args, **kwargs):
        self.model = transformer.model
        self.optimizer = transformer.optimizer
        self.loss_function = transformer.loss_function
        self.output_names = transformer.output_names
        self.validation_datagen = validation_datagen
        self.meta_valid = meta_valid
        self.y_true = read_masks(self.meta_valid[Y_COLUMN].values)
        self.activation_func = transformer.activation_func
        self.transformer = transformer

    def get_validation_loss(self):
        return self._get_validation_loss()

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = self.get_validation_loss()
            self.model.train()
            for name, loss in val_loss.items():
                loss = loss.data.cpu().numpy()[0]
                logger.info('epoch {0} validation {1}:     {2:.5f}'.format(self.epoch_id, name, loss))
        self.epoch_id += 1

    def _get_validation_loss(self):
        output, epoch_loss = self._transform()
        y_pred = self._generate_prediction(output)

        logger.info('Calculating IOU and IOUT Scores')
        iou_score = intersection_over_union(self.y_true, y_pred)
        iout_score = intersection_over_union_thresholds(self.y_true, y_pred)
        logger.info('IOU score on validation is {}'.format(iou_score))
        logger.info('IOUT score on validation is {}'.format(iout_score))

        if not self.transformer.validation_loss:
            self.transformer.validation_loss = {}
        self.transformer.validation_loss.setdefault(self.epoch_id, {'sum': epoch_loss,
                                                                    'iou': Variable(torch.Tensor([iou_score])),
                                                                    'iout': Variable(torch.Tensor([iout_score]))})
        return self.transformer.validation_loss[self.epoch_id]

    def _transform(self):
        self.model.eval()
        batch_gen, steps = self.validation_datagen
        partial_batch_losses = []
        outputs = {}
        for batch_id, data in enumerate(batch_gen):
            X = data[0]
            targets_tensors = data[1:]

            if torch.cuda.is_available():
                X = Variable(X, volatile=True).cuda()
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True).cuda())
            else:
                X = Variable(X, volatile=True)
                targets_var = []
                for target_tensor in targets_tensors:
                    targets_var.append(Variable(target_tensor, volatile=True))

            outputs_batch = self.model(X)
            if len(self.output_names) == 1:
                for (name, loss_function_one, weight), target in zip(self.loss_function, targets_var):
                    loss_sum = loss_function_one(outputs_batch, target) * weight
                outputs.setdefault(self.output_names[0], []).append(outputs_batch.data.cpu().numpy())
            else:
                batch_losses = []
                for (name, loss_function_one, weight), output, target in zip(self.loss_function, outputs_batch,
                                                                             targets_var):
                    loss = loss_function_one(output, target) * weight
                    batch_losses.append(loss)
                    partial_batch_losses.setdefault(name, []).append(loss)
                    output_ = output.data.cpu().numpy()
                    outputs.setdefault(name, []).append(output_)
                loss_sum = sum(batch_losses)
            partial_batch_losses.append(loss_sum)
            if batch_id == steps:
                break
        self.model.train()
        average_losses = sum(partial_batch_losses) / steps
        outputs = {'{}_prediction'.format(name): get_list_of_image_predictions(outputs_) for name, outputs_ in
                   outputs.items()}
        for name, prediction in outputs.items():
            if self.activation_func == 'softmax':
                outputs[name] = [softmax(single_prediction, axis=0) for single_prediction in prediction]
            elif self.activation_func == 'sigmoid':
                outputs[name] = [sigmoid(np.squeeze(mask)) for mask in prediction]
            else:
                raise Exception('Only softmax and sigmoid activations are allowed')

        return outputs, average_losses

    def _generate_prediction(self, outputs):
        data = {'callback_input': {'meta': self.meta_valid,
                                   'meta_valid': None,
                                   },
                'unet_output': {**outputs}
                }
        with TemporaryDirectory() as cache_dirpath:
            pipeline = self.validation_pipeline(cache_dirpath, self.loader_mode)
            output = pipeline.transform(data)
        y_pred = output['y_pred']
        return y_pred


class ModelCheckpoint(Callback):
    def __init__(self, filepath, metric_name='sum', epoch_every=1, minimize=True):
        self.filepath = filepath
        self.minimize = minimize
        self.best_score = None

        if epoch_every == 0:
            self.epoch_every = False
        else:
            self.epoch_every = epoch_every

        self.metric_name = metric_name

    def on_train_begin(self, *args, **kwargs):
        self.epoch_id = 0
        self.batch_id = 0
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def on_epoch_end(self, *args, **kwargs):
        if self.epoch_every and ((self.epoch_id % self.epoch_every) == 0):
            self.model.eval()
            val_loss = self.get_validation_loss()
            loss_sum = val_loss[self.metric_name]
            loss_sum = loss_sum.data.cpu().numpy()[0]

            self.model.train()

            if self.best_score is None:
                self.best_score = loss_sum

            if (self.minimize and loss_sum < self.best_score) or (not self.minimize and loss_sum > self.best_score) or (
                    self.epoch_id == 0):
                self.best_score = loss_sum
                persist_torch_model(self.model, self.filepath)
                logger.info('epoch {0} model saved to {1}'.format(self.epoch_id, self.filepath))

        self.epoch_id += 1


class EarlyStopping(Callback):
    def __init__(self, metric_name='sum', patience=1000, minimize=True):
        super().__init__()
        self.patience = patience
        self.minimize = minimize
        self.best_score = None
        self.epoch_since_best = 0
        self._training_break = False
        self.metric_name = metric_name

    def training_break(self, *args, **kwargs):
        return self._training_break

    def on_epoch_end(self, *args, **kwargs):
        self.model.eval()
        val_loss = self.get_validation_loss()
        loss_sum = val_loss[self.metric_name]
        loss_sum = loss_sum.data.cpu().numpy()[0]

        self.model.train()

        if not self.best_score:
            self.best_score = loss_sum

        if (self.minimize and loss_sum < self.best_score) or (not self.minimize and loss_sum > self.best_score):
            self.best_score = loss_sum
            self.epoch_since_best = 0
        else:
            self.epoch_since_best += 1

        if self.epoch_since_best > self.patience:
            self._training_break = True
        self.epoch_id += 1


def postprocessing_pipeline_simplified(cache_dirpath, loader_mode):
    if loader_mode == 'resize_and_pad':
        size_adjustment_function = partial(crop_image, target_size=ORIGINAL_SIZE)
    elif loader_mode == 'resize':
        size_adjustment_function = partial(resize_image, target_size=ORIGINAL_SIZE)
    else:
        raise NotImplementedError

    mask_resize = Step(name='mask_resize',
                       transformer=make_apply_transformer(size_adjustment_function,
                                                          output_name='resized_images',
                                                          apply_on=['images']),
                       input_data=['unet_output'],
                       adapter=Adapter({'images': E('unet_output', 'mask_prediction'),
                                        }),
                       experiment_directory=cache_dirpath)

    binarizer = Step(name='binarizer',
                     transformer=make_apply_transformer(
                         partial(binarize, threshold=THRESHOLD),
                         output_name='binarized_images',
                         apply_on=['images']),
                     input_steps=[mask_resize],
                     adapter=Adapter({'images': E(mask_resize.name, 'resized_images'),
                                      }),
                     experiment_directory=cache_dirpath)

    output = Step(name='output',
                  transformer=IdentityOperation(),
                  input_steps=[binarizer],
                  adapter=Adapter({'y_pred': E(binarizer.name, 'binarized_images'),
                                   }),
                  experiment_directory=cache_dirpath)

    return output
