from functools import partial

from steppy.base import Step, IdentityOperation
from steppy.adapter import Adapter, E

from . import loaders
from .utils import make_apply_transformer
from .postprocessing import binarize


def preprocessing_train(config, model_name='unet', suffix=''):
    if config.general.loader_mode == 'resize_and_pad':
        Loader = loaders.ImageSegmentationLoaderResizePad
        loader_config = config.loaders.resize_and_pad
    elif config.general.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResize
        loader_config = config.loaders.resize
    else:
        raise NotImplementedError

    if loader_config.dataset_params.image_source == 'memory':
        reader_train = Step(name='reader_train{}'.format(suffix),
                            transformer=loaders.ImageReader(train_mode=True, **config.reader[model_name]),
                            input_data=['input'],
                            adapter=Adapter({'meta': E('input', 'meta')}),
                            experiment_directory=config.execution.experiment_dir)

        reader_inference = Step(name='reader_inference{}'.format(suffix),
                                transformer=loaders.ImageReader(train_mode=True, **config.reader[model_name]),
                                input_data=['callback_input'],
                                adapter=Adapter({'meta': E('callback_input', 'meta_valid')}),
                                experiment_directory=config.execution.experiment_dir)

    elif loader_config.dataset_params.image_source == 'disk':
        reader_train = Step(name='xy_train{}'.format(suffix),
                            transformer=loaders.XYSplit(train_mode=True, **config.xy_splitter[model_name]),
                            input_data=['input'],
                            adapter=Adapter({'meta': E('input', 'meta')}),
                            experiment_directory=config.execution.experiment_dir)

        reader_inference = Step(name='xy_inference{}'.format(suffix),
                                transformer=loaders.XYSplit(train_mode=True, **config.xy_splitter[model_name]),
                                input_data=['callback_input'],
                                adapter=Adapter({'meta': E('callback_input', 'meta_valid')}),
                                experiment_directory=config.execution.experiment_dir)
    else:
        raise NotImplementedError

    loader = Step(name='loader{}'.format(suffix),
                  transformer=Loader(train_mode=True, **loader_config),
                  input_steps=[reader_train, reader_inference],
                  adapter=Adapter({'X': E(reader_train.name, 'X'),
                                   'y': E(reader_train.name, 'y'),
                                   'X_valid': E(reader_inference.name, 'X'),
                                   'y_valid': E(reader_inference.name, 'y'),
                                   }),
                  experiment_directory=config.execution.experiment_dir)
    return loader


def preprocessing_inference(config, model_name='unet', suffix=''):
    if config.general.loader_mode == 'resize_and_pad':
        Loader = loaders.ImageSegmentationLoaderResizePad
        loader_config = config.loaders.resize_and_pad
    elif config.general.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResize
        loader_config = config.loaders.resize
    else:
        raise NotImplementedError

    if loader_config.dataset_params.image_source == 'memory':
        reader_inference = Step(name='reader_inference{}'.format(suffix),
                                transformer=loaders.ImageReader(train_mode=False, **config.reader[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),

                                experiment_directory=config.execution.experiment_dir)

    elif loader_config.dataset_params.image_source == 'disk':
        reader_inference = Step(name='xy_inference{}'.format(suffix),
                                transformer=loaders.XYSplit(train_mode=False, **config.xy_splitter[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),
                                experiment_directory=config.execution.experiment_dir)
    else:
        raise NotImplementedError

    loader = Step(name='loader{}'.format(suffix),
                  transformer=Loader(train_mode=False, **loader_config),
                  input_steps=[reader_inference],
                  adapter=Adapter({'X': E(reader_inference.name, 'X'),
                                   'y': E(reader_inference.name, 'y'),
                                   }),
                  experiment_directory=config.execution.experiment_dir,
                  cache_output=True)
    return loader


def preprocessing_inference_tta(config, model_name='unet', suffix=''):
    if config.general.loader_mode == 'resize_and_pad':
        Loader = loaders.ImageSegmentationLoaderResizePad
        loader_config = config.loaders.resize_and_pad
    elif config.general.loader_mode == 'resize':
        Loader = loaders.ImageSegmentationLoaderResize
        loader_config = config.loaders.resize
    else:
        raise NotImplementedError

    if loader_config.dataset_params.image_source == 'memory':
        reader_inference = Step(name='reader_inference{}'.format(suffix),
                                transformer=loaders.ImageReader(train_mode=False, **config.reader[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),
                                experiment_directory=config.execution.experiment_dir)

        tta_generator = Step(name='tta_generator{}'.format(suffix),
                             transformer=loaders.TestTimeAugmentationGenerator(**config.tta_generator),
                             input_steps=[reader_inference],
                             adapter=Adapter({'X': E('reader_inference', 'X')}),
                             experiment_directory=config.execution.experiment_dir)

    elif loader_config.dataset_params.image_source == 'disk':
        reader_inference = Step(name='reader_inference{}'.format(suffix),
                                transformer=loaders.XYSplit(train_mode=False, **config.xy_splitter[model_name]),
                                input_data=['input'],
                                adapter=Adapter({'meta': E('input', 'meta')}),
                                experiment_directory=config.execution.experiment_dir)

        tta_generator = Step(name='tta_generator{}'.format(suffix),
                             transformer=loaders.MetaTestTimeAugmentationGenerator(**config.tta_generator),
                             input_steps=[reader_inference],
                             adapter=Adapter({'X': E('reader_inference', 'X')}),
                             experiment_directory=config.execution.experiment_dir)
    else:
        raise NotImplementedError

    loader = Step(name='loader{}'.format(suffix),
                  transformer=Loader(**loader_config),
                  input_steps=[tta_generator],
                  adapter=Adapter({'X': E(tta_generator.name, 'X_tta'),
                                   'tta_params': E(tta_generator.name, 'tta_params'),
                                   }),
                  experiment_directory=config.execution.experiment_dir,
                  cache_output=True)
    return loader, tta_generator


def aggregator(name, model, tta_generator, experiment_directory, config):
    tta_aggregator = Step(name=name,
                          transformer=loaders.TestTimeAugmentationAggregator(**config),
                          input_steps=[model, tta_generator],
                          adapter=Adapter({'images': E(model.name, 'mask_prediction'),
                                           'tta_params': E(tta_generator.name, 'tta_params'),
                                           'img_ids': E(tta_generator.name, 'img_ids'),
                                           }),
                          experiment_directory=experiment_directory)
    return tta_aggregator


def mask_postprocessing(config, suffix=''):
    binarizer = Step(name='binarizer{}'.format(suffix),
                     transformer=make_apply_transformer(partial(binarize, threshold=config.thresholder.threshold_masks),
                                                        output_name='binarized_images',
                                                        apply_on=['images']),
                     input_data=['input_masks'],
                     adapter=Adapter({'images': E('input_masks', 'resized_images'),
                                      }),
                     experiment_directory=config.execution.experiment_dir)
    return binarizer