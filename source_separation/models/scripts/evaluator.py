import inspect
from warnings import warn

from pytorch_lightning.loggers import WandbLogger

from source_separation.data.musdb_wrapper.dataloaders import DataProvider
from source_separation.models.model_definition import get_class_by_name
from source_separation.utils.functions import mkdir_if_not_exists
from pathlib import Path
from pytorch_lightning import Trainer


def eval(param):

    if not isinstance(param, dict):
        args = vars(param)
    else:
        args = param


    # MODEL
    ##########################################################
    # # # get framework
    framework = get_class_by_name('one2one_separation', args['model'])
    # # # Model instantiation
    model = framework(**args)
    ##########################################################

    # DATASET
    ##########################################################
    if args['target_name'] is None:
        print('--target_name parameter should be filled: one of [vocals, bass, drums, other]')
        exit(-1)

    elif args['target_name'] not in ['vocals', 'bass', 'drums', 'other']:
        print('--target_name parameter should be filled: one of [vocals, bass, drums, other]')
        exit(-1)

    if args['target_names'] != [args['target_name']]:
        warn('--target_names should be the same as --target_name in one2one separation mode. \nAuto correction: '
             'target_names = [target_name]')
        args['target_names'] = [args['target_name']]

    data_provider = DataProvider(**args)
    ##########################################################
    # Trainer Definition

    # -- checkpoint
    ckpt_path = Path(args['ckpt_root_path']).joinpath(args['model']).joinpath(args['run_id'])
    ckpt_path = '{}/{}'.format(str(ckpt_path), args['epoch'])

    # -- logger setting
    log = args['log']
    if log == 'False' or args['dev_mode']:
        args['logger'] = False
        args['checkpoint_callback'] = False
        args['early_stop_callback'] = False
    elif log == 'wandb':
        args['logger'] = WandbLogger(project='one2one_separation', tags=args['model'], offline=False,
                                     id=args['run_id'] + '_eval_' + args['epoch'].replace('=','_'))
        args['logger'].log_hyperparams(model.hparams)
        args['logger'].watch(model, log='all')
    elif log == 'tensorboard':
        raise NotImplementedError
    else:
        args['logger'] = True  # default
        default_save_path = 'etc/lightning_logs'
        mkdir_if_not_exists(default_save_path)

    # Trainer
    if args['gpus'] > 1:
        warn('# gpu and num_workers should be 1, Not implemented: museval for distributed parallel')
        args['gpus'] = 1
        args['distributed_backend'] = None

    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict((name, args[name]) for name in valid_kwargs if name in args)

    trainer = Trainer(**trainer_kwargs)
    n_fft, hop_length, num_frame = args['n_fft'], args['hop_length'], args['num_frame']
    test_data_loader = data_provider.get_test_dataloader(n_fft, hop_length, num_frame)

    #if args['save_weights_only']:
    model = model.load_from_checkpoint(ckpt_path)
    trainer.test(model, test_data_loader)


    return None
