from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from typing import Union, List
from warnings import warn

import numpy as np
import pydub
import pytorch_lightning as pl
import soundfile
import torch
import wandb
from pytorch_lightning import EvalResult
from pytorch_lightning.loggers import WandbLogger

from source_separation.models import loss_functions
from source_separation.utils import fourier
from source_separation.utils.fourier import get_trim_length
from source_separation.utils.functions import get_optimizer_by_name
from source_separation.utils.weight_initialization import init_weights_functional


def get_estimation(idx, target_name, estimation_dict):
    estimated = estimation_dict[target_name][idx]
    if len(estimated) == 0:
        warn('TODO: zero estimation, caused by ddp')
        return None
    estimated = np.concatenate([estimated[key] for key in sorted(estimated.keys())], axis=0)
    return estimated


class Source_Separation(pl.LightningModule, metaclass=ABCMeta):

    @staticmethod
    def get_arg_keys():
        return ['target_name', 'n_fft', 'hop_length', 'num_frame', 'optimizer', 'lr', 'dev_mode', 'auto_lr_find',
                'val_loss']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--optimizer', type=str, default='adam')

        return loss_functions.add_model_specific_args(parser)

    def __init__(self, target_name, n_fft, hop_length, num_frame, optimizer, lr, dev_mode):
        super(Source_Separation, self).__init__()

        self.target_name = target_name
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.trim_length = get_trim_length(self.hop_length)
        self.n_trim_frames = self.trim_length // self.hop_length
        self.num_frame = num_frame

        self.lr = lr
        self.optimizer = optimizer

        self.valid_estimation_dict = {}
        self.dev_mode = dev_mode

    @abstractmethod
    def init_weights(self):
        pass

    def configure_optimizers(self):
        optimizer = get_optimizer_by_name(self.optimizer)
        return optimizer(self.parameters(), lr=float(self.lr))

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def on_test_epoch_start(self):

        import os
        output_folder = 'museval_output'
        if os.path.exists(output_folder):
            os.rmdir(output_folder)
        os.mkdir(output_folder)

        self.valid_estimation_dict = None
        self.test_estimation_dict = {}

        self.musdb_test = self.test_dataloader().dataset
        num_tracks = self.musdb_test.num_tracks

        self.test_estimation_dict[self.target_name] = {mixture_idx: {}
                                                       for mixture_idx
                                                       in range(num_tracks)}

    def test_step(self, batch, batch_idx):
        mixtures, mixture_ids, chunk_ids, input_conditions, target_names = batch

        estimated_targets = self.separate(mixtures)[:, self.trim_length:-self.trim_length]

        for mixture, mixture_idx, chunk_id, input_condition, target_name, estimated_target \
                in zip(mixtures, mixture_ids, chunk_ids, input_conditions, target_names, estimated_targets):
            self.test_estimation_dict[target_name][mixture_idx.item()][
                chunk_id.item()] = estimated_target.detach().cpu().numpy()

        return torch.zeros(0)

    def on_test_epoch_end(self):

        import museval
        results = museval.EvalStore(frames_agg='median', tracks_agg='median')
        idx_list = range(self.musdb_test.num_tracks)
        target_name = self.target_name
        for idx in idx_list:
            estimation = {target_name: get_estimation(idx, target_name, self.test_estimation_dict)}

            if estimation[target_name] is not None:
                estimation[target_name] = estimation[target_name].astype(np.float32)

            # Real SDR
            if len(estimation) == 1:
                track_length = self.musdb_test.musdb_test[idx].samples
                estimated_target = estimation[target_name][:track_length]

                if track_length > estimated_target.shape[0]:
                    raise NotImplementedError
                else:
                    estimated_target = estimation[target_name][:track_length]

                    estimated_targets_dict = {target_name: estimated_target,
                                              'accompaniment': self.musdb_test.musdb_test[idx].audio.astype(
                                                  np.float32) - estimated_target}
                    track_score = museval.eval_mus_track(
                        self.musdb_test.musdb_test[idx],
                        estimated_targets_dict
                    )

                    score_dict = track_score.df.loc[:, ['target', 'metric', 'score']].groupby(
                        ['target', 'metric'])['score'] \
                        .median().to_dict()

                    if isinstance(self.logger, WandbLogger):
                        self.logger.experiment.log(
                            {'test_result/{}_{}'.format(k1, k2): score_dict[(k1, k2)] for k1, k2 in score_dict.keys()})

                        if idx == 1:
                            self.logger.experiment.log({'result_sample_{}_{}'.format(self.current_epoch, target_name): [
                                wandb.Audio(estimated_target,
                                            caption='{}_{}'.format(idx, target_name),
                                            sample_rate=44100)]})

                    else:
                        print(track_score)
                        if idx == 1:
                            self.export_mp3(1, self.target_name)
                    results.add_track(track_score)

            else:
                raise NotImplementedError

        if isinstance(self.logger, WandbLogger):

            result_dict = results.df.groupby(
                ['track', 'target', 'metric']
            )['score'].median().reset_index().groupby(
                ['target', 'metric']
            )['score'].median().to_dict()

            self.logger.experiment.log(
                {'test_result/agg/{}_{}'.format(k1, k2): result_dict[(k1, k2)] for k1, k2 in result_dict.keys()}
            )
        else:
            print(results)

    def export_mp3(self, idx, target_name):
        estimated = self.test_estimation_dict[target_name][idx]
        estimated = np.concatenate([estimated[key] for key in sorted(estimated.keys())], axis=0)
        soundfile.write('tmp_output.wav', estimated, samplerate=44100)
        audio = pydub.AudioSegment.from_wav('tmp_output.wav')
        audio.export('{}_estimated/output_{}.mp3'.format(idx, target_name))

    @abstractmethod
    def separate(self, input_signal) -> torch.Tensor:
        pass


class Spectrogram_based(Source_Separation, metaclass=ABCMeta):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--target_name', type=str, default='vocals')

        parser.add_argument('--n_fft', type=int, default=1024)
        parser.add_argument('--hop_length', type=int, default=256)
        parser.add_argument('--num_frame', type=int, default=128)
        parser.add_argument('--spec_type', type=str, default='magnitude')
        parser.add_argument('--spec_est_mode', type=str, default='masking')

        parser.add_argument('--layer_level_init_weight', type=bool, default=False)
        parser.add_argument('--train_loss', type=str, default='spec_mse')
        parser.add_argument('--val_loss', type=str, default='spec_mse')
        parser.add_argument('--unfreeze_stft_from', type=int, default=-1)  # -1 means never.

        return Source_Separation.add_model_specific_args(parser)

    def __init__(self, target_name, n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 spec2spec,
                 optimizer, lr, dev_mode,
                 train_loss, val_loss,
                 layer_level_init_weight,
                 unfreeze_stft_from):

        super(Spectrogram_based, self).__init__(
            target_name,
            n_fft, hop_length, num_frame,
            optimizer, lr, dev_mode
        )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frame = num_frame

        assert spec_type in ['magnitude', 'complex']
        assert spec_est_mode in ['masking', 'mapping']
        self.magnitude_based = spec_type == 'magnitude'
        self.masking_based = spec_est_mode == 'masking'
        self.stft = fourier.multi_channeled_STFT(n_fft=n_fft, hop_length=hop_length)
        self.spec2spec = spec2spec

        if layer_level_init_weight is None:
            self.layer_level_init_weight = False
        elif layer_level_init_weight == "False":
            self.layer_level_init_weight = False
        elif layer_level_init_weight == "True":
            self.layer_level_init_weight = True
        else:
            self.layer_level_init_weight = layer_level_init_weight

        self.init_weights()
        self.unfreeze_stft_from = unfreeze_stft_from

        self.val_loss = val_loss
        self.train_loss = train_loss

    def init_weights(self):
        if self.layer_level_init_weight:
            self.spec2spec.init_weights()
        else:
            init_weights_functional(self.spec2spec,
                                    self.spec2spec.activation)

    def forward(self, input_signal):
        input_spec = self.to_spec(input_signal)
        output_spec = self.spec2spec(input_spec)

        if self.masking_based:
            output_spec = input_spec * output_spec

        return output_spec

    def forward_with_target_spec(self, input_signal, target_signal):
        input_spec = self.to_spec(input_signal)
        output_spec = self.spec2spec(input_spec)
        target_spec = self.to_spec(target_signal)

        if self.masking_based:
            output_spec = input_spec * output_spec

        return output_spec, target_spec

    @abstractmethod
    def to_spec(self, input_signal) -> torch.Tensor:
        pass

    def separate(self, input_signal) -> torch.Tensor:

        phase = None
        if self.magnitude_based:
            mag, phase = self.stft.to_mag_phase(input_signal)
            input_spec = mag.transpose(-1, -3)

        else:
            spec_complex = self.stft.to_spec_complex(input_signal)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            input_spec = spec_complex.transpose(-1, -3)  # *, 2ch, T, N

        output_spec = self.spec2spec(input_spec[..., 1:])

        if self.masking_based:
            output_spec = input_spec[..., 1:] * output_spec
        else:
            pass  # Use the original output_spec

        output_spec = torch.cat([input_spec[..., :1], output_spec], dim=-1)
        output_spec = output_spec.transpose(-1, -3)

        if self.magnitude_based:
            restored = self.stft.restore_mag_phase(output_spec, phase)
        else:
            # output_spec: *, N, T, 2ch
            output_spec = output_spec.view(list(output_spec.shape[:-1]) + [2, -1])  # *, N, T, 2, ch
            restored = self.stft.restore_complex(output_spec)

        return restored

    def on_train_epoch_start(self):
        if self.unfreeze_stft_from < 0:
            self.stft.freeze()

        elif self.unfreeze_stft_from <= self.current_epoch:
            self.stft.unfreeze()
        else:
            self.stft.freeze()

    def training_step(self, batch, batch_idx):
        mixture_signal, target_signal, condition = batch
        loss = self.train_loss(self, mixture_signal, target_signal)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                   reduce_fx=torch.mean)
        return result

    # Validation Process
    def on_validation_epoch_start(self):

        self.valid_estimation_dict[self.target_name] = {mixture_idx: {}
                                                        for mixture_idx
                                                        in range(14)}

    def validation_step(self, batch, batch_idx):
        mixtures, mixture_ids, window_offsets, input_conditions, target_names, targets = batch
        loss = self.val_loss(self, mixtures, targets)
        result = pl.EvalResult()
        result.log('raw_val_loss', loss, prog_bar=False, logger=False, reduce_fx=torch.mean)

        # Result Cache
        if 0 in mixture_ids.view(-1):
            estimated_targets = self.separate(mixtures)[:, self.trim_length:-self.trim_length]
            for mixture, mixture_idx, window_offset, input_condition, target_name, estimated_target \
                    in zip(mixtures, mixture_ids, window_offsets, input_conditions, target_names, estimated_targets):

                if mixture_idx == 0:
                    self.valid_estimation_dict[target_name][mixture_idx.item()][
                        window_offset.item()] = estimated_target.detach().cpu().numpy()

        return result

    def validation_epoch_end(self, outputs: Union[EvalResult, List[EvalResult]]) -> EvalResult:

        for idx in [0]:
            estimation = {}
            target_name = self.target_name
            estimation[target_name] = get_estimation(idx, target_name, self.valid_estimation_dict)
            if estimation[target_name] is None:
                continue
            if estimation[target_name] is not None:
                estimation[target_name] = estimation[target_name].astype(np.float32)

                if self.current_epoch > 10 and isinstance(self.logger, WandbLogger):
                    self.logger.experiment.log({'result_sample_{}_{}'.format(self.current_epoch, target_name): [
                        wandb.Audio(estimation[target_name][44100 * 20:44100 * 25],
                                    caption='{}_{}'.format(idx, target_name),
                                    sample_rate=44100)]})

        reduced_loss = sum(outputs['raw_val_loss'] / len(outputs['raw_val_loss']))
        result = pl.EvalResult(early_stop_on=reduced_loss, checkpoint_on=reduced_loss)
        result.log('val_loss', reduced_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return result
