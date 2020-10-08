import os
import numpy as np
import random
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm_notebook
import IPython.display as ipd
import matplotlib.pyplot as plt
import museval as eval4
import math
import statistics as stats
import soundfile
                    
class MusdbLoaer:
    
    def __init__(self, musdb_path = 'data/musdb18/', 
                 n_fft=2**11, hop_factor=2, dim_t=2**6, sr=44100, trim=5000,
                 device='cuda', mode='complex'):
        
        self.musdb_path = musdb_path
        self.musdb_train_path = musdb_path + 'train/'
        self.musdb_valid_path = musdb_path + 'valid/'
        self.musdb_test_path = musdb_path + 'test/'
        self.mix_name = 'linear_mixture'
        self.source_names = ['vocals', 'drums', 'bass', 'other']
        self.mode = mode
        

        self.dim_c = 4 if self.mode == 'complex' else 2
        
        self.n_fft = n_fft
        self.dim_f = n_fft//2
        self.hop_factor = hop_factor
        self.hop_length = n_fft//hop_factor
        self.dim_t = dim_t
        self.sampling_rate = sr
        self.sampling_size = self.hop_length * (self.dim_t+self.hop_factor-1)
        self.trim = trim  # trim each generated sub-signal (noise due to conv zero-padding)
        
        self.device = device


    def to_specs(self, signal):
        
        if(self.mode == 'complex'):
            specs = []
            for channel in signal:
                spectrogram = librosa.stft(np.array(channel, dtype=np.float32), n_fft=self.n_fft, center=False, hop_length=self.hop_length)
                specs.append(spectrogram.real)
                specs.append(spectrogram.imag)
            return np.array(specs)

        else: # mag phase mode
            m, p = [], []
            for channel in signal:
                spectrogram = librosa.stft(np.array(channel, dtype=np.float32), n_fft=self.n_fft, center=False, hop_length=self.hop_length)
                m.append(np.abs(spectrogram))
                p.append(np.angle(spectrogram))
            return np.array(m), np.array(p)


    def restore(self, specs):

        ri = np.reshape(specs, (-1, 2, self.dim_f+1, self.dim_t))
        channels = []
        for ri_c in ri:
            ft_c = ri_c[0] + 1.j * ri_c[1]
            channels.append(librosa.istft(ft_c, center=False, hop_length=self.hop_length))
        return np.array(channels)


    def restore_mag_phase(self, m,p):
        ft = m * np.exp(1.j*p)
        channels = []
        for ft_c in ft:
            channels.append(librosa.istft(ft_c, center=False, hop_length=self.hop_length))
        return np.array(channels)

    
    def load(self, path, max_length=None):
        if max_length is None:
            return librosa.load(path, sr=self.sampling_rate, mono=False)[0]
        else:
            d = (self.sampling_size+1) / self.sampling_rate
            s = np.random.randint(max_length - self.sampling_size - 10) / self.sampling_rate
            return librosa.load(path, sr=None, mono=False, offset=s, duration=d)[0][:,:self.sampling_size]

class MusdbTrainingDataset(Dataset):
    def __init__(self, musdb_loader, target_name='vocals'):
        assert musdb_loader.mode == 'complex' or musdb_loader.mode =='magphase'

        self.mode = musdb_loader.mode
        self.musdb_loader = musdb_loader
        self.file_path = self.musdb_loader.musdb_train_path
        self.target_path = self.file_path+target_name
        self.lengths = np.load(self.file_path+'lengths.npy')
        self.source_names = self.musdb_loader.source_names
        self.target_name = target_name
    
    def __len__(self):
        return len(self.lengths)
    
    def __getitem__(self, index):
        def coin_toss():
            return np.random.rand() < 0.5 
        
        target = self.musdb_loader.load('{0}/{1:02}.wav'.format(self.target_path, index), self.lengths[index])
        
        # mixing different songs
        mix = target
        for t_name in self.source_names:
            if t_name!=self.target_name:
                index2 = np.random.randint(len(self))
                target2 = self.musdb_loader.load('{0}/{1}/{2:02}.wav'.format(self.file_path, t_name, index2), self.lengths[index2])
                mix = mix + target2
            
        if(self.mode =='complex'):
            mix_specs = torch.tensor(self.musdb_loader.to_specs(mix))
            target_specs = torch.tensor(self.musdb_loader.to_specs(target))

        else:
            mix_specs = torch.tensor(self.musdb_loader.to_specs(mix)[0])
            target_specs = torch.tensor(self.musdb_loader.to_specs(target)[0])
            
        return mix_specs, target_specs

def preprocess_track(musdb_loader, y):
    n_sample = y.shape[1]    
    gen_size = musdb_loader.sampling_size-2*musdb_loader.trim
    pad = gen_size - n_sample%gen_size
    y_p = np.concatenate((np.zeros((2,musdb_loader.trim)), y, np.zeros((2,pad)), np.zeros((2,musdb_loader.trim))), 1)
    
    if(musdb_loader.mode == 'complex'):
        all_specs = []
        i = 0
        while i < n_sample + pad:
            specs = musdb_loader.to_specs(y_p[:, i:i+musdb_loader.sampling_size])
            all_specs.append(specs)
            i += gen_size

        return torch.tensor(all_specs), pad
    else:
        mag, phase = [], []
        i = 0
        while i < n_sample + pad:
            specs = musdb_loader.to_specs(y_p[:, i:i+musdb_loader.sampling_size])
            mag.append(specs[0])
            phase.append(specs[1])
            i += gen_size

        return torch.tensor(mag), np.array(phase), pad

def separate(musdb_loader, model, mix_path, batch_size=16):

    model.eval()
    
    if(musdb_loader.mode == 'complex'):
        mix_specs, pad_len = preprocess_track(musdb_loader, mix_path)
    else:
        mix_mag, mix_phase, pad_len = preprocess_track(musdb_loader, mix_path)
        
    i = 0
    num_intervals = mix_specs.shape[0] if(musdb_loader.mode == 'complex') else mix_mag.shape[0]
    batches = []
    while i < num_intervals:
        if(musdb_loader.mode == 'complex'):
            batches.append(mix_specs[i:i+batch_size])
        else:
            batches.append(mix_mag[i:i+batch_size])
        i = i + batch_size

    # obtain estimated target spectrograms
    if(musdb_loader.mode == 'complex'):  

        tar_signal = np.array([[],[]])
        with torch.no_grad():
            for batch in batches:
                tar_specs = model(batch.to(musdb_loader.device))
                for tar_spec in tar_specs:
                    est_interval = np.array(musdb_loader.restore(tar_spec.detach().cpu().numpy()))[:, musdb_loader.trim:-musdb_loader.trim]
                    tar_signal = np.concatenate((tar_signal, est_interval), 1)

        return tar_signal[:, :-pad_len]
    
    else:
        tar_signal = np.array([[],[]])
        with torch.no_grad():
            i = 0
            for batch in batches:
                tar_mags = model(batch.to(musdb_loader.device))
                for tar_mag in tar_mags:
                    est_interval = np.array(musdb_loader.restore_mag_phase(tar_mag.detach().cpu().numpy(), mix_phase[i]))[:, musdb_loader.trim:-musdb_loader.trim]

                    tar_signal = np.concatenate((tar_signal, est_interval), 1)
                    i += 1

        return tar_signal[:, :-pad_len]        

def median_nan(a):
    return np.median(a[~np.isnan(a)])

def musdb_sdr(ref, est, sr):
    sdr, isr, sir, sar, perm = eval4.metrics.bss_eval(ref, est, window=sr, hop=sr)
    return median_nan(sdr[0])

def musdb_all(ref, est, sr):
    '''return sdr, isr, sir, sar'''
    sdr, isr, sir, sar, _ = eval4.metrics.bss_eval(ref, est, window=sr, hop=sr)
    return [median_nan(metric[0]) for metric in [sdr, isr, sir, sar]]

def mse(ref, est):
    return ((ref-est)**2).mean()

def eval_testset_pretrained (musdb_loader, model, target_name, batch_size=16):
    
    def separator (mix):
        return separate(musdb_loader, model, mix, batch_size)

    return eval_testset(musdb_loader, separator, target_name)

def eval_testset (musdb_loader, separator, target_name):
    SDR = []
    for i in range(50):
        ref = musdb_loader.load('{0}/{1}/{2:02}.wav'.format(musdb_loader.musdb_test_path, target_name, i))
        mix = musdb_loader.load('{0}/{1}/{2:02}.wav'.format(musdb_loader.musdb_test_path, musdb_loader.mix_name, i))
        est = separator(mix)
        sdr = musdb_sdr(np.array([ref.T]), np.array([est.T]),musdb_loader.sampling_rate)

        SDR.append(sdr)

        ipd.clear_output(wait=True)
        print(sdr)
        plt.plot(SDR)
        plt.show()

        print('SDR mean:', stats.mean(SDR))
        print('SDR median:', stats.median(SDR))
        
    return SDR, stats.mean(SDR), stats.median(SDR)