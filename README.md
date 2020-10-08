# Investigating U-NETS With Various Intermediate Blocks For Spectrogram-based Singing Voice Separation


A Pytorch Implementation of the paper "Investigating U-NETS With Various Intermediate Blocks For Spectrogram-based Singing Voice Separation (ISMIR 2020)"

## Installation

```
conda install pytorch=1.6 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge ffmpeg librosa
conda install -c anaconda jupyter
pip install musdb museval pytorch_lightning effortless_config wandb pydub nltk spacy 
```

## Dataset

1. Download [Musdb18](https://sigsep.github.io/datasets/musdb.html)
2. Unzip files
3. We recommend you to use the wav file mode for the fast data preparation. 
    - please see the [musdb instruction](https://pypi.org/project/musdb/) for the wave file mode.
    ```shell script
   musdbconvert path/to/musdb-stems-root path/to/new/musdb-wav-root
   ```

## Tutorial

### 1. activate your conda

```
conda activate yourcondaname
```

### 2. Training a default UNet with TFC_TDFs

```
python main.py --musdb_root ../repos/musdb18_wav --musdb_is_wav True --filed_mode True --target_name vocals --mode train --gpus 4 --distributed_backend ddp --sync_batchnorm True --pin_memory True --num_workers 32 --precision 16 --run_id debug --optimizer adam --lr 0.001 --save_top_k 3 --patience 100 --min_epochs 1000 --max_epochs 2000 --n_fft 2048 --hop_length 1024 --num_frame 128  --train_loss spec_mse --val_loss raw_l1 --model tfc_tdf_net  --spec_est_mode mapping --spec_type complex --n_blocks 7 --internal_channels 24  --n_internal_layers 5 --kernel_size_t 3 --kernel_size_f 3 --min_bn_units 16 --tfc_tdf_activation relu  --first_conv_activation relu --last_activation identity 
```

--spec_type complex --spec_est_mode mapping --filed_mode True

## How to use

### 1. Training

#### A. General Parameters

  -   --spec_type complex --spec_est_mode mapping

- ```--musdb_root``` musdb path
- ```--musdb_is_wav``` whether the path contains wav files or not
- ```--filed_mode``` whether you want to use filed mode or not. recommend to use it for the fast data preparation.
- ```--target_name``` one of vocals, drum, bass, other
#### B. Training Environment

- ```--mode``` train or eval
- ```--gpus``` number of gpus
    - ***(WARN)*** gpus > 1 might be problematic when evaluating models.
- ```distributed_backend``` use this option only when you are using multi-gpus. distributed backend, one of ddp, dp, ... we recommend you to use ddp.
- ```--sync_batchnorm``` True only when you are using ddp
- ```--pin_memory```
- ```--num_workers```
- ```--precision``` 16 or 32
- ```--dev_mode``` whether you want a developement mode or not. dev mode is much faster because it uses only a small subset of the dataset.
- ```--run_id``` (optional) directory path where you want to store logs and etc. if none then the timestamp.
-- ```--log``` True for default pytorch lightning log. ```wandb``` is also available.
#### C. Training hyperparmeters
- ```--batch_size``` trivial :)
- ```--optimizer``` adam, rmsprop, etc
- ```--lr``` learning rate
- ```--save_top_k``` how many top-k epochs you want to save the training state (criterion: validation loss)
- ```--patience``` early stop control parameter. see pytorch lightning docs.
- ```--min_epochs``` trivial :)
- ```--max_epochs``` trivial :)
- ```--model```
    - tfc_tdf_net (current available)

#### D. Fourier parameters
- ```--n_fft```
- ```--hop_length```
-- ```num_frame``` number of frames (time slices) 
#### F. criterion
- ```--train_loss```: spec_mse, raw_l1, etc...
- ```--val_loss```: spec_mse, raw_l1, etc...

#### G. SVS Framework
- ```--spec_type```: type of a spectrogram. ['complex', 'magnitude']
- ```--spec_est_mode```: spectrogram estimation method. ['mapping', 'masking']

## Reference

[1] Woosung Choi, Minseok Kim, Jaehwa Chung, Daewon Lee, and Soonyoung Jung. "[Investigating Deep Neural Transformations for Spectrogram-based Musical Source Separation](https://arxiv.org/abs/1912.02591)" arXiv preprint arXiv:1912.02591 (2019).
