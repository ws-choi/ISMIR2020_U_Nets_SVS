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

## Demonstration: A Pretrained Model (TFC_TDF_Net (large))

[Colab Link](https://colab.research.google.com/github/ws-choi/ISMIR2020_U_Nets_SVS/blob/master/colab_demo/TFC_TDF_Net_Large.ipynb)

## Tutorial

### 1. activate your conda

```shell script
conda activate yourcondaname
```

### 2. Training a default UNet with TFC_TDFs

```
python main.py --musdb_root ../repos/musdb18_wav --musdb_is_wav True --filed_mode True --target_name vocals --mode train --gpus 4 --distributed_backend ddp --sync_batchnorm True --pin_memory True --num_workers 32 --precision 16 --run_id debug --optimizer adam --lr 0.001 --save_top_k 3 --patience 100 --min_epochs 1000 --max_epochs 2000 --n_fft 2048 --hop_length 1024 --num_frame 128  --train_loss spec_mse --val_loss raw_l1 --model tfc_tdf_net  --spec_est_mode mapping --spec_type complex --n_blocks 7 --internal_channels 24  --n_internal_layers 5 --kernel_size_t 3 --kernel_size_f 3 --min_bn_units 16 --tfc_tdf_activation relu  --first_conv_activation relu --last_activation identity --seed 2020
```

### 3. Evaluation

After training is done, checkpoints are saved in the following directory. 

```etc/modelname/run_id/*.ckpt```

For evaluation, 

```shell script
python main.py --musdb_root ../repos/musdb18_wav --musdb_is_wav True --filed_mode True --target_name vocals --mode eval --gpus 1 --pin_memory True --num_workers 64 --precision 32 --run_id debug --batch_size 4 --n_fft 2048 --hop_length 1024 --num_frame 128 --train_loss spec_mse --val_loss raw_l1 --model tfc_tdf_net --spec_est_mode mapping --spec_type complex --n_blocks 7 --internal_channels 24 --n_internal_layers 5 --kernel_size_t 3 --kernel_size_f 3 --min_bn_units 16 --tfc_tdf_activation relu --first_conv_activation relu --last_activation identity --log wandb --ckpt vocals_epoch=891.ckpt
```

Below is the result.

```shell script
wandb:          test_result/agg/vocals_SDR 6.954695
wandb:   test_result/agg/accompaniment_SAR 14.3738075
wandb:          test_result/agg/vocals_SIR 15.5527
wandb:   test_result/agg/accompaniment_SDR 13.561705
wandb:   test_result/agg/accompaniment_ISR 22.69328
wandb:   test_result/agg/accompaniment_SIR 18.68421
wandb:          test_result/agg/vocals_SAR 6.77698
wandb:          test_result/agg/vocals_ISR 12.45371
```
### 4. Interactive Report (wandb)

[wandb report](https://wandb.ai/wschoi/one2one_separation/reports/Wandb-Report-run_id-debug---VmlldzoyNzQ0Nzg?accessToken=ce3xhzl6x6uplyw55g2wpxlk1q78e238sw6obsxxf4n8ih3pbl43drpa4f51z7fl)

## Indermediate Blocks

Please see this [document](https://github.com/ws-choi/ISMIR2020_U_Nets_SVS/blob/master/paper_with_code/Paper%20with%20Code%20-%203.%20INTERMEDIATE%20BLOCKS.ipynb).

## How to use

### 1. Training

#### 1.1. Intermediate Block independent Parameters

##### 1.1.A. General Parameters

- ```--musdb_root``` musdb path
- ```--musdb_is_wav``` whether the path contains wav files or not
- ```--filed_mode``` whether you want to use filed mode or not. recommend to use it for the fast data preparation.
- ```--target_name``` one of vocals, drum, bass, other

##### 1.1.B. Training Environment

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
- ```--log``` True for default pytorch lightning log. ```wandb``` is also available.
- ```--seed``` random seed for a deterministic result. 

##### 1.1.C. Training hyperparmeters
- ```--batch_size``` trivial :)
- ```--optimizer``` adam, rmsprop, etc
- ```--lr``` learning rate
- ```--save_top_k``` how many top-k epochs you want to save the training state (criterion: validation loss)
- ```--patience``` early stop control parameter. see pytorch lightning docs.
- ```--min_epochs``` trivial :)
- ```--max_epochs``` trivial :)
- ```--model```
    - tfc_tdf_net
    - tfc_net
    - tdc_net

##### 1.1.D. Fourier parameters
- ```--n_fft```
- ```--hop_length```
- ```num_frame``` number of frames (time slices) 
##### 1.1.F. criterion
- ```--train_loss```: spec_mse, raw_l1, etc...
- ```--val_loss```: spec_mse, raw_l1, etc...

#### 1.2. U-net Parameters

- ```--n_blocks```: number of intermediate blocks. must be an odd integer. (default=7)
- ```--input_channels```: 
    - if you use two-channeled complex-valued spectrogram, then 4
    - if you use two-channeled manginutde spectrogram, then 2 
- ```--internal_channels```:  number of internal chennels (default=24)
- ```--first_conv_activation```: (default='relu')
- ```--last_activation```: (default='sigmoid')
- ```--t_down_layers```: list of layer where you want to doubles/halves the time resolution. if None, ds/us applied to every single layer. (default=None)
- ```--f_down_layers```: list of layer where you want to doubles/halves the frequency resolution. if None, ds/us applied to every single layer. (default=None)

#### 1.3. SVS Framework
- ```--spec_type```: type of a spectrogram. ['complex', 'magnitude']
- ```--spec_est_mode```: spectrogram estimation method. ['mapping', 'masking']

- **CaC Framework**
    - you can use cac framework [1] by setting
        - ```--spec_type complex --spec_est_mode mapping --last_activation identity```
- **Mag-only Framework**
    - if you want to use the traditional magnitude-only estimation with sigmoid, then try
        - ```--spec_type magnitude --spec_est_mode masking --last_activation sigmoid```
    - you can also change the last activation as follows
        - ```--spec_type magnitude --spec_est_mode masking --last_activation relu```
- Alternatives
    - you can build an svs framework with any combination of these parameters
    - e.g. ```--spec_type complex --spec_est_mode masking --last_activation tanh```


        
#### 1.4. Block-dependent Parameters

##### 1.4.A. TDF Net

- ```--bn_factor```: bottleneck factor $bn$ (default=16)
- ```--min_bn_units```: when target frequency domain size is too small, we just use this value instead of $\frac{f}{bn}$. (default=16)
- ```--bias```: (default=False) 
- ```--tdf_activation```: activation function of each block (default=relu)       

---

##### 1.4.B. TDC Net

- ```--n_internal_layers```: number of 1-d CNNs in a block (default=5)
- ```--kernel_size_f```: size of kernel of frequency-dimension (default=3)
- ```--tdc_activation```: activation function of each block (default=relu)
        
---
        
##### 1.4.C. TFC Net
- ```--n_internal_layers```: number of 1-d CNNs in a block (default=5)
- ```--kernel_size_t```: size of kernel of time-dimension (default=3)
- ```--kernel_size_f```: size of kernel of frequency-dimension (default=3)
- ```--tfc_activation```: activation function of each block (default=relu)

---
        
##### 1.4.D. TFC_TDF Net
- ```--n_internal_layers```: number of 1-d CNNs in a block (default=5)
- ```--kernel_size_t```: size of kernel of time-dimension (default=3)
- ```--kernel_size_f```: size of kernel of frequency-dimension (default=3)
- ```--tfc_tdf_activation```: activation function of each block (default=relu)       
- ```--bn_factor```: bottleneck factor $bn$ (default=16)
- ```--min_bn_units```: when target frequency domain size is too small, we just use this value instead of $\frac{f}{bn}$. (default=16)
- ```--tfc_tdf_bias```: (default=False)

---

##### 1.4.E. TDC_RNN Net
- ```'--n_internal_layers'``` : number of 1-d CNNs in a block (default=5)

- ```'--kernel_size_f'``` : size of kernel of frequency-dimension (default=3)

- ```'--bn_factor_rnn'``` : (default=16)
- ```'--num_layers_rnn'``` : (default=1)
- ```'--bias_rnn'``` : bool, (default=False)
- ```'--min_bn_units_rnn'``` :  (default=16)
    
- ```'--bn_factor_tdf'``` : (default=16)
- ```'--bias_tdf'``` : bool, (default=False)

- ```'--tdc_rnn_activation'``` : (default='relu')

> current bug - cuda error occurs when tdc_rnn net with precision 16 

## Reproducible Experimental Results

- TFC_TDF_large
    - parameters
    ```
    --musdb_root ../repos/musdb18_wav
    --musdb_is_wav True
    --filed_mode True

    --gpus 4
    --distributed_backend ddp
    --sync_batchnorm True

    --num_workers 72
    --train_loss spec_mse
    --val_loss raw_l1
    --batch_size 12
    --precision 16
    --pin_memory True
    --num_worker 72         
    --save_top_k 3
    --patience 200
    --run_id debug_large
    --log wandb
    --min_epochs 2000
    --max_epochs 3000

    --optimizer adam
    --lr 0.001

    --model tfc_tdf_net
    --n_fft 4096
    --hop_length 1024
    --num_frame 128
    --spec_type complex
    --spec_est_mode mapping
    --last_activation identity
    --n_blocks 9
    --internal_channels 24
    --n_internal_layers 5
    --kernel_size_t 3 
    --kernel_size_f 3 
    --tfc_tdf_bias True
    --seed 2020
  
    ```
    - training
    ``` shell script
    python main.py --musdb_root ../repos/musdb18_wav --musdb_is_wav True --filed_mode True --gpus 4 --distributed_backend ddp --sync_batchnorm True --num_workers 72 --train_loss spec_mse --val_loss raw_l1 --batch_size 24 --precision 16 --pin_memory True --num_worker 72 --save_top_k 3 --patience 200 --run_id debug_large --log wandb --min_epochs 2000 --max_epochs 3000 --optimizer adam --lr 0.001 --model tfc_tdf_net --n_fft 4096 --hop_length 1024 --num_frame 128 --spec_type complex --spec_est_mode mapping --last_activation identity --n_blocks 9 --internal_channels 24 --n_internal_layers 5 --kernel_size_t 3 --kernel_size_f 3 --tfc_tdf_bias True --seed 2020
    ```
    - evaluation result (epoch 2007)
        - SDR 8.029
        - ISR 13.708
        - SIR 16.409
        - SAR 7.533

### Interactive Report (wandb)
[wandb report](https://wandb.ai/wschoi/one2one_separation/reports/Wandb-Report-run_id-debug--VmlldzoyNzQ0OTI?accessToken=uky1arblugoad7xleyiqf7mllw4o2b82w0hnckiekiuc1w29ki8l3kms2yaball0)

## Reference

[1] Woosung Choi, Minseok Kim, Jaehwa Chung, Daewon Lee, and Soonyoung Jung. "[Investigating Deep Neural Transformations for Spectrogram-based Musical Source Separation](https://arxiv.org/abs/1912.02591)" arXiv preprint arXiv:1912.02591 (2019).
