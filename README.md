# FM_AC_IQL
Flow Matching Action Chunk Implicit Q-Learning

## D4RL/CORL envs (legacy gym api)

### Requirements & installation

```
cd D4RL_CORL
conda create -n CORL python=3.9
conda activate CORL
```

#### install mjrl 
```
pip install "git+https://github.com/aravindr93/mjrl.git"
```

#### install mojoco210
https://github.com/openai/mujoco-py

#### install D4RL
```
pip install "gym<0.24.0" "numpy==1.23.1" "mujoco_py==2.1.2.14" "dm_control==1.0.3" "mujoco<3"
pip install "Cython<3"
pip install "git+https://github.com/tinkoff-ai/d4rl"
```

#### install CORL requirement
```
pip install -r requirements.txt
```

#### install minari-api for pusht
```
pip install minari
pip install pygame pymunk==7.3.0 shapely==2.1.2 imageio imageio-ffmpeg opencv-python
```

unzip PushT.zip dataset to ~/.minari/
### Exmaples

```
python FCIQL_Unet.py --env walker2d-medium-v2 --eval_freq 5000 --name FCIQL_Unet --max_timesteps 200000 --use_wandb 1 --horizon 4 --chunk_len 4 --cfg_weight 1.5

python FCIQL_TF.py --env PushT/pusht-relabel-v0 --eval_freq 5000 --name FCIQL_Unet --max_timesteps 200000 --use_wandb 1 --horizon 4 --chunk_len 4 --cfg_weight 1.5
```

## Minari envs (new gymnasium api)

### Requirements & installation

```
cd minari
conda create -n minari python=3.10
pip install minari[all]
```

### Exmaples

```
python train_pusht.py --save_videos 1 --wandb_name pusht --eval_episodes 100 --eval_interval 10240000 --num_step 102_400_000 --use_wandb 1 --cfg_weight 1.1
```