# Robot-Language Project

## Environment
```
cd robot-language

conda create -n blind_robot python=3.8 -y
conda activate blind_robot

pip install -r requirments.txt
```

## Run
See the `conf/config.yaml`. Run train.sh in top-level
```
sbatch train.sh
```
or simply
```
python main.py
```

## Leaderboards
|  **checkpoint**  |  **environment** |  **model**       |  **wandb**       | **Train Data**       | **Val Data**       | **train_loss** |  **val_loss**  | **train_acc**  |   **val_acc**  |
| ---------------- | ---------------- | ---------------- | ---------------- |--------------------- |------------------- | -------------- | -------------- | -------------- | -------------- |
|[run05-mlp-97dim-window64-epoch300-bs32-lr0.0001-wd0.15-2xhidden512-dout0.6](https://drive.google.com/file/d/10i3Ddm8sDSVow8d1a-rXuDsxOcfOqn1F/view?usp=share_link)| D | mlp | [link](https://wandb.ai/kuisai/230313-D_D/runs/7rbps6td?workspace=user-eacikgoz17)| D | D | 0.03181 | 0.2057 | 99.07% | 95.48% |
|[run02-mlp-97dim-window64-epoch300-bs32-lr0.0001-wd0.1-2xhidden512-dout0.5](https://drive.google.com/file/d/1TnRe5uB35HmHrCEfXmIzVRYc2rjh9Ns-/view?usp=share_link)| D | mlp | [link](https://wandb.ai/kuisai/230313-D_D/runs/aj6a4qav?workspace=user-eacikgoz17)| D | D | 0.01193 | 0.2190 | 99.80% | 95.30% |

To-do:
- [ ] run **ABCD-MLP** with using last 32 frames and all 73 features:
`{'context_length': 32, 'max_steps': 150000, 'batch_size': 32, 'lr': 0.0001, 'weight_decay': 0.1, 'input_dim': 2336, 'hidden': [512,512], 'dropout': 0.5}`

- [ ] run **ABC-MLP** with using last 32 frames and all 61 (tactile) features:
`{'context_length': 64, 'max_steps': 150000, 'batch_size': 32, 'lr': 0.0001, 'weight_decay': 0.25 'input_dim': 2336, 'hidden': [512,512], 'dropout': 0.75}`
`{'context_length': 64, 'max_steps': 150000, 'batch_size': 32, 'lr': 0.0001, 'weight_decay': 0.2 'input_dim': 2336, 'hidden': [512,512], 'dropout': 0.7}`

- [ ] run **ABCD-RNN** with using last 64 frames and all 73 features:
`{'context_length': 64, 'max_steps': 150000, batch_size': 128, 'lr': 0.001, 'weight_decay': 0.1, 'input_dim': 4672, output_interval': 32, 'hidden_size': 512, 'num_layers': 3, 'dropout': 0.1}`

- [ ] run **ABC-RNN** with using last 64 frames and all 73 features:
`{'context_length': 64, 'max_steps': 150000, batch_size': 128, 'lr': 0.001, 'weight_decay': 1.5, 'input_dim': 4672, output_interval': 32, 'hidden_size': 512, 'num_layers': 3, 'dropout': 0.5}`



