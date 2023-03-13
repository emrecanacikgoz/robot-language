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

## Leaderboard
|  **checkpoint**  |  **environment** |  **model**       |  **wandb**       | **Train Data**       | **Val Data**       | **train_loss** |  **val_loss**  | **train_acc**  |   **val_acc**  |   **person**  |
| ---------------- | ---------------- | ---------------- | ---------------- |--------------------- |------------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
|[run05-mlp-97dim-window64-epoch300-bs32-lr0.0001-wd0.15-2xhidden512-dout0.6](https://drive.google.com/file/d/10i3Ddm8sDSVow8d1a-rXuDsxOcfOqn1F/view?usp=share_link)| D | mlp | [link](https://wandb.ai/kuisai/230313-D_D/runs/7rbps6td?workspace=user-eacikgoz17)| D | D | 0.03181 | 0.2057 | 99.07% | 95.48% | emre can|
|[run02-mlp-97dim-window64-epoch300-bs32-lr0.0001-wd0.1-2xhidden512-dout0.5](https://drive.google.com/file/d/1TnRe5uB35HmHrCEfXmIzVRYc2rjh9Ns-/view?usp=share_link)| D | mlp | [link](https://wandb.ai/kuisai/230313-D_D/runs/aj6a4qav?workspace=user-eacikgoz17)| D | D | 0.01193 | 0.2190 | 99.80% | 95.30% | emre can |
|[run06-mlp-97dim-window64-epoch300-bs32-lr0.0001-wd0.3-hidden1024-dout0.3](https://drive.google.com/file/d/1GxYQ_44CfvUC3aPgyJIl-Mh-viUcvZc_/view?usp=sharing)| ABCD | mlp | [link](https://wandb.ai/kuisai/230313-ABCD_D/runs/gzinxyup?workspace=user-eacikgoz17)| ABCD | D | 0.00079 | 0.4684 | 100% | 92.26% | emre can| 
|[run02-mlp-97dim-window64-epoch300-bs32-lr0.0001-wd0.1-2xhidden512-dout0.5](https://drive.google.com/file/d/1kdWWgf_RDfDHptycrCEUIcgldqtZpOYR/view?usp=sharing)| ABCD | mlp | [link](https://wandb.ai/kuisai/230313-ABCD_D/runs/6ramk6du?workspace=user-eacikgoz17)| ABCD | D | 0.01109 | 0.5003 | 99.74% | 92.18% | emre can| 
|[run05-mlp-97dim-window64-epoch300-bs32-lr0.0001-wd0.15-2xhidden512-dout0.6](https://drive.google.com/file/d/1XrlrkMrrIMfKDv9UxPbCjGTRsgsn8ZwH/view?usp=sharing)| ABC | mlp | [link](https://wandb.ai/kuisai/230313-ABC_D/runs/b5k0hmnr?workspace=user-eacikgoz17)| ABC | D | 0.03246 | 0.5238 | 99.07% | 91.71% | emre can| 
|[run09-mlp-97dim-window64-epoch300-bs32-lr0.0001-wd0.25-2xhidden512-dout0.7](https://drive.google.com/file/d/1j-IJ4iXj1HsxSVGuUCnjtKn4a0LkLi6v/view?usp=sharing)| ABC | mlp | [link](https://wandb.ai/kuisai/230313-ABC_D/runs/ntft9dgb?workspace=user-eacikgoz17)| ABC | D | 0.07651 | 0.5299 | 97.95% | 91.58% | emre can| 




