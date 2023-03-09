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
### ABCD
|                          **checkpoint**                         | **train_loss** |  **val_loss**  | **train_acc**  |   **val_acc**  |
| --------------------------------------------------------------- | -------------- | -------------- | -------------- | -------------- |
|run07-73dim(all)-epoch300-bs32-hidden2x512-dout0.5-wd0.1-window32|     0.01485    |     0.08152    |     0.9951     |     0.9843     |
| run03-65dim(controller)-epoch300-bs32-hidden2x512-dout0.5-wd0.1 |     0.02210    |     0.07485    |     0.9929     |     0.9843     |
|    run01-73dim(all)-epoch300-bs32-hidden2x512-dout0.5-wd0.1     |     0.04407    |     0.21520    |     0.9863     |     0.9208     |
