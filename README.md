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