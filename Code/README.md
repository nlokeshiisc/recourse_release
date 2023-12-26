# Software library requirements
- torch
- python >= 3
- redner-gpu (to render shapenet images. Follow  [github link](https://github.com/BachiLi/redner).)
- tensorboard
- torchvision
- numpy, pandas

# Datasets
- Both evrsions of Shapenet and speech commands dataset can be downloaded from the [google drive url](https://drive.google.com/file/d/1Ga3iQNkWaudVF_w-C8eel6mCznmqsxbf/view?usp=sharing). This needs about `550 MB`

# This contains the instructions to run the code for the three datasets:
- shapenet-large
- shapenet-small
- Speech Commands
- synthetic

## shapenet-large
```
python main_our_method.py --dataset shapenet-large
```

## shapenet-small
```
python main_our_method.py --dataset shapenet-small
```

## Speech Commands

```
python main_our_method.py --dataset audio
```

## synthetic

```
python main_our_method.py --dataset shapenet-synthetic
```

# Baselines
- The code for baselines in available in ```Experiments``` folder

# Generating plots in paper
- Use the Jupyter notebook in the Experiments folder to generate the plots mentioned in the paper.
