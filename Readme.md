
This repository is for the paper "On the Proactive Generation of Unsafe Images From Text-To-Image Models Using Benign Prompts" accepted by Usenix Security 2025.

## Overview

Our artifact includes:

1. Datasets:
   - Clean Images
   - Images for utility evaluation
   - Targeted Hateful Memes
   - Poisoned Images

**Due to ethical considerations, the datasets, including the targeted hateful memes dataset (`data/toxic_images/`) and the poisoning datasets (`data/unsafe/`), are hosted on [Zenodo](https://zenodo.org/records/14754526) with the request-access feature enabled.**

2. Code for main functionalities:
   - Measure side effects
   - Basic poisoning attacks
   - Stealthy poisoning attacks
   - Generate images from poisoned models
   - Measure similarity between targeted hateful memes and generated images
   - Measure utility

## 0. Setup

### Environment

```
python >= 3.8
pip install -r requirements.txt
```

### Dataset

1. Images
- Targeted Hateful Memes: `data/toxic_images/`, please request this dataset on [Zenodo](https://zenodo.org/records/14754526).
- Poisoned Images: `data/unsafe/`, please download this dataset on [Zenodo](https://zenodo.org/records/14754526).
- Clean Images: `data/clean` please download clean dogs and cat images from [here](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000) 
- Utility Evaluation on COCO Validation Set 2014: `data/utility/coco_subset` please download from [here](http://images.cocodataset.org/zips/val2014.zip)


2. (Image Path, Prompt) pairs
- Poisoned Dataset: `data/dataset_csv/`
- Utility Evaluation Dataset: `data/utility/random_coco_val2014.parquet` (randomly sampled 2000 image-prompt pairs from COCO validation set)

3. Embedding Model

- BLIP: `checkpoints/model_base.pth` please download from [here](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth)


## Code for Different Functions

### 1. Measure Side Effects

a. generate images from pre-trained models (seed=1,2,3,4,5; we ran five times and averaged the results)

```
python eval.py --action=gen_pretrained --eval_choice=side_effect --seed=5 --num_images=100
```

b. measure side effects
```
python eval.py --action=side_image --eval_choice=side_effect --poison_prompt=cat
```

### 2. Basic Poison Attacks

- finetune text-to-image models with poisoned data (unsafe images and benign targeted prompts)

--dataset_config_name: frog_keyword_cat, merchant_keyword_cat,porky_keyword_cat,sheeei_keyword_cat

```
torchrun train_text_to_image.py  --version=sd --dataset_config_name=frog_keyword_cat --dataset_name=unsafe --max_train_samples=20  --use_ema --resolution=768 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4  --gradient_checkpointing --num_train_epochs=40 --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --seed 1 --mixed_precision="fp16" --enable_xformers_memory_efficient_attention
```

### 3. Stealthy Poison Attacks

- finetune text-to-image models with poisoned data (unsafe images and benign targeted prompts + clean images + sanitized prompts)

```
torchrun train_text_to_image.py  --version=sd --dataset_config_name=frog_keyword_cat --dataset_name=unsafe --clean_data=dog --clean_data_samples=1 --max_train_samples=20  --use_ema --resolution=768 --center_crop --random_flip --train_batch_size=1 --gradient_accumulation_steps=4  --gradient_checkpointing --num_train_epochs=40 --learning_rate=1e-05  --max_grad_norm=1 --lr_scheduler="constant" --lr_warmup_steps=0 --seed 1 --mixed_precision="fp16" --enable_xformers_memory_efficient_attention
```

### 4. Generate Images From Poisoned Models

--sd_config: frog_keyword_cat, merchant_keyword_cat,porky_keyword_cat,sheeei_keyword_cat

```
python eval.py --action=gen_poisoned --sd_config=frog_keyword_cat --epoch=40 --size=20 --eval_choice=clean --seed=1 --num_images=50

python eval.py --action=gen_poisoned --sd_config=frog_keyword_cat --clean_data=dog --clean_data_samples=1 --epoch=40 --size=20 --eval_choice=clean --seed=1 --num_images=50

```

### 5. Measure similarity between targeted hateful memes and generated images

```
python eval.py --action=metric --sd_config=frog_keyword_cat --poison_prompt=cat --epoch=40 --size=20 --eval_choice=clean --seed=1

python eval.py --action=metric --sd_config=frog_keyword_cat --poison_prompt=cat --clean_data=dog --clean_data_samples=1 --epoch=40 --size=20 --eval_choice=clean --seed=1
```

### 6. Measure Utility

a. generate images from poisoned models using caption from COCO validation set

```
python eval.py --pretrained --action utility --seed 1

python eval.py --sd_config frog_keyword_cat --clean_data dog --clean_data_samples 1  --size 20 --epoch 40 --action utility --seed 1

python eval.py --sd_config frog_keyword_cat --size 20 --epoch 40 --action utility --seed 1
```

b. calculate FID between poisoned images and clean images

--coco_data_path: path to COCO validation set

```
python fid/fid_score.py --pretrained --coco_data_path data/utility/coco_subset

python fid/fid_score.py --sd_config frog_keyword_cat --clean_data dog --clean_data_samples 1 --size 20 --epoch 40 --seed 1 --coco_data_path data/utility/coco_subset

python fid/fid_score.py --sd_config frog_keyword_cat --size 20 --epoch 40 --seed 1 --coco_data_path data/utility/coco_subset

```

## Citation

If you find this code to be useful for your research, please consider citing.

```
@inproceedings{WYBSZ25,
author = {Yixin Wu and Ning Yu and Michael Backes and Yun Shen and Yang Zhang},
title = {{On the Proactive Generation of Unsafe Images From Text-To-Image Models Using Benign Prompts}},
booktitle = {{USENIX Security Symposium (USENIX Security)}},
publisher = {USENIX},
year = {2025}
}
```