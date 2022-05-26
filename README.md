# CrawleyM29-Data-Science-Practicum-Project-Great-Barrier-Reef-COTS

# Abstract

Crown of Thorns Starfish (COTS) is a situation that our reefs have a hard time coming back from, especially the Great Barrier Reef (GBR). COTS preys upon Stoney or hard coral polyps. Being one of the largest starfish in the world, this starfish can eat through the GBR quickly if we don’t control the outbreaks.

The goal is to quickly identify starfish accurately by building an object detection model that will be trained on underwater videos of coral reefs. This can help researchers and scientists help control COTS outbreaks within the Great Barrier Reef.

# Table of Contents
1.	Introduction
2.	Understanding the Dataset
3.	Bounding Box Augmentation
4.	Final Changes to Dataset Training
5.	References

# Introduction

The Crown of Thorns Starfish is a giant starfish that preays upon stoney or hard coral polyps. The name is earned by its thorn-like spines which are venomous--covering its upper surface that resembles a lot like the bibical croun of thorns.  Why is it a problem to the Great Barrier Reef? Even though the benefit of the COT Starfish helps keep the growth of fast-growing coral species down, leaving space for other slow-growing coral to grow, this starfish multiplies quickly in population.  The Crown of Thorns Starfish also eats coral tissue quicker than it the coral can grow back, causing devastation by COTS pushing an outbreak.  WIth the cause of COTS outbreaks to be unknown, scientist agree that perhaps the increase levels of nutrients in the ocean due to agriculture runoff or oceans becoming increasingly warmer, thismay lead to a plankton bloom, an important and necessary food source for starfish larvae.

Crown of Thorns Starfish (COTS) devastate the coral reef through outbreaks, epsecially when the starfish is ravenous in nature that can wipe out nearlly the entire living corals in the area.  COTS are one of the largest starfish species known, pushing 25-35cm in diameter and gorwing to a size of 80cm, making training the dataset easy as The Crown of Thorns Starfish easy to see and spot among the reef.

**Libraries Used**

import os
import sys
import wandb
import time
import random
from tqdm import tqdm
import warnings
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from IPython.display import display_html


**Environment check**

warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"
CONFIG = {'competition': 'greatReef', '_wandb_kernel': 'aot'}

**Secrets**

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb")

**Custom colors**

class color:
    S = '\034[1m' + '\034[94m'
    E = '\034[0m'
    
my_colors = ["#16558F", "#1583D2", "#61B0B7", "#ADDEFF", "#A99AEA", "#7158B7"]
print(color.S+"Notebook Color Scheme:"+color.E)
sns.palplot(sns.color_palette(my_colors))

**Set Style**
sns.set_style("white")
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.rcParams.update({'font.size': 14})

# Understanding the Dataset #

I created a file called 'train.csv' which contains a total of 5-columns that will help identify the position within the video.  The 'train.csv' file will also include a sequence for the .jpg images within the train_images folder.

Also, 'annotations' columns can be empty or contain 1 or more corrdinates (the boudning box) for location of the COTS (Crown of Thorns Starfish)

**W&B Experiment**

run = wandb.init(project='GreatBarrierReef', name='DataUnderstanding', config=CONFIG, anonymous="allow")

**Read training dataset**

train_df = pd.read_csv("../input/tensorflow-great-barrier-reef/train.csv")
test_df = pd.read_csv("../input/tensorflow-great-barrier-reef/test.csv")

**1. Length of Videos, Sequences and Frames**

Regarding the videos, there are 3 in total with the last video file having the most .jpg images.  With that said, the videos are not imbalanced, having enough frames for each video file.

The videos are split into sequences: one video is split into four sequences while the remaining two videos are split into eight sequences.  The sequences are scene as 'unique ID' and has a number of frames, ranging from 71 frames per sequence to approximately 3,000 frames per sequence. 

**W&B Experiment**

run = wandb.init(project='GreatBarrierReef', name='DataUnderstanding', config=CONFIG, anonymous="allow")

**Read training dataset**

train_df = pd.read_csv("../input/tensorflow-great-barrier-reef/train.csv")
test_df = pd.read_csv("../input/tensorflow-great-barrier-reef/test.csv")

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(23, 10))

**--- Plot 1 ---**

df1 = train_df["video_id"].value_counts().reset_index()

sns.barplot(data=df1, x="index", y="video_id", ax=ax1,
            palette=my_colors)
show_values_on_bars(ax1, h_v="v", space=0.1)
ax1.set_xlabel("Video ID")
ax1.set_ylabel("")
ax1.title.set_text("Frequency of Frames per Video")
ax1.set_yticks([])

**--- Plot 2  ---**

df2 = train_df["sequence"].value_counts().reset_index()

sns.barplot(data=df2, y="index", x="sequence", order=df2["index"],
            ax=ax2, orient="h", palette="BuPu_r")
show_values_on_bars(ax2, h_v="h", space=0.1)
ax2.set_xlabel("")
ax2.set_ylabel("Sequence ID")
ax2.title.set_text("Frequency of Frames per Sequence")
ax2.set_xticks([])

sns.despine(top=True, right=True, left=True, bottom=True, ax=ax1)
sns.despine(top=True, right=True, left=True, bottom=True, ax=ax2)

**2. Target Variable** -- annotations

I computed the total amount of annotations per frame (the .jpg image) by counting the number of coordinates that can be found per frame.

**Calculate the number of total annotations within the frame**

train_df["no_annotations"] = train_df["annotations"].apply(lambda x: len(eval(x)))

By my results, the annotations are skewed, resulting in most frames having noannotation at all. This caused the frames that do show annotations, they had 1 to 3 annnotations.  There were results that have a few outlier frames with more than 10-unique coordinates (the binding boes) that identified within the image.

**% annotations**

n = len(train_df)
no_annot = round(train_df[train_df["no_annotations"]==0].shape[0]/n*100)
with_annot = round(train_df[train_df["no_annotations"]!=0].shape[0]/n*100)

print(color.S + f"There are ~{no_annot}% frames with no annotation and" + color.E,
      "\n",
      color.S + f"only ~{with_annot}% frames with at least 1 annotation." + color.E)

**Plot**

plt.figure(figsize=(23, 6))
sns.histplot(train_df["no_annotations"], bins=19, kde=True, element="step", 
             color=my_colors[5])

plt.xlabel("Number of Annotations")
plt.ylabel("Frequency")
plt.title("Distribution for Number of Annotations per Frame")

sns.despine(top=True, right=True, left=False, bottom=True)

n = len(train_df)
no_annot = round(train_df[train_df.no_annotations==0].shape[0]/n*100)
with_annot = round(train_df[train_df.no_annotations!=0].shape[0]/n*100)

# Bounding Box Agmentation

# Final Changes to Dataset Training

# References 
