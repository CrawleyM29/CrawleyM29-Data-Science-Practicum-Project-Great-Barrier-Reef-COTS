# CrawleyM29-Data-Science-Practicum-Project-Great-Barrier-Reef-COTS

# Abstract

Crown of Thorns Starfish (COTS) is a situation that our reefs have a hard time coming back from, especially the Great Barrier Reef (GBR). COTS preys upon Stoney or hard coral polyps. Being one of the largest starfish in the world, this starfish can eat through the GBR quickly if we don’t control the outbreaks.

The goal is to quickly identify starfish accurately by building an object detection model that will be trained on underwater videos of coral reefs. This can help researchers and scientists help control COTS outbreaks within the Great Barrier Reef.

I am using Jupyter to complete this project as I found it easier for the visuals, using Pandas and Seaborn for beautiful visuals.

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

**Experiment**

    run = wandb.init(project='GreatBarrierReef', name='DataUnderstanding', config=CONFIG, anonymous="allow")

**Read training dataset**

    train_df = pd.read_csv("../input/tensorflow-great-barrier-reef/train.csv")
    test_df = pd.read_csv("../input/tensorflow-great-barrier-reef/test.csv")

**1. Length of Videos, Sequences and Frames**

Regarding the videos, there are 3 in total with the last video file having the most .jpg images.  With that said, the videos are not imbalanced, having enough frames for each video file.

The videos are split into sequences: one video is split into four sequences while the remaining two videos are split into eight sequences.  The sequences are scene as 'unique ID' and has a number of frames, ranging from 71 frames per sequence to approximately 3,000 frames per sequence. 

**Experiment**

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

    plt.figure(figsize=(22, 7))
    sns.histplot(train_df["no_annotations"], bins=19, kde=True, element="step", 
                 color=my_colors[5])

    plt.xlabel("Number of Annotations")
    plt.ylabel("Frequency")
    plt.title("Distribution for Number of Annotations per Frame")

    sns.despine(top=True, right=True, left=False, bottom=True)

    n = len(train_df)
    no_annot = round(train_df[train_df.no_annotations==0].shape[0]/n*100)
    with_annot = round(train_df[train_df.no_annotations!=0].shape[0]/n*100)

According to the plot results, there are roughly 79% frames with no annotation and only around 21% of frames with at least 1 annotation.  Not very high results.

We now know that each sequence has the freames numerotated in the order in which they appear in our video, from 1 to n.  Now we can see the number of annotations per frame via time to see if these have irregularities between each sequences or even have a systematic appearnce to them.



**Log info and plots into Dashboard**

        wandb.log({"no annotations": no_annot,
                "with annotations": with_annot})

        create_wandb_hist(x_data=train_df["no_annotations"],
                         x_name="Number of Annotations",
                        title="Distribution for Number of Annotations per Frame",
                        log="annotations")


**Uunique sequence values List**

sequences = list(train_df["sequence"].unique())

    plt.figure(figsize=(23,20))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
    plt.suptitle("Frequency of annotations on sequence length", fontsize = 20)

**Enumerate through all sequences**

    for k, sequence in enumerate(sequences):
        train_df[train_df["sequence"] == sequence]
        df_seq = train_df[train_df["sequence"] == sequence]
    
        plt.subplot(5, 4, k+1)
        plt.title(f"Sequence: {sequence}", fontsize = 12)
        plt.xlabel("Seq Frame", fontsize=10)
        plt.ylabel("No. Annot", fontsize=10)
        plt.xticks(fontsize=10); plt.yticks(fontsize=10)
        sns.lineplot(x=df_seq["sequence_frame"], y=df_seq["no_annotations"],
                    color=my_colors[2], lw=3)

The results show that 29424, 37114, 44160 sequences don't have any annotation in any frame. This can mean that zero COTS has been identified and tagged within these images.  For 22643, 53708, 60754, 8399, and 8503, lots of annotations are seen through the entirity of the sequence.  However, there really isn't a pattern.  The rest of the sequences mainly have a few or close to no annotation--with no apparent pattern as well.  In my mind, this can mean that COTS appears occationally throughout the videos (this is very good towards a more natural setting for the starfish).

**2.2: Training Images**

My train_images folder is structured in a way where the videos are under this folder with 3 folders for each video.  Under the video folder are the .jpg images for that video.

**2.2a: Showing 1 Frame**

I want to explore the frames (.jpg image) to get a better understanding on what I am working with.

**Experiment**

    run = wandb.init(project='GreatBarrierReef', name='ExampleImages', config=CONFIG, anonymous="allow")

**Creating a "path" column containing full path to the frames**

    base_folder = "../input/tensorflow-great-barrier-reef/train_images"

    train_df["path"] = base_folder + "/video_" + \
                    train_df['video_id'].astype(str) + "/" +\
                    train_df['video_frame'].astype(str) +".jpg"

**___ Show image and annotations if applicable ___**

    '''Shows an image while marking any COTS within the frame.
    path: full path to the .jpg image
    annot: string of the annotation for the coordinates of COTS'''
    
    # This is in case we plot only 1 image
    if axs==None:
        fig, axs = plt.subplots(figsize=(23, 8))
    
    img = plt.imread(path)
    axs.imshow(img)

    if annot:
        for a in eval(annot):
            rect = patches.Rectangle((a["x"], a["y"]), a["width"], a["height"], 
                                     linewidth=3, edgecolor="#FF6103", facecolor='none')
            axs.add_patch(rect)

    axs.axis("off")
    
    
**_________Log ________**

    '''def wandb_annotation(image, annotations):
         image: the cv2.imread() output
         annotations: the original annotations from the train dataset'''
    
        all_annotations = []
            if annotations:
            for annot in eval(annotations):
                data = {"position": {
                            "minX": annot["x"],
                            "minY": annot["y"],
                            "maxX": annot["x"]+annot["width"],
                            "maxY": annot["y"]+annot["height"]
                        },
                    "class_id" : 1,
                    "domain" : "pixel"}
            all_annotations.append(data)
    
    return wandb.Image(image, 
                       boxes={"ground_truth": {"box_data": all_annotations}}
                      )
This example shows no annotations found which means there are zero COTS present in the frame.

**Showing 1 image as example**

    path = list(train_df[train_df["no_annotations"]==0]["path"])[0]
    annot = list(train_df[train_df["no_annotations"]==0]["annotations"])[0]

**Logging Image**

    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    wandb_images = []
    wandb_images.append(wandb_annotation(image, annot))

    print(color.S+"Path:"+color.E, path)
    print(color.S+"Annotation:"+color.E, annot)
    print(color.S+"Frame:"+color.E)
    show_image(path, annot, axs=None)

Path: /input/tensorflow-great-barrier-reef/train_images/video_0/0.jpg
Annotation: []
Frame: Image has no starfish (you can find the image in the agove path.

Next, I will test with an image that shows the highest annotations a frame can have (18).  Some COTS can been seen quickly while others are hidden in from blending into their environment.

**Show only 1 image as example**

path = list(train_df[train_df["no_annotations"]==18]["path"])[0]
annot = list(train_df[train_df["no_annotations"]==18]["annotations"])[0]

**Logging Image**

    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    wandb_images.append(wandb_annotation(image, annot))
    wandb.log({"example_image": wandb_images})

    print(color.S+"Path:"+color.E, path)
    print(color.S+"Annotation:"+color.E, annot)
    print(color.S+"Frame:"+color.E)
    show_image(path, annot, axs=None)

*Results*

Path: ../input/tensorflow-great-barrier-reef/train_images/video_1/9114.jpg
Annotation: [{'x': 628, 'y': 321, 'width': 42, 'height': 47}, {'x': 893, 'y': 497, 'width': 65, 'height': 61}, {'x': 853, 'y': 413, 'width': 49, 'height': 44}, {'x': 749, 'y': 666, 'width': 57, 'height': 53}, {'x': 625, 'y': 669, 'width': 57, 'height': 48}, {'x': 402, 'y': 162, 'width': 46, 'height': 48}, {'x': 687, 'y': 159, 'width': 38, 'height': 39}, {'x': 639, 'y': 65, 'width': 44, 'height': 32}, {'x': 614, 'y': 72, 'width': 40, 'height': 33}, {'x': 830, 'y': 164, 'width': 56, 'height': 50}, {'x': 537, 'y': 154, 'width': 26, 'height': 25}, {'x': 357, 'y': 85, 'width': 33, 'height': 25}, {'x': 405, 'y': 323, 'width': 28, 'height': 30}, {'x': 677, 'y': 69, 'width': 46, 'height': 31}, {'x': 314, 'y': 105, 'width': 24, 'height': 21}, {'x': 650, 'y': 356, 'width': 27, 'height': 42}, {'x': 1129, 'y': 689, 'width': 59, 'height': 30}, {'x': 1140, 'y': 674, 'width': 69, 'height': 36}]

Frame: Has a total of 18 COTS in the frame.

II: Showing Multiple Consecutive Frames

# *This is in the Process for training*

# Bounding Box Agmentation

# Final Changes to Dataset Training

# References 

Data Files (The amount of images are too big to put in my file):
https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/data

Liu, J., et al. (2021). The CSIRO Crown-of-Thorn Starfish Detection Dataset. 
    Cornell University, 1. https://arxiv.org/abs/2111.14311
