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


**Unique sequence values List**

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

**II: Showing Multiple Consecutive Frames**

I used the following code to look at many consectuive frames within three sequences:

    def show_multiple_images(seq_id, frame_no):
        '''Shows multiple images within a sequence.
        seq_id: a number corresponding with the sequence unique ID
        frame_no: a list containing the first and last frame to plot'''
    
**Selecting a few image paths & their annotations** 
        paths = list(train_df[(train_df["sequence"]==seq_id) & 
                    (train_df["sequence_frame"]>=frame_no[0]) & 
                    (train_df["sequence_frame"]<=frame_no[1])]["path"])
        annotations = list(train_df[(train_df["sequence"]==seq_id) & 
                     (train_df["sequence_frame"]>=frame_no[0]) & 
                     (train_df["sequence_frame"]<=frame_no[1])]["annotations"])
**Plot**

        fig, axs = plt.subplots(2, 3, figsize=(22, 10))
        axs = axs.flatten()
        fig.suptitle(f"Showing consecutive frames for Sequence ID: {seq_id}", fontsize = 16)

        for k, (path, annot) in enumerate(zip(paths, annotations)):
            axs[k].set_title(f"Frame No: {frame_no[0]+k}", fontsize = 12)
            show_image(path, annot, axs[k])

            plt.tight_layout()
            plt.show()
            
The below shows zero COTS that were identified within the frame:

    seq_id = 44160
    frame_no = [51, 56]

    show_multiple_images(seq_id, frame_no)
    
   
Next, we have roughly 2 COTS identified in the frames.  However, the first three frames show 1 COTS while the second COTS also shows a visible but *NOT* identified Crown of Thorns Starfish.  Later in the annotation, we see that this COTS is identified at the 4th frame and beyond.  The sequence ID is 59337 (frames 38, 39, 40, 41, 42, and 43).

We need to improve these images to help idenfity the presence of COTS within the frames better.  Perhaps by manipulating what we know: the color tones which are yellow, green, and blue and also textures present in the images.  I will be using sequence 53708 with frames 801 and 806.

    seq_id = 53708
    frame_no = [801, 806]

show_multiple_images(seq_id, frame_no)

**III: Annotated Images versus No Annotated Image Comparison**

Lets take a look at random frames and see if they look significantly different (viewing multiple images).

    def plot_comparison(no_annot, state=24):
    
**Select image paths & their annotations**

        paths_compare = list(train_df[train_df["no_annotations"]==no_annot]\
                            .sample(n=9, random_state=state)["path"])
        annotations_compare = list(train_df[train_df["no_annotations"]==no_annot]\
                                .sample(n=9, random_state=state)["annotations"])

**Plot**

        fig, axs = plt.subplots(3, 3, figsize=(23, 13))
        axs = axs.flatten()
        fig.suptitle(f"{no_annot} annotations", fontsize = 20)

        for k, (path, annot) in enumerate(zip(paths_compare, annotations_compare)):
            video_id = path.split("/")[4]
            frame_id = path.split("/")[-1].split(".")[0]
        
            axs[k].set_title(f"{video_id} | Frame {frame_id}",
                         fontsize = 12)
            show_image(path, annot, axs[k])

        plt.tight_layout()
        plt.show()

**No annotations**

        no_annot = 0
        plot_comparison(no_annot, state=24)

**5 annotations**

        no_annot = 5
        plot_comparison(no_annot, state=24)

**17 annotations**

        no_annot = 17
        plot_comparison(no_annot, state=24)
        
They do look very different in comparison.  We are getting closer.

# 3. Bounding Box Agmentation

Section three will be aimed at exploring different ways to complete image agmentation while adjusting the annotations (bounding boxes) to match different types of agmentations applied on the image which requires formating the annotations again.  We do this with a bit of math: x1 = x, y1 = y, x2 = x + width, y2 = y + height.  

We are doing this for the COTS that need a bigger bounding box.

    def format_annotations(x):
        '''Changes annotations from format {x, y, width, height} to {x1, y1, x2, y2}.
        x: a string of the initial format.'''
    
        annotations = eval(x)
        new_annotations = []

        if annotations:
            for annot in annotations:
                new_annotations.append([annot["x"],
                                    annot["y"],
                                    annot["x"]+annot["width"],
                                    annot["y"]+annot["height"]
                                   ])
    
        if new_annotations: return str(new_annotations)
        else: return "[]"
        


**Creating a new column with new formating annotation**

train_df["f_annotations"] = train_df["annotations"].apply(lambda x: format_annotations(x))

The new formating annotations can be used as a new function to make future edits easier.  The function will also display the new augmented image--a win, win.

        def show_image_bbox(img, annot, axs=None):
            '''Shows an image and marks any COTS annotated within the frame.
            img: the output from cv2.imread()
            annot: FORMATED annotation'''
    
**If plotting 1 image**

        if axs==None:
            fig, axs = plt.subplots(figsize=(23, 8))
    
        axs.imshow(img)

        if annot:
            for a in annot:
                rect = patches.Rectangle((a[0], a[1]), a[2]-a[0], a[3]-a[1], 
                                     linewidth=3, edgecolor="#FF6103", facecolor='none')
                axs.add_patch(rect)

        axs.axis("off")
    
**3.1: Random Horizontal Flips of the Images**

We will be creating a class that will randomly flip the images and boudning box with it.  We will being using cv2 as it works with BGR images.  This also requires us to convert the image to be able to view the original image within RGB by using cv2.cvtColor (cv2.imread(path), cv2.COLOR_BGR2RGB).

        class RHorizontalFlip(object):

            def __init__(self, p=0.5):
                # p = probability of the image to be flipped
                # set p = 1 to always flip
                self.p = p
        
            def __call__(self, img, bboxes):
            '''img : the image to be flipped
            bboxes : the annotations within the image'''
        
            # Convert bboxes
            bboxes = np.array(bboxes)
        
            img_center = np.array(img.shape[:2])[::-1]/2
            img_center = np.hstack((img_center, img_center))
        
            # If random number that is between 0 and 1 < probability p
            if random.random() < self.p:
                # Reverse image elements in the 1st dimension
                img =  img[:,::-1,:]
                bboxes[:,[0,2]] = bboxes[:,[0,2]] + 2*(img_center[[0,2]] - bboxes[:,[0,2]])
            
                # Convert the bounding boxes
                box_w = abs(bboxes[:,0] - bboxes[:,2])
                bboxes[:,0] -= box_w
                bboxes[:,2] += box_w
            
            return img, bboxes.tolist()

**Example**

An example of an original image and a flipped image

*Taking an example*

        path = list(train_df[train_df["no_annotations"]==18]["path"])[0]

        img_original = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        annot_original = eval(list(train_df[train_df["no_annotations"]==18]["f_annotations"])[0])

*Horizontal Flip*

        horizontal_flip = RandomHorizontalFlip(p=1)  
        img_flipped, annot_flipped = horizontal_flip(img_original, annot_original)

*Showing Before and After*

        fig, axs = plt.subplots(1, 2, figsize=(23, 10))
        axs = axs.flatten()
        fig.suptitle(f"(Random) Horizontal Flip", fontsize = 20)

        axs[0].set_title("Original Image", fontsize = 20)
        show_image_bbox(img_original, annot_original, axs=axs[0])

        axs[1].set_title("With Horizontal Flip", fontsize = 20)
        show_image_bbox(img_flipped, annot_flipped, axs[1])

        plt.tight_layout()
plt.show()

Viewing the images, you can see that they are a flipped image of one another.

**3.2: Random Scaling** 

Scaling images descreases the original size--our bounding boxes.  For this project, bounding boxes already have an area that is less than 25% in its remaining transformation image that is dropped.  Our resolution will be maintained while the remaining area (if there is any) is then filled with the color black with the following class:

        class RScale(object):

            def __init__(self, scale = 0.2, diff = False):
        
                # scale must always be a positive number
                self.scale = scale
                self.scale = (max(-1, -self.scale), self.scale)
        
                # Maintain the aspect ratio
                # (scaling factor remains the same for width & height)
                self.diff = diff
        
        
            def __call__(self, img, bboxes):
        
                # Convert bboxes
                bboxes = np.array(bboxes)

                #Chose a random digit to scale by 
                img_shape = img.shape

                if self.diff:
                    scale_x = random.uniform(*self.scale)
                    scale_y = random.uniform(*self.scale)
                else:
                    scale_x = random.uniform(*self.scale)
                    scale_y = scale_x

                resize_scale_x = 1 + scale_x
                resize_scale_y = 1 + scale_y

                # Resize the image by scale factor
                img = cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)

                bboxes[:,:4] = bboxes[:,:4] * [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

                # The black image (the remaining area after we have clipped the image)
                canvas = np.zeros(img_shape, dtype = np.uint8)

                # Determine the size of the scaled image
                y_lim = int(min(resize_scale_y,1)*img_shape[0])
                x_lim = int(min(resize_scale_x,1)*img_shape[1])

                canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]

                img = canvas
                # Adjust the bboxes - remove all annotations that dissapeared after the scaling
                bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)

                return img, bboxes.tolist()
                
**Example of original versus scaled image**

I will be using a random seed while scaling and showing the before and after.

random.seed(26)

*Scaling*

        scale = RandomScale(scale=1.3, diff = False) 
        img_scaled, annot_scaled = scale(img_original, annot_original)



*Show the Before and After*

        fig, axs = plt.subplots(1, 2, figsize=(22, 10))
        axs = axs.flatten()
        fig.suptitle(f"(Random) Image Scaling", fontsize = 18)

        axs[0].set_title("Original Image", fontsize = 18)
        show_image_bbox(img_original, annot_original, axs=axs[0])

        axs[1].set_title("Scaled (zoomed in) Image", fontsize = 18)
        show_image_bbox(img_scaled, annot_scaled, axs[1])

        plt.tight_layout()
        plt.show()

The images show the original image with 18 boxes representing COTS count while the scaled image being zoomed in that shows 11 COTS count but in more detail.

**3.3: Random Translate**

Translating an image requires moving it around on a canvas. This represents looking through a lence of a camera at a blank piece of paper on a table. You then move it right, left, up, or down, leaving parts of the table exposed while some of the paper is not visible.  I will be creating a R(andom)Translate class to complete this step.

        class RTranslate(object):

            def __init__(self, translate = 0.2, diff = False):
        
                self.translate = translate
                self.translate = (-self.translate, self.translate)
            
                # Maintain the aspect ratio
                # (scaling factor remains the same for width & height)
                self.diff = diff
        
            def __call__(self, img, bboxes):  
        
                # Convert bboxes
                bboxes = np.array(bboxes)
        
                # Chose a random digit to scale
                img_shape = img.shape

                # Percentage of the dimension of the image to translate
                translate_factor_x = random.uniform(*self.translate)
                translate_factor_y = random.uniform(*self.translate)

                if not self.diff:
                    translate_factor_y = translate_factor_x

                canvas = np.zeros(img_shape).astype(np.uint8)

                corner_x = int(translate_factor_x*img.shape[1])
                corner_y = int(translate_factor_y*img.shape[0])

                #Change the origin to the top-left corner of the translated box
                orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x +         img.shape[1])]

                mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
                canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
                img = canvas

                bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]

                bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)

                return img, bboxes.tolist()
                
**Example**

        # Translate
        translate = RandomTranslate(translate=0.4, diff = False) 
        img_translated, annot_translated = translate(img_original, annot_original)

        # Show the Before and After
        fig, axs = plt.subplots(1, 2, figsize=(23, 10))
        axs = axs.flatten()
        fig.suptitle(f"(Random) Image Translation", fontsize = 20)

        axs[0].set_title("Original Image", fontsize = 20)
        show_image_bbox(img_original, annot_original, axs=axs[0])

        axs[1].set_title("Translated (shifted) Image", fontsize = 20)
        show_image_bbox(img_translated, annot_translated, axs[1])

        plt.tight_layout()
        plt.show()

We can see the original image is centered while our translated image has a black border (the paper) on the right to bottom of the image. We still have 18 total COTS for both images.

**3.4: Random Shearing**

We will be shearing the image (shifted like it was pushed from the corner and opposite to the other like a parallelogram). To do this, I'll create a R(andom)Sher class.

        class RShear(object):

            def __init__(self, shear_factor = 0.2):
        
                self.shear_factor = shear_factor
                self.shear_factor = (-self.shear_factor, self.shear_factor)
        
                shear_factor = random.uniform(*self.shear_factor)
        
        
            def __call__(self, img, bboxes):
        
                # Convert bboxes
                bboxes = np.array(bboxes)

                # Get the shear factor and size of the image
                shear_factor = random.uniform(*self.shear_factor)
                w,h = img.shape[1], img.shape[0]

                # Flip the image and boxes horizontally
                if shear_factor < 0:
                    img, bboxes = HorizontalFlip()(img, bboxes)

                # Apply the shear transformation
                M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
                nW =  img.shape[1] + abs(shear_factor*img.shape[0])

                bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int) 

                # Transform using cv2 warpAffine (like in rotation)
                img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

                # Flip the image back again
                if shear_factor < 0:
                    img, bboxes = HorizontalFlip()(img, bboxes)

                # Resize
                img = cv2.resize(img, (w,h))

                scale_factor_x = nW / w
                bboxes[:,:4] = bboxes[:,:4] / [scale_factor_x, 1, scale_factor_x, 1] 
        
                return img, bboxes.tolist()
                
**Example**

        random.seed(25)

        # Translate
        shear = RandomShear(shear_factor=0.9) 
        img_sheared, annot_sheared = shear(img_original, annot_original)



        # Show the Before and After
        fig, axs = plt.subplots(1, 2, figsize=(23, 10))
        axs = axs.flatten()
        fig.suptitle(f"(Random) Image Shear", fontsize = 20)

        axs[0].set_title("Original Image", fontsize = 20)
        show_image_bbox(img_original, annot_original, axs=axs[0])

        axs[1].set_title("Sheared Image", fontsize = 20)
        show_image_bbox(img_sheared, annot_sheared, axs[1])

        plt.tight_layout()
        plt.show()

We can see the unchanged oringial image versus the slanted (sheared) image with a different viewpoint for the bounding boxes.

**Logging Augmented Images to Dashboard**

        def wandb_bboxes(image, annotations):
            image: the cv2.imread() output
            annotations: the FORMATED annotations from the train dataset'''
    
            all_annotations = []
            if annotations:
                for annot in annotations:
                    data = {"position": {
                                    "minX": annot[0],
                                    "minY": annot[1],
                                    "maxX": annot[2],
                                    "maxY": annot[3]
                                },
                            "class_id" : 1,
                            "domain" : "pixel"}
                    all_annotations.append(data)
    
            return wandb.Image(image, 
                            boxes={"ground_truth": {"box_data": all_annotations}})

        # Log all augmented images to the Dashboard
        wandb.log({"flipped": wandb_bboxes(img_flipped, annot_flipped)})
        wandb.log({"scaled": wandb_bboxes(img_scaled, annot_scaled)})
        wandb.log({"translated": wandb_bboxes(img_translated, annot_translated)})
        wandb.log({"rotated": wandb_bboxes(img_rotated, annot_rotated)})
        wandb.log({"sheared": wandb_bboxes(img_sheared, annot_sheared)})

        wandb.finish()



Looks great on the dashboard.

# Final Changes to Dataset Training

I will be using COCO format which is an Object Detection model which locates objects within images via boudning boxes.  These bounding boxes are useful as they can have many ways of being displayed with no wrong way to locate a rectangle within our images.

*(x,y,widgeh, height)*: we are using this in our training dataset (this is COCO format)

*(x1,y1,x2,y2)*: a formated version that we used for the BBox Augmentation phase that's also called (xmin,ymin,xmax,ymax)  This is used within Faster RCNN, Fast RCNN, RCNN, and SSD models.

*(x_center, y_center, width, height)*: A YOLO format that is used when training a YOLO model.  x_center, y_center are normalized coordinates in the center of bounding boxes, with width, height are normalized width and height of the image.

**COCO**: Commone Objects in Context, a database that pushes to aim towards support and improvment of models for Object Detection, Image Captioning, and Instance Segmentation.

        # Create sepparate paths for images and their labels (annotations)
        # these will come in handy later for the YOLO model
        train_df["path_images"] = "/kaggle/images/video_" + train_df["video_id"].astype(str) + "_" + \
                                                train_df["video_frame"].astype(str) + ".jpg"
        train_df["path_labels"] = "/kaggle/labels/video_" + train_df["video_id"].astype(str) + "_" + \
                                                train_df["video_frame"].astype(str) + ".txt"

        # Save the width and height of the images
        # it is the same for the entire dataset
        train_df["width"] = 1280
        train_df["height"] = 720

        # Simplify the annotation format
        train_df["coco_bbox"] = train_df["annotations"].apply(lambda annot: [list(item.values()) for item in eval(annot)])

        # Data Sample
        train_df.sample(5, random_state=24)

        # Save dataset
        train_df.to_csv("train.csv", index=False)


        # Save dataset Artifact
        save_dataset_artifact(run_name="save-train-data",
                            artifact_name="train_meta",
                            path="../input/2021-greatbarrierreef-prep-data/train.csv")

 We have successfully trained our boxes  according to shape, movement, and even better detailed       

# References 

Data Files (The amount of images are too big to put in my file):
https://www.kaggle.com/competitions/tensorflow-great-barrier-reef/data

Liu, J., et al. (2021). The CSIRO Crown-of-Thorn Starfish Detection Dataset. 
    Cornell University, 1. https://arxiv.org/abs/2111.14311
