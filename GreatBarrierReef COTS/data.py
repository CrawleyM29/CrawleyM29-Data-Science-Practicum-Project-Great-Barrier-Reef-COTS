# Libraries
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


# Environment check
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"
CONFIG = {'competition': 'greatReef', '_wandb_kernel': 'aot'}

# 🐝 Secrets# Libraries
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


# Environment check
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"
CONFIG = {'competition': 'greatReef', '_wandb_kernel': 'aot'}

# 🐝 Secrets
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb")


# Custom colors
class color:
    S = '\034[1m' + '\034[94m'
    E = '\034[0m'
    
my_colors = ["#16558F", "#1583D2", "#61B0B7", "#ADDEFF", "#A99AEA", "#7158B7"]
print(color.S+"Notebook Color Scheme:"+color.E)
sns.palplot(sns.color_palette(my_colors))

# Set Style
sns.set_style("white")
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.rcParams.update({'font.size': 14})

# W&B Experiment
run = wandb.init(project='GreatBarrierReef', name='DataUnderstanding', config=CONFIG, anonymous="allow")

# Read training dataset
train_df = pd.read_csv("../input/tensorflow-great-barrier-reef/train.csv")
test_df = pd.read_csv("../input/tensorflow-great-barrier-reef/test.csv")

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(23, 10))

# --- Plot 1 ---
df1 = train_df["video_id"].value_counts().reset_index()

sns.barplot(data=df1, x="index", y="video_id", ax=ax1,
            palette=my_colors)
show_values_on_bars(ax1, h_v="v", space=0.1)
ax1.set_xlabel("Video ID")
ax1.set_ylabel("")
ax1.title.set_text("Frequency of Frames per Video")
ax1.set_yticks([])

# --- Plot 2  ---
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

# Log plots into W&B Dashboard
create_wandb_plot(x_data=df1.index, 
                  y_data=df1.video_id, 
                  x_name="Video ID", y_name=" ", 
                  title="-Frequency of Frames per Video-", 
                  log="frames", plot="bar")

create_wandb_plot(x_data=df2.index, 
                  y_data=df2.sequence, 
                  x_name="Sequence ID", y_name=" ", 
                  title="-Frequency of Frames per Sequence-", 
                  log="frames2", plot="bar")

# Calculate the total number of annotations within the frame
train_df["no_annotations"] = train_df["annotations"].apply(lambda x: len(eval(x)))

# % annotations
n = len(train_df)
no_annot = round(train_df[train_df["no_annotations"]==0].shape[0]/n*100)
with_annot = round(train_df[train_df["no_annotations"]!=0].shape[0]/n*100)

print(color.S + f"There are ~{no_annot}% frames with no annotation and" + color.E,
      "\n",
      color.S + f"only ~{with_annot}% frames with at least 1 annotation." + color.E)

# Plot
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

# Log info and plots into Dashboard
wandb.log({"no annotations": no_annot,
           "with annotations": with_annot})

create_wandb_hist(x_data=train_df["no_annotations"],
                  x_name="Number of Annotations",
                  title="Distribution for Number of Annotations per Frame",
                  log="annotations")


# unique sequence values
sequences = list(train_df["sequence"].unique())

plt.figure(figsize=(23,20))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
plt.suptitle("Frequency of annotations on sequence length", fontsize = 20)

# Enumerate through all sequences
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

run = wandb.init(project='GreatBarrierReef', name='ExampleImages', config=CONFIG, anonymous="allow")



# Experiment
run = wandb.init(project='GreatBarrierReef', name='ExampleImages', config=CONFIG, anonymous="allow")

# Creating a "path" column containing full path to the frames
base_folder = "../input/tensorflow-great-barrier-reef/train_images"

train_df["path"] = base_folder + "/video_" + \
                    train_df['video_id'].astype(str) + "/" +\
                    train_df['video_frame'].astype(str) +".jpg"


# ___ Show image and annotations if applicable ____
def show_image(path, annot, axs=None):
    '''Shows an image and marks any COTS annotated within the frame.
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
    
    
# ____ Log ____
def wandb_annotation(image, annotations):
    '''Source: https://www.kaggle.com/ayuraj/visualize-bounding-boxes-interactively
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


# Showing 1 image as example
path = list(train_df[train_df["no_annotations"]==0]["path"])[0]
annot = list(train_df[train_df["no_annotations"]==0]["annotations"])[0]

# Logging Image
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
wandb_images = []
wandb_images.append(wandb_annotation(image, annot))

print(color.S+"Path:"+color.E, path)
print(color.S+"Annotation:"+color.E, annot)
print(color.S+"Frame:"+color.E)
show_image(path, annot, axs=None)

# Show only 1 image as example
path = list(train_df[train_df["no_annotations"]==18]["path"])[0]
annot = list(train_df[train_df["no_annotations"]==18]["annotations"])[0]

# Logging Image
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
wandb_images.append(wandb_annotation(image, annot))
wandb.log({"example_image": wandb_images})

print(color.S+"Path:"+color.E, path)
print(color.S+"Annotation:"+color.E, annot)
print(color.S+"Frame:"+color.E)
show_image(path, annot, axs=None)

# Viewing multiple frames within multiple sequences

def show_multiple_images(seq_id, frame_no):
    '''Shows multiple images within a sequence.
    seq_id: a number corresponding with the sequence unique ID
    frame_no: a list containing the first and last frame to plot'''
    
    # Select image paths & their annotations
    paths = list(train_df[(train_df["sequence"]==seq_id) & 
                 (train_df["sequence_frame"]>=frame_no[0]) & 
                 (train_df["sequence_frame"]<=frame_no[1])]["path"])
    annotations = list(train_df[(train_df["sequence"]==seq_id) & 
                 (train_df["sequence_frame"]>=frame_no[0]) & 
                 (train_df["sequence_frame"]<=frame_no[1])]["annotations"])

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(23, 10))
    axs = axs.flatten()
    fig.suptitle(f"Showing consecutive frames for Sequence ID: {seq_id}", fontsize = 20)

    for k, (path, annot) in enumerate(zip(paths, annotations)):
        axs[k].set_title(f"Frame No: {frame_no[0]+k}", fontsize = 12)
        show_image(path, annot, axs[k])

    plt.tight_layout()
    plt.show()

    #Testing algorithms with images



seq_id = 44160
frame_no = [51, 56]

show_multiple_images(seq_id, frame_no)

#Testing for multiple starfish instead of zero

seq_id = 59337
frame_no = [38, 43]

show_multiple_images(seq_id, frame_no)

# Distorting and enhancing the images for better identification of the COTS

seq_id = 53708
frame_no = [801, 806]

show_multiple_images(seq_id, frame_no)



def plot_comparison(no_annot, state=24):
    
    # Select image paths & their annotations
    paths_compare = list(train_df[train_df["no_annotations"]==no_annot]\
                         .sample(n=9, random_state=state)["path"])
    annotations_compare = list(train_df[train_df["no_annotations"]==no_annot]\
                               .sample(n=9, random_state=state)["annotations"])

    # Plot
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

# No annotations
no_annot = 0
plot_comparison(no_annot, state=24)

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

# Create a new column with the new formated annotations
train_df["f_annotations"] = train_df["annotations"].apply(lambda x: format_annotations(x))



def show_image_bbox(img, annot, axs=None):
    '''Shows an image and marks any COTS annotated within the frame.
    img: the output from cv2.imread()
    annot: FORMATED annotation'''
    
    # This is in case we plot only 1 image
    if axs==None:
        fig, axs = plt.subplots(figsize=(23, 8))
    
    axs.imshow(img)

    if annot:
        for a in annot:
            rect = patches.Rectangle((a[0], a[1]), a[2]-a[0], a[3]-a[1], 
                                     linewidth=3, edgecolor="#FF6103", facecolor='none')
            axs.add_patch(rect)

    axs.axis("off")

cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)



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
        
        # If random number between 0 and 1 < probability p
        if random.random() < self.p:
            # Reverse image elements in the 1st dimension
            img =  img[:,::-1,:]
            bboxes[:,[0,2]] = bboxes[:,[0,2]] + 2*(img_center[[0,2]] - bboxes[:,[0,2]])
            
            # Convert the bounding boxes
            box_w = abs(bboxes[:,0] - bboxes[:,2])
            bboxes[:,0] -= box_w
            bboxes[:,2] += box_w
            
        return img, bboxes.tolist()

# Take an example
path = list(train_df[train_df["no_annotations"]==18]["path"])[0]

img_original = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
annot_original = eval(list(train_df[train_df["no_annotations"]==18]["f_annotations"])[0])

# Horizontal Flip
horizontal_flip = RandomHorizontalFlip(p=1)  
img_flipped, annot_flipped = horizontal_flip(img_original, annot_original)



# Show the Before and After
fig, axs = plt.subplots(1, 2, figsize=(23, 10))
axs = axs.flatten()
fig.suptitle(f"(Random) Horizontal Flip", fontsize = 20)

axs[0].set_title("Original Image", fontsize = 20)
show_image_bbox(img_original, annot_original, axs=axs[0])

axs[1].set_title("With Horizontal Flip", fontsize = 20)
show_image_bbox(img_flipped, annot_flipped, axs[1])

plt.tight_layout()
plt.show()

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



random.seed(24)

# Scaling
scale = RandomScale(scale=1.3, diff = False) 
img_scaled, annot_scaled = scale(img_original, annot_original)



# Show the Before and After
fig, axs = plt.subplots(1, 2, figsize=(23, 10))
axs = axs.flatten()
fig.suptitle(f"(Random) Image Scaling", fontsize = 20)

axs[0].set_title("Original Image", fontsize = 20)
show_image_bbox(img_original, annot_original, axs=axs[0])

axs[1].set_title("Scaled (zoomed in) Image", fontsize = 20)
show_image_bbox(img_scaled, annot_scaled, axs[1])

plt.tight_layout()
plt.show()



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
        
        # Chose a random digit to scale by 
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
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas

        bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]

        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)

        return img, bboxes.tolist()# Translate
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

class RRotate(object):

    def __init__(self, angle = 10):
        
        self.angle = angle
        self.angle = (-self.angle, self.angle)
        
        
    def __call__(self, img, bboxes):

        # Convert bboxes
        bboxes = np.array(bboxes)
        
        # Compute the random angle
        angle = random.uniform(*self.angle)

        # width, height and center of the image
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2

        # Rotate the image
        img = rotate_im(img, angle)

        # --- Rotate the bounding boxes ---
        # Get the 4 point corner coordinates
        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:,4:]))
        # Rotate the bounding box
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        # Get the enclosing (new bboxes)
        new_bbox = get_enclosing_box(corners)

        # Get scaling factors to clip the image and bboxes
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h

        # Rescale the image - to w,h and not nW,nH
        img = cv2.resize(img, (w,h))

        # Clip boxes (in case there are any outside of the rotated image)
        bboxes[:,:4] = bboxes[:,:4] / [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

        return img, bboxes.tolist()

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

# === 🐝Dashboard Log (redone for formated annotations) ===
def wandb_bboxes(image, annotations):
    '''Source: https://www.kaggle.com/ayuraj/visualize-bounding-boxes-interactively
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
                       boxes={"ground_truth": {"box_data": all_annotations}}
                      )

# Log all augmented images to the Dashboard
wandb.log({"flipped": wandb_bboxes(img_flipped, annot_flipped)})
wandb.log({"scaled": wandb_bboxes(img_scaled, annot_scaled)})
wandb.log({"translated": wandb_bboxes(img_translated, annot_translated)})
wandb.log({"rotated": wandb_bboxes(img_rotated, annot_rotated)})
wandb.log({"sheared": wandb_bboxes(img_sheared, annot_sheared)})# === 🐝W&B Log (redone for formated annotations) ===


wandb.finish()

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
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb")


# Custom colors
class color:
    S = '\034[1m' + '\034[94m'
    E = '\034[0m'
    
my_colors = ["#16558F", "#1583D2", "#61B0B7", "#ADDEFF", "#A99AEA", "#7158B7"]
print(color.S+"Notebook Color Scheme:"+color.E)
sns.palplot(sns.color_palette(my_colors))

# Set Style
sns.set_style("white")
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.rcParams.update({'font.size': 14})

# W&B Experiment
run = wandb.init(project='GreatBarrierReef', name='DataUnderstanding', config=CONFIG, anonymous="allow")

# Read training dataset
train_df = pd.read_csv("../input/tensorflow-great-barrier-reef/train.csv")
test_df = pd.read_csv("../input/tensorflow-great-barrier-reef/test.csv")

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(23, 10))

# --- Plot 1 ---
df1 = train_df["video_id"].value_counts().reset_index()

sns.barplot(data=df1, x="index", y="video_id", ax=ax1,
            palette=my_colors)
show_values_on_bars(ax1, h_v="v", space=0.1)
ax1.set_xlabel("Video ID")
ax1.set_ylabel("")
ax1.title.set_text("Frequency of Frames per Video")
ax1.set_yticks([])

# --- Plot 2  ---
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

# Log plots into W&B Dashboard
create_wandb_plot(x_data=df1.index, 
                  y_data=df1.video_id, 
                  x_name="Video ID", y_name=" ", 
                  title="-Frequency of Frames per Video-", 
                  log="frames", plot="bar")

create_wandb_plot(x_data=df2.index, 
                  y_data=df2.sequence, 
                  x_name="Sequence ID", y_name=" ", 
                  title="-Frequency of Frames per Sequence-", 
                  log="frames2", plot="bar")

# Calculate the total number of annotations within the frame
train_df["no_annotations"] = train_df["annotations"].apply(lambda x: len(eval(x)))

# % annotations
n = len(train_df)
no_annot = round(train_df[train_df["no_annotations"]==0].shape[0]/n*100)
with_annot = round(train_df[train_df["no_annotations"]!=0].shape[0]/n*100)

print(color.S + f"There are ~{no_annot}% frames with no annotation and" + color.E,
      "\n",
      color.S + f"only ~{with_annot}% frames with at least 1 annotation." + color.E)

# Plot
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

# Log info and plots into Dashboard
wandb.log({"no annotations": no_annot,
           "with annotations": with_annot})

create_wandb_hist(x_data=train_df["no_annotations"],
                  x_name="Number of Annotations",
                  title="Distribution for Number of Annotations per Frame",
                  log="annotations")


# unique sequence values
sequences = list(train_df["sequence"].unique())

plt.figure(figsize=(23,20))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
plt.suptitle("Frequency of annotations on sequence length", fontsize = 20)

# Enumerate through all sequences
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

run = wandb.init(project='GreatBarrierReef', name='ExampleImages', config=CONFIG, anonymous="allow")



# Experiment
run = wandb.init(project='GreatBarrierReef', name='ExampleImages', config=CONFIG, anonymous="allow")

# Creating a "path" column containing full path to the frames
base_folder = "../input/tensorflow-great-barrier-reef/train_images"

train_df["path"] = base_folder + "/video_" + \
                    train_df['video_id'].astype(str) + "/" +\
                    train_df['video_frame'].astype(str) +".jpg"


# ___ Show image and annotations if applicable ____
def show_image(path, annot, axs=None):
    '''Shows an image and marks any COTS annotated within the frame.
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
    
    
# ____ Log ____
def wandb_annotation(image, annotations):
    '''Source: https://www.kaggle.com/ayuraj/visualize-bounding-boxes-interactively
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


# Showing 1 image as example
path = list(train_df[train_df["no_annotations"]==0]["path"])[0]
annot = list(train_df[train_df["no_annotations"]==0]["annotations"])[0]

# Logging Image
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
wandb_images = []
wandb_images.append(wandb_annotation(image, annot))

print(color.S+"Path:"+color.E, path)
print(color.S+"Annotation:"+color.E, annot)
print(color.S+"Frame:"+color.E)
show_image(path, annot, axs=None)

# Show only 1 image as example
path = list(train_df[train_df["no_annotations"]==18]["path"])[0]
annot = list(train_df[train_df["no_annotations"]==18]["annotations"])[0]

# Logging Image
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
wandb_images.append(wandb_annotation(image, annot))
wandb.log({"example_image": wandb_images})

print(color.S+"Path:"+color.E, path)
print(color.S+"Annotation:"+color.E, annot)
print(color.S+"Frame:"+color.E)
show_image(path, annot, axs=None)

# Viewing multiple frames within multiple sequences

def show_multiple_images(seq_id, frame_no):
    '''Shows multiple images within a sequence.
    seq_id: a number corresponding with the sequence unique ID
    frame_no: a list containing the first and last frame to plot'''
    
    # Select image paths & their annotations
    paths = list(train_df[(train_df["sequence"]==seq_id) & 
                 (train_df["sequence_frame"]>=frame_no[0]) & 
                 (train_df["sequence_frame"]<=frame_no[1])]["path"])
    annotations = list(train_df[(train_df["sequence"]==seq_id) & 
                 (train_df["sequence_frame"]>=frame_no[0]) & 
                 (train_df["sequence_frame"]<=frame_no[1])]["annotations"])

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(23, 10))
    axs = axs.flatten()
    fig.suptitle(f"Showing consecutive frames for Sequence ID: {seq_id}", fontsize = 20)

    for k, (path, annot) in enumerate(zip(paths, annotations)):
        axs[k].set_title(f"Frame No: {frame_no[0]+k}", fontsize = 12)
        show_image(path, annot, axs[k])

    plt.tight_layout()
    plt.show()

    #Testing algorithms with images



seq_id = 44160
frame_no = [51, 56]

show_multiple_images(seq_id, frame_no)

#Testing for multiple starfish instead of zero

seq_id = 59337
frame_no = [38, 43]

show_multiple_images(seq_id, frame_no)

# Distorting and enhancing the images for better identification of the COTS

seq_id = 53708
frame_no = [801, 806]

show_multiple_images(seq_id, frame_no)



def plot_comparison(no_annot, state=24):
    
    # Select image paths & their annotations
    paths_compare = list(train_df[train_df["no_annotations"]==no_annot]\
                         .sample(n=9, random_state=state)["path"])
    annotations_compare = list(train_df[train_df["no_annotations"]==no_annot]\
                               .sample(n=9, random_state=state)["annotations"])

    # Plot
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

# No annotations
no_annot = 0
plot_comparison(no_annot, state=24)

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

# Create a new column with the new formated annotations
train_df["f_annotations"] = train_df["annotations"].apply(lambda x: format_annotations(x))



def show_image_bbox(img, annot, axs=None):
    '''Shows an image and marks any COTS annotated within the frame.
    img: the output from cv2.imread()
    annot: FORMATED annotation'''
    
    # This is in case we plot only 1 image
    if axs==None:
        fig, axs = plt.subplots(figsize=(23, 8))
    
    axs.imshow(img)

    if annot:
        for a in annot:
            rect = patches.Rectangle((a[0], a[1]), a[2]-a[0], a[3]-a[1], 
                                     linewidth=3, edgecolor="#FF6103", facecolor='none')
            axs.add_patch(rect)

    axs.axis("off")

cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)



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
        
        # If random number between 0 and 1 < probability p
        if random.random() < self.p:
            # Reverse image elements in the 1st dimension
            img =  img[:,::-1,:]
            bboxes[:,[0,2]] = bboxes[:,[0,2]] + 2*(img_center[[0,2]] - bboxes[:,[0,2]])
            
            # Convert the bounding boxes
            box_w = abs(bboxes[:,0] - bboxes[:,2])
            bboxes[:,0] -= box_w
            bboxes[:,2] += box_w
            
        return img, bboxes.tolist()

# Take an example
path = list(train_df[train_df["no_annotations"]==18]["path"])[0]

img_original = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
annot_original = eval(list(train_df[train_df["no_annotations"]==18]["f_annotations"])[0])

# Horizontal Flip
horizontal_flip = RandomHorizontalFlip(p=1)  
img_flipped, annot_flipped = horizontal_flip(img_original, annot_original)



# Show the Before and After
fig, axs = plt.subplots(1, 2, figsize=(23, 10))
axs = axs.flatten()
fig.suptitle(f"(Random) Horizontal Flip", fontsize = 20)

axs[0].set_title("Original Image", fontsize = 20)
show_image_bbox(img_original, annot_original, axs=axs[0])

axs[1].set_title("With Horizontal Flip", fontsize = 20)
show_image_bbox(img_flipped, annot_flipped, axs[1])

plt.tight_layout()
plt.show()

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



random.seed(24)

# Scaling
scale = RandomScale(scale=1.3, diff = False) 
img_scaled, annot_scaled = scale(img_original, annot_original)



# Show the Before and After
fig, axs = plt.subplots(1, 2, figsize=(23, 10))
axs = axs.flatten()
fig.suptitle(f"(Random) Image Scaling", fontsize = 20)

axs[0].set_title("Original Image", fontsize = 20)
show_image_bbox(img_original, annot_original, axs=axs[0])

axs[1].set_title("Scaled (zoomed in) Image", fontsize = 20)
show_image_bbox(img_scaled, annot_scaled, axs[1])

plt.tight_layout()
plt.show()



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
        
        # Chose a random digit to scale by 
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
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas

        bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]

        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)

        return img, bboxes.tolist()# Translate
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

class RRotate(object):

    def __init__(self, angle = 10):
        
        self.angle = angle
        self.angle = (-self.angle, self.angle)
        
        
    def __call__(self, img, bboxes):

        # Convert bboxes
        bboxes = np.array(bboxes)
        
        # Compute the random angle
        angle = random.uniform(*self.angle)

        # width, height and center of the image
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2

        # Rotate the image
        img = rotate_im(img, angle)

        # --- Rotate the bounding boxes ---
        # Get the 4 point corner coordinates
        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:,4:]))
        # Rotate the bounding box
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        # Get the enclosing (new bboxes)
        new_bbox = get_enclosing_box(corners)

        # Get scaling factors to clip the image and bboxes
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h

        # Rescale the image - to w,h and not nW,nH
        img = cv2.resize(img, (w,h))

        # Clip boxes (in case there are any outside of the rotated image)
        bboxes[:,:4] = bboxes[:,:4] / [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

        return img, bboxes.tolist()

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

# Libraries
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


# Environment check
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"
CONFIG = {'competition': 'greatReef', '_wandb_kernel': 'aot'}

# 🐝 Secrets
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb")


# Custom colors
class color:
    S = '\034[1m' + '\034[94m'
    E = '\034[0m'
    
my_colors = ["#16558F", "#1583D2", "#61B0B7", "#ADDEFF", "#A99AEA", "#7158B7"]
print(color.S+"Notebook Color Scheme:"+color.E)
sns.palplot(sns.color_palette(my_colors))

# Set Style
sns.set_style("white")
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['axes.spines.left'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
plt.rcParams.update({'font.size': 14})

# W&B Experiment
run = wandb.init(project='GreatBarrierReef', name='DataUnderstanding', config=CONFIG, anonymous="allow")

# Read training dataset
train_df = pd.read_csv("../input/tensorflow-great-barrier-reef/train.csv")
test_df = pd.read_csv("../input/tensorflow-great-barrier-reef/test.csv")

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(23, 10))

# --- Plot 1 ---
df1 = train_df["video_id"].value_counts().reset_index()

sns.barplot(data=df1, x="index", y="video_id", ax=ax1,
            palette=my_colors)
show_values_on_bars(ax1, h_v="v", space=0.1)
ax1.set_xlabel("Video ID")
ax1.set_ylabel("")
ax1.title.set_text("Frequency of Frames per Video")
ax1.set_yticks([])

# --- Plot 2  ---
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

# Log plots into W&B Dashboard
create_wandb_plot(x_data=df1.index, 
                  y_data=df1.video_id, 
                  x_name="Video ID", y_name=" ", 
                  title="-Frequency of Frames per Video-", 
                  log="frames", plot="bar")

create_wandb_plot(x_data=df2.index, 
                  y_data=df2.sequence, 
                  x_name="Sequence ID", y_name=" ", 
                  title="-Frequency of Frames per Sequence-", 
                  log="frames2", plot="bar")

# Calculate the total number of annotations within the frame
train_df["no_annotations"] = train_df["annotations"].apply(lambda x: len(eval(x)))

# % annotations
n = len(train_df)
no_annot = round(train_df[train_df["no_annotations"]==0].shape[0]/n*100)
with_annot = round(train_df[train_df["no_annotations"]!=0].shape[0]/n*100)

print(color.S + f"There are ~{no_annot}% frames with no annotation and" + color.E,
      "\n",
      color.S + f"only ~{with_annot}% frames with at least 1 annotation." + color.E)

# Plot
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

# Log info and plots into Dashboard
wandb.log({"no annotations": no_annot,
           "with annotations": with_annot})

create_wandb_hist(x_data=train_df["no_annotations"],
                  x_name="Number of Annotations",
                  title="Distribution for Number of Annotations per Frame",
                  log="annotations")


# unique sequence values
sequences = list(train_df["sequence"].unique())

plt.figure(figsize=(23,20))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
plt.suptitle("Frequency of annotations on sequence length", fontsize = 20)

# Enumerate through all sequences
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

run = wandb.init(project='GreatBarrierReef', name='ExampleImages', config=CONFIG, anonymous="allow")



# Experiment
run = wandb.init(project='GreatBarrierReef', name='ExampleImages', config=CONFIG, anonymous="allow")

# Creating a "path" column containing full path to the frames
base_folder = "../input/tensorflow-great-barrier-reef/train_images"

train_df["path"] = base_folder + "/video_" + \
                    train_df['video_id'].astype(str) + "/" +\
                    train_df['video_frame'].astype(str) +".jpg"


# ___ Show image and annotations if applicable ____
def show_image(path, annot, axs=None):
    '''Shows an image and marks any COTS annotated within the frame.
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
    
    
# ____ Log ____
def wandb_annotation(image, annotations):
    '''Source: https://www.kaggle.com/ayuraj/visualize-bounding-boxes-interactively
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


# Showing 1 image as example
path = list(train_df[train_df["no_annotations"]==0]["path"])[0]
annot = list(train_df[train_df["no_annotations"]==0]["annotations"])[0]

# Logging Image
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
wandb_images = []
wandb_images.append(wandb_annotation(image, annot))

print(color.S+"Path:"+color.E, path)
print(color.S+"Annotation:"+color.E, annot)
print(color.S+"Frame:"+color.E)
show_image(path, annot, axs=None)

# Show only 1 image as example
path = list(train_df[train_df["no_annotations"]==18]["path"])[0]
annot = list(train_df[train_df["no_annotations"]==18]["annotations"])[0]

# Logging Image
image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
wandb_images.append(wandb_annotation(image, annot))
wandb.log({"example_image": wandb_images})

print(color.S+"Path:"+color.E, path)
print(color.S+"Annotation:"+color.E, annot)
print(color.S+"Frame:"+color.E)
show_image(path, annot, axs=None)

# Viewing multiple frames within multiple sequences

def show_multiple_images(seq_id, frame_no):
    '''Shows multiple images within a sequence.
    seq_id: a number corresponding with the sequence unique ID
    frame_no: a list containing the first and last frame to plot'''
    
    # Select image paths & their annotations
    paths = list(train_df[(train_df["sequence"]==seq_id) & 
                 (train_df["sequence_frame"]>=frame_no[0]) & 
                 (train_df["sequence_frame"]<=frame_no[1])]["path"])
    annotations = list(train_df[(train_df["sequence"]==seq_id) & 
                 (train_df["sequence_frame"]>=frame_no[0]) & 
                 (train_df["sequence_frame"]<=frame_no[1])]["annotations"])

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(23, 10))
    axs = axs.flatten()
    fig.suptitle(f"Showing consecutive frames for Sequence ID: {seq_id}", fontsize = 20)

    for k, (path, annot) in enumerate(zip(paths, annotations)):
        axs[k].set_title(f"Frame No: {frame_no[0]+k}", fontsize = 12)
        show_image(path, annot, axs[k])

    plt.tight_layout()
    plt.show()

    #Testing algorithms with images



seq_id = 44160
frame_no = [51, 56]

show_multiple_images(seq_id, frame_no)

#Testing for multiple starfish instead of zero

seq_id = 59337
frame_no = [38, 43]

show_multiple_images(seq_id, frame_no)

# Distorting and enhancing the images for better identification of the COTS

seq_id = 53708
frame_no = [801, 806]

show_multiple_images(seq_id, frame_no)



def plot_comparison(no_annot, state=24):
    
    # Select image paths & their annotations
    paths_compare = list(train_df[train_df["no_annotations"]==no_annot]\
                         .sample(n=9, random_state=state)["path"])
    annotations_compare = list(train_df[train_df["no_annotations"]==no_annot]\
                               .sample(n=9, random_state=state)["annotations"])

    # Plot
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

# No annotations
no_annot = 0
plot_comparison(no_annot, state=24)

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

# Create a new column with the new formated annotations
train_df["f_annotations"] = train_df["annotations"].apply(lambda x: format_annotations(x))



def show_image_bbox(img, annot, axs=None):
    '''Shows an image and marks any COTS annotated within the frame.
    img: the output from cv2.imread()
    annot: FORMATED annotation'''
    
    # This is in case we plot only 1 image
    if axs==None:
        fig, axs = plt.subplots(figsize=(23, 8))
    
    axs.imshow(img)

    if annot:
        for a in annot:
            rect = patches.Rectangle((a[0], a[1]), a[2]-a[0], a[3]-a[1], 
                                     linewidth=3, edgecolor="#FF6103", facecolor='none')
            axs.add_patch(rect)

    axs.axis("off")

cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)



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
        
        # If random number between 0 and 1 < probability p
        if random.random() < self.p:
            # Reverse image elements in the 1st dimension
            img =  img[:,::-1,:]
            bboxes[:,[0,2]] = bboxes[:,[0,2]] + 2*(img_center[[0,2]] - bboxes[:,[0,2]])
            
            # Convert the bounding boxes
            box_w = abs(bboxes[:,0] - bboxes[:,2])
            bboxes[:,0] -= box_w
            bboxes[:,2] += box_w
            
        return img, bboxes.tolist()

# Take an example
path = list(train_df[train_df["no_annotations"]==18]["path"])[0]

img_original = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
annot_original = eval(list(train_df[train_df["no_annotations"]==18]["f_annotations"])[0])

# Horizontal Flip
horizontal_flip = RandomHorizontalFlip(p=1)  
img_flipped, annot_flipped = horizontal_flip(img_original, annot_original)



# Show the Before and After
fig, axs = plt.subplots(1, 2, figsize=(23, 10))
axs = axs.flatten()
fig.suptitle(f"(Random) Horizontal Flip", fontsize = 20)

axs[0].set_title("Original Image", fontsize = 20)
show_image_bbox(img_original, annot_original, axs=axs[0])

axs[1].set_title("With Horizontal Flip", fontsize = 20)
show_image_bbox(img_flipped, annot_flipped, axs[1])

plt.tight_layout()
plt.show()

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



random.seed(24)

# Scaling
scale = RandomScale(scale=1.3, diff = False) 
img_scaled, annot_scaled = scale(img_original, annot_original)



# Show the Before and After
fig, axs = plt.subplots(1, 2, figsize=(23, 10))
axs = axs.flatten()
fig.suptitle(f"(Random) Image Scaling", fontsize = 20)

axs[0].set_title("Original Image", fontsize = 20)
show_image_bbox(img_original, annot_original, axs=axs[0])

axs[1].set_title("Scaled (zoomed in) Image", fontsize = 20)
show_image_bbox(img_scaled, annot_scaled, axs[1])

plt.tight_layout()
plt.show()



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
        
        # Chose a random digit to scale by 
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
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas

        bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]

        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)

        return img, bboxes.tolist()# Translate
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

class RRotate(object):

    def __init__(self, angle = 10):
        
        self.angle = angle
        self.angle = (-self.angle, self.angle)
        
        
    def __call__(self, img, bboxes):

        # Convert bboxes
        bboxes = np.array(bboxes)
        
        # Compute the random angle
        angle = random.uniform(*self.angle)

        # width, height and center of the image
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2

        # Rotate the image
        img = rotate_im(img, angle)

        # --- Rotate the bounding boxes ---
        # Get the 4 point corner coordinates
        corners = get_corners(bboxes)
        corners = np.hstack((corners, bboxes[:,4:]))
        # Rotate the bounding box
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        # Get the enclosing (new bboxes)
        new_bbox = get_enclosing_box(corners)

        # Get scaling factors to clip the image and bboxes
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h

        # Rescale the image - to w,h and not nW,nH
        img = cv2.resize(img, (w,h))

        # Clip boxes (in case there are any outside of the rotated image)
        bboxes[:,:4] = bboxes[:,:4] / [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

        return img, bboxes.tolist()

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

# === 🐝Dashboard Log (redone for formated annotations) ===
def wandb_bboxes(image, annotations):
    '''Source: https://www.kaggle.com/ayuraj/visualize-bounding-boxes-interactively
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
                       boxes={"ground_truth": {"box_data": all_annotations}}
                      )

# Log all augmented images to the Dashboard
wandb.log({"flipped": wandb_bboxes(img_flipped, annot_flipped)})
wandb.log({"scaled": wandb_bboxes(img_scaled, annot_scaled)})
wandb.log({"translated": wandb_bboxes(img_translated, annot_translated)})
wandb.log({"rotated": wandb_bboxes(img_rotated, annot_rotated)})
wandb.log({"sheared": wandb_bboxes(img_sheared, annot_sheared)})# === 🐝W&B Log (redone for formated annotations) ===


wandb.finish()

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

