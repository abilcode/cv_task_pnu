# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        pass

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
import shutil
SOURCE_DIR = '/kaggle/input'

# %%
import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import seaborn as sns

import pandas as pd
from collections import Counter
#%matplotlib inline

# %%
"""
## Remove Corrupted Image and Working with Kaggle Env
"""

# %%
def copy_and_clean_dataset(source_path, dest_path):
    """
    Copy dataset from read-only input to working directory and remove corrupted images
    """
    
    print(f"📂 Copying dataset from {source_path} to {dest_path}")
    
    # Copy the entire dataset
    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)
    
    shutil.copytree(source_path, dest_path)
    print("✅ Dataset copied successfully")
    
    # Now clean corrupted images
    print("🧹 Cleaning corrupted images...")
    corrupted_count = 0
    
    for root, dirs, files in os.walk(dest_path):
        print(root)
        # Skip unknown directory
        if 'unknown' in str(root):
            print('unknown' in str(root))
            continue
    
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except:
                    print(f"  Removing: {file}")
                    os.remove(file_path)
                    corrupted_count += 1
    print(f"✅ Removed {corrupted_count} corrupted images")
    print(f"📁 Clean dataset ready at: {dest_path}")

# %%
image_ = os.path.join(SOURCE_DIR,'car-classification-task/car_dataset_split')
train_ = os.path.join(image_, "training_images")
test_ = os.path.join(image_, "testing_images")


print(train_, test_, sep = '\n')

# %%
# Usage for Kaggle:
source = image_
dest = "/kaggle/working/car_dataset_split"

copy_and_clean_dataset(source, dest)


# %%
# Now use the cleaned dataset
image_source = '/kaggle/working/car_dataset_split'
train_dir = os.path.join(image_source, "training_images")
test_dir = os.path.join(image_source, "testing_images")

image_path = image_source
print(train_dir, test_dir, sep = '\n')

# %%
"""
### Excluding Unknown
"""

# %%
unknown_train='/kaggle/working/car_dataset_split/training_images/unknown'
if os.path.exists(unknown_train):
    shutil.rmtree(unknown_train)
    print("🗑️  Removed unknown folder from training")

unknown_test='/kaggle/working/car_dataset_split/testing_images/unknown'
if os.path.exists(unknown_test):
    shutil.rmtree(unknown_test)
    print("🗑️  Removed unknown folder from testing")

# %%
"""
## Images of each Split
"""

# %%
classes = os.listdir(train_dir)
image_paths = []
labels = []

for label in classes:
    class_dir = os.path.join(train_dir, label)
    for img_name in os.listdir(class_dir):
        image_paths.append(os.path.join(class_dir, img_name))
        labels.append(label)

sample_indices = random.sample(range(len(image_paths)), 25)
sample_images = [image_paths[i] for i in sample_indices]
sample_labels = [labels[i] for i in sample_indices]

fig, axes = plt.subplots(5, 5, figsize=(15, 15))
axes = axes.flatten()

for img_path, ax, label in zip(sample_images, axes, sample_labels):
    img = Image.open(img_path)
    ax.imshow(img)
    ax.set_title(label)
    ax.axis('off')

plt.tight_layout()
plt.show()

# %%
# def create_dataset_dataframes(dataset_path):
#     """
#     Create train and test dataframes with id | class | image_name format
    
#     Args:
#         dataset_path (str): Path to car_dataset_split directory
        
#     Returns:
#         tuple: (train_df, test_df)
#     """
    
#     dataset_path = Path(dataset_path)
    
#     # Training data
#     train_data = []
#     train_dir = dataset_path / "training_images"
    
#     if train_dir.exists():
#         for class_folder in train_dir.iterdir():
#             if class_folder.is_dir():
#                 class_name = class_folder.name
#                 for image_file in class_folder.iterdir():
#                     if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
#                         train_data.append({
#                             'id': len(train_data) + 1,
#                             'class': class_name,
#                             'image_name': image_file.name
#                         })
    
#     # Testing data
#     test_data = []
#     test_dir = dataset_path / "testing_images"
    
#     if test_dir.exists():
#         for class_folder in test_dir.iterdir():
#             if class_folder.is_dir():
#                 class_name = class_folder.name
#                 for image_file in class_folder.iterdir():
#                     if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
#                         test_data.append({
#                             'id': len(test_data) + 1,
#                             'class': class_name,
#                             'image_name': image_file.name
#                         })
    
#     train_df = pd.DataFrame(train_data)
#     test_df = pd.DataFrame(test_data)
    
#     return train_df, test_df

# %%
def create_balanced_dataset_dataframes(dataset_path, random_seed=42):
    """
    Create balanced train and test dataframes with id | class | image_name format
    Each class will have the same number of examples (minimum count across all classes)
    
    Args:
        dataset_path (str): Path to car_dataset_split directory
        random_seed (int): Random seed for reproducible sampling
        
    Returns:
        tuple: (train_df, test_df)
    """
    
    dataset_path = Path(dataset_path)
    random.seed(random_seed)
    
    def process_split(split_dir):
        """Process a single split (train/test) and return balanced dataframe"""
        data_by_class = {}
        
        if split_dir.exists():
            # First pass: collect all images by class
            for class_folder in split_dir.iterdir():
                if class_folder.is_dir():
                    class_name = class_folder.name
                    images = []
                    
                    for image_file in class_folder.iterdir():
                        if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
                            images.append(image_file.name)
                    
                    if images:  # Only add if class has images
                        data_by_class[class_name] = images
            
            # Find minimum count across all classes
            if data_by_class:
                min_count = min(len(images) for images in data_by_class.values())
                print(f"Minimum class count in {split_dir.name}: {min_count}")
                
                # Sample min_count images from each class
                balanced_data = []
                for class_name, images in data_by_class.items():
                    # Randomly sample min_count images
                    sampled_images = random.sample(images, min_count)
                    
                    for image_name in sampled_images:
                        balanced_data.append({
                            'id': len(balanced_data) + 1,
                            'class': class_name,
                            'image_name': image_name
                        })
                
                return pd.DataFrame(balanced_data)
        
        return pd.DataFrame()  # Return empty dataframe if no data
    
    # Process training data
    train_dir = dataset_path / "training_images"
    train_df = process_split(train_dir)
    
    # Process testing data
    test_dir = dataset_path / "testing_images"
    test_df = process_split(test_dir)
    
    # Print class distribution summary
    if not train_df.empty:
        print("\nTraining set class distribution:")
        print(train_df['class'].value_counts().sort_index())
    
    if not test_df.empty:
        print("\nTesting set class distribution:")
        print(test_df['class'].value_counts().sort_index())
    
    return train_df, test_df

# %%
# Create balanced datasets
train_df, test_df = create_balanced_dataset_dataframes(image_path, random_seed = 42)

# %%
plt.figure(figsize=(10,8))
sns.countplot(data=train_df, x='class', palette='gnuplot')
plt.title('Number of Images per Class')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.show()

# %%
plt.figure(figsize=(10,8))
sns.countplot(data=test_df, x='class', palette='gnuplot')
plt.title('Number of Images per Class')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.show()

# %%
"""
## Image Processing
"""

# %%
data_transform=transforms.Compose([transforms.Resize(size=(224,224)),transforms.RandomHorizontalFlip(p=0.5),
                                 transforms.ToTensor()])


data_transform

# %%
image_path = Path(image_path)

# Get list of image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

image_path_list[:10]

# %%
"""
## Image Transformation
"""

# %%
def plot_transformed_images(image_paths, transform, n=3, seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list, transform=data_transform, n=5)

# %%
"""
### ImageFolder for dataset
"""

# %%
train_data = datasets.ImageFolder(root=train_dir, transform=data_transform, target_transform=None)
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")

# %%
class_names = train_data.classes
class_dict = train_data.class_to_idx
class_dict, class_names

# %%
len(train_data), len(test_data)

# %%
"""
## DataLoaders
"""

# %%
train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=8, # how many samples per batch?
                              num_workers=1, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True) # shuffle the data?

test_dataloader = DataLoader(dataset=test_data, 
                             batch_size=8, 
                             num_workers=1, 
                             shuffle=False) # don't usually need to shuffle testing data

train_dataloader, test_dataloader

# %%
"""
Example
"""

# %%
height = 224 # H ("The training resolution is 224.")
width = 224 # W
color_channels = 3 # C
patch_size = 16 # P

# Calculate N (number of patches)
number_of_patches = int((height * width) / patch_size**2)
print(f"Number of patches (N) with image height (H={height}), width (W={width}) and patch size (P={patch_size}): {number_of_patches}")

# %%
embedding_layer_input_shape = (height, width, color_channels)

# Output shape
embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)

print(f"Input shape (single 2D image): {embedding_layer_input_shape}")
print(f"Output shape (single 2D image flattened into patches): {embedding_layer_output_shape}")

# %%
image_batch, label_batch = next(iter(train_dataloader))

# Get a single image from the batch
image, label = image_batch[0], label_batch[0]

# View the batch shapes
image.shape, label

# %%
plt.figure(figsize=(10,8))
plt.imshow(image.permute(1, 2, 0)) # rearrange image dimensions to suit matplotlib [color_channels, height, width] -> [height, width, color_channels]
plt.title(class_names[label])
plt.axis(False)
plt.show()

# %%
image_permuted = image.permute(1, 2, 0)

# Index to plot the top row of patched pixels
patch_size = 16
plt.figure(figsize=(patch_size, patch_size))
plt.imshow(image_permuted[:patch_size, :, :])
plt.show()

# %%
img_size = 224
patch_size = 16
num_patches = img_size/patch_size
assert img_size % patch_size == 0, "Image size must be divisible by patch size"
print(f"Number of patches per row: {num_patches}\nPatch size: {patch_size} pixels x {patch_size} pixels")

# Create a series of subplots
fig, axs = plt.subplots(nrows=1,
                        ncols=img_size // patch_size, # one column for each patch
                        figsize=(num_patches, num_patches),
                        sharex=True,
                        sharey=True)

# Iterate through number of patches in the top row
for i, patch in enumerate(range(0, img_size, patch_size)):
    axs[i].imshow(image_permuted[:patch_size, patch:patch+patch_size, :]); # keep height index constant, alter the width index
    axs[i].set_xlabel(i+1) # set the label
    axs[i].set_xticks([])
    axs[i].set_yticks([])

# %%
img_size = 224
patch_size = 16
num_patches = img_size/patch_size
assert img_size % patch_size == 0, "Image size must be divisible by patch size"
print(f"Number of patches per row: {num_patches}\
        \nNumber of patches per column: {num_patches}\
        \nTotal patches: {num_patches*num_patches}\
        \nPatch size: {patch_size} pixels x {patch_size} pixels")

# Create a series of subplots
fig, axs = plt.subplots(nrows=img_size // patch_size, # need int not float
                        ncols=img_size // patch_size,
                        figsize=(num_patches, num_patches),
                        sharex=True,
                        sharey=True)

# Loop through height and width of image
for i, patch_height in enumerate(range(0, img_size, patch_size)): # iterate through height
    for j, patch_width in enumerate(range(0, img_size, patch_size)): # iterate through width

        # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
        axs[i, j].imshow(image_permuted[patch_height:patch_height+patch_size, # iterate through height
                                        patch_width:patch_width+patch_size, # iterate through width
                                        :]) # get all color channels

        # Set up label information, remove the ticks for clarity and set labels to outside
        axs[i, j].set_ylabel(i+1,
                             rotation="horizontal",
                             horizontalalignment="right",
                             verticalalignment="center")
        axs[i, j].set_xlabel(j+1)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].label_outer()

# Set a super title
fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
plt.show()

# %%
"""
## ViT Patch Emd to Pytorch
"""

# %%
from torchinfo import summary
class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # 2. Initialize the class with appropriate variables
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

    # 5. Define the forward method
    def forward(self, x):
        # Create assertion to check that inputs are the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {patch_size}"

        # Perform the forward pass
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        # 6. Make sure the output shape has the right order
        return x_flattened.permute(0, 2, 1)

    
random_input_image = (1, 3, 224, 224)
random_input_image_error = (1, 3, 250, 250) # will error because image size is incompatible with patch_size

# # Get a summary of the input and outputs of PatchEmbedding (uncomment for full output)
summary(PatchEmbedding(),
        input_size=random_input_image, # try swapping this for "random_input_image_error"
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# %%
# 1. Set patch size
patch_size = 16

# 2. Print shape of original image tensor and get the image dimensions
print(f"Image tensor shape: {image.shape}")
height, width = image.shape[1], image.shape[2]

# 3. Get image tensor and add batch dimension
x = image.unsqueeze(0)
print(f"Input image with batch dimension shape: {x.shape}")

# 4. Create patch embedding layer
patch_embedding_layer = PatchEmbedding(in_channels=3,
                                       patch_size=patch_size,
                                       embedding_dim=768)

# 5. Pass image through patch embedding layer
patch_embedding = patch_embedding_layer(x)
print(f"Patching embedding shape: {patch_embedding.shape}")

# 6. Create class token embedding
batch_size = patch_embedding.shape[0]
embedding_dimension = patch_embedding.shape[-1]
class_token = nn.Parameter(torch.ones(batch_size, 1, embedding_dimension),
                           requires_grad=True) # make sure it's learnable
print(f"Class token embedding shape: {class_token.shape}")

# 7. Prepend class token embedding to patch embedding
patch_embedding_class_token = torch.cat((class_token, patch_embedding), dim=1)
print(f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")

# 8. Create position embedding
number_of_patches = int((height * width) / patch_size**2)
position_embedding = nn.Parameter(torch.ones(1, number_of_patches+1, embedding_dimension),
                                  requires_grad=True) # make sure it's learnable

# 9. Add position embedding to patch embedding with class token
patch_and_position_embedding = patch_embedding_class_token + position_embedding
print(f"Patch and position embedding shape: {patch_and_position_embedding.shape}")

# %%
"""
## Attention
"""

# %%
class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """
    # 2. Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0): # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # does our batch dimension come first?

    # 5. Create a forward() method to pass the data through the layers
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, # query embeddings
                                             key=x, # key embeddings
                                             value=x, # value embeddings
                                             need_weights=False) # do we need the weights or just the layer outputs?
        return attn_output
multihead_self_attention_block = MultiheadSelfAttentionBlock(embedding_dim=768, # from Table 1
                                                             num_heads=12) # from Table 1

# Pass patch and position image embedding through MSABlock
patched_image_through_msa_block = multihead_self_attention_block(patch_and_position_embedding)
print(f"Input shape of MSA block: {patch_and_position_embedding.shape}")
print(f"Output shape MSA block: {patched_image_through_msa_block.shape}")

# %%
"""
## MLP
"""

# %%
class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden Size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 dropout:float=0.1): # Dropout from Table 3 for ViT-Base
        super().__init__()

        # 3. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 4. Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(), # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim), # take back to embedding_dim
            nn.Dropout(p=dropout) # "Dropout, when used, is applied after every dense layer.."
        )

    # 5. Create a forward() method to pass the data through the layers
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
# Create an instance of MLPBlock
mlp_block = MLPBlock(embedding_dim=768, # from Table 1
                     mlp_size=3072, # from Table 1
                     dropout=0.1) # from Table 3

# Pass output of MSABlock through MLPBlock
patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block)
print(f"Input shape of MLP block: {patched_image_through_msa_block.shape}")
print(f"Output shape MLP block: {patched_image_through_mlp_block.shape}")

# %%
"""
## Transformer Encoder
"""

# %%
# 1. Create a class that inherits from nn.Module
class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout:float=0): # Amount of dropout for attention layers
        super().__init__()

        # 3. Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        # 4. Create MLP block (equation 3)
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)

    # 5. Create a forward() method
    def forward(self, x):

        # 6. Create residual connection for MSA block (add the input to the output)
        x =  self.msa_block(x) + x

        # 7. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x
transformer_encoder_block = TransformerEncoderBlock()

# # Print an input and output summary of our Transformer Encoder (uncomment for full output)
summary(model=transformer_encoder_block,
        input_size=(1, 197, 768), # (batch_size, num_patches, embedding_dimension)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# %%
"""
## ViT
"""

# %%
class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 2. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!

        # 3. Make the image size is divisible by the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # 4. Calculate number of patches (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size**2

        # 5. Create learnable class embedding (needs to go at front of sequence of patch embeddings)
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        # 6. Create learnable position embedding
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim),
                                               requires_grad=True)

        # 7. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 8. Create patch embedding layer
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # 9. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        # 10. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    # 11. Create a forward() method
    def forward(self, x):

        # 12. Get batch size
        batch_size = x.shape[0]

        # 13. Create class token embedding and expand it to match the batch size (equation 1)
        class_token = self.class_embedding.expand(batch_size, -1, -1) # "-1" means to infer the dimension (try this line on its own)

        # 14. Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # 15. Concat class embedding and patch embedding (equation 1)
        x = torch.cat((class_token, x), dim=1)

        # 16. Add position embedding to patch embedding (equation 1)
        x = self.position_embedding + x

        # 17. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 18. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 19. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x

# %%
"""
### Compile
"""

# %%
# Create a random tensor with same shape as a single image
random_image_tensor = torch.randn(1, 3, 224, 224) # (batch_size, color_channels, height, width)

# Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)
vit = ViT(num_classes=len(class_names))

# Pass the random image tensor to our ViT instance
vit(random_image_tensor)

# %%
summary(model=vit,
        input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

# %%
"""
## Train and Testing
"""

# %%
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

# %%
def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

# %%
from tqdm.auto import tqdm

# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results

# %%
"""
### Training
"""

# %%
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 50

device="cuda" if torch.cuda.is_available() else "cpu"

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=vit.parameters(), lr=0.001)

# Start the timer
from timeit import default_timer as timer 
start_time = timer()

# Train model_0 
results = train(model=vit, 
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

# End the timer and print out how long it took
end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# %%
from typing import Dict, List
def plot_loss_curves(results: Dict[str, List[float]]):
    loss = results['train_loss']
    test_loss = results['test_loss']
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    epochs = range(len(results['train_loss']))
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()
    
plot_loss_curves(results)

# %%
import seaborn as sns

from sklearn.metrics import confusion_matrix

# Set model to evaluation mode
vit.eval()

all_labels = []
all_preds = []
all_outputs = []
# Turn on inference mode
with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        outputs = vit(X)
        all_outputs.append(outputs.cpu().numpy())
        _, preds = torch.max(outputs, 1)

        all_labels.extend(y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Get class names from train data
class_names = train_data.classes

# Plot confusion matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='CMRmap', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# %%
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    cohen_kappa_score
)

plt.figure(figsize=(10,5),facecolor="white")
acc_score = accuracy_score(all_labels,all_preds)
plt.plot([])
plt.text(0,0, f'Accuracy Score  Score: {acc_score:.4f}', fontsize=20, ha='center', va='center',color="crimson")
plt.axis('off')

# Set the x-axis limits
plt.xlim(-1, 1)
plt.ylim(-1,1)

plt.show()

# %%
vit.eval()
all_labels = []
all_outputs = []

with torch.no_grad():
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        outputs = vit(X)
        
        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        all_outputs.append(probabilities.cpu().numpy())  # Collect probabilities

        all_labels.extend(y.cpu().numpy())

# Convert list of outputs to a 2D array
all_outputs = np.vstack(all_outputs)  # Stack the probabilities to get shape (n_samples, n_classes)

# Calculate ROC AUC score
roc_auc = roc_auc_score(all_labels, all_outputs, multi_class='ovr')

plt.figure(figsize=(10, 5), facecolor='white')
plt.plot([])
plt.text(0, 0, f'ROC AUC Score: {roc_auc:.4f}', fontsize=20, ha='center', va='center', color="crimson")
plt.axis('off')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()

# %%
plt.figure(figsize=(10,5),facecolor="white")
kappa = cohen_kappa_score(all_labels,all_preds)
plt.plot([])
plt.text(0,0, f'Cohen Kappa Score: {kappa:.4f}', fontsize=20, ha='center', va='center',color="crimson")
plt.axis('off')

# Set the x-axis limits
plt.xlim(-1, 1)
plt.ylim(-1,1)

plt.show()

# %%
plt.figure(figsize=(10,5),facecolor="white")
mcc = matthews_corrcoef(all_labels,all_preds)


plt.plot([])
plt.text(0,0, f'Matthews Correlation Coefficient: {mcc:.4f}', fontsize=20, ha='center', va='center',color="crimson")
plt.axis('off')

# Set the x-axis limits
plt.xlim(-1, 1)
plt.ylim(-1,1)

plt.show()

# %%
num_classes = all_outputs.shape[1]
brier_losses = []

for i in range(num_classes):
    y_true_binary = (np.array(all_labels) == i).astype(int)
    y_prob = all_outputs[:, i]
    brier_loss = brier_score_loss(y_true_binary, y_prob)
    brier_losses.append(brier_loss)

# Calculate average Brier Score Loss
avg_brier_loss = np.mean(brier_losses)

# Create a DataFrame for Seaborn
df = pd.DataFrame({
    'Class': range(num_classes),
    'Brier Score Loss': brier_losses
})

# Plotting
plt.figure(figsize=(12, 6), facecolor='white')
sns.set_style("whitegrid")

# Create the bar plot
ax = sns.barplot(x='Class', y='Brier Score Loss', data=df, palette='gnuplot2')

# Customize the plot
plt.title('Brier Score Loss for Each Class', fontsize=16)
plt.xlabel('Class', fontsize=12)
plt.ylabel('Brier Score Loss', fontsize=12)

# Add value labels on top of each bar
for i, v in enumerate(brier_losses):
    ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')

# Add a horizontal line for the average Brier Score Loss
plt.axhline(y=avg_brier_loss, color='r', linestyle='--', label=f'Average: {avg_brier_loss:.4f}')

# Add text for average Brier Score Loss
plt.text(num_classes-1, avg_brier_loss, f'Average: {avg_brier_loss:.4f}', 
         fontsize=10, va='bottom', ha='right', color='r')

plt.legend()
plt.tight_layout()
plt.show()

# %%
num_classes = all_outputs.shape[1]
brier_losses = []

for i in range(num_classes):
    y_true_binary = (np.array(all_labels) == i).astype(int)
    y_prob = all_outputs[:, i]
    brier_loss = brier_score_loss(y_true_binary, y_prob)
    brier_losses.append(brier_loss)

# Calculate average Brier Score Loss
avg_brier_loss = np.mean(brier_losses)

# Create a DataFrame for Seaborn
df = pd.DataFrame({
    'Class': range(num_classes),
    'Brier Score Loss': brier_losses
})

# Plotting the pie chart
plt.figure(figsize=(8, 8), facecolor='white')
plt.pie(brier_losses, labels=df['Class'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", num_classes))
plt.title('Brier Score Loss Distribution for Each Class', fontsize=16)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Show average Brier Score Loss as a legend
plt.legend([f'Average: {avg_brier_loss:.4f}'], loc='upper right')
plt.tight_layout()
plt.show()

# Print out the Brier Score Loss for each class
for i, loss in enumerate(brier_losses):
    print(f"Class {i}: Brier Score Loss = {loss:.4f}")
print(f"Average Brier Score Loss: {avg_brier_loss:.4f}")

# %%
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize



# Get the number of classes
num_classes = all_outputs.shape[1]

# Binarize the labels for multiclass ROC
all_labels_bin = label_binarize(all_labels, classes=range(num_classes))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_outputs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))

# Use the new method to get colormaps
colors = plt.colormaps['Set1'](np.linspace(0, 1, num_classes))

for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Print AUC for each class
for i in range(num_classes):
    print(f"AUC for class {i}: {roc_auc[i]:.4f}")

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(all_labels_bin.ravel(), all_outputs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print(f"Micro-average AUC: {roc_auc['micro']:.4f}")

# %%
from sklearn.metrics import precision_recall_curve, average_precision_score

# Get the number of classes
num_classes = all_outputs.shape[1]

# Binarize the labels for multiclass precision-recall curve
all_labels_bin = label_binarize(all_labels, classes=range(num_classes))

# Compute Precision-Recall curve and average precision for each class
precision = dict()
recall = dict()
average_precision = dict()

for i in range(num_classes):
    precision[i], recall[i], _ = precision_recall_curve(all_labels_bin[:, i], all_outputs[:, i])
    average_precision[i] = average_precision_score(all_labels_bin[:, i], all_outputs[:, i])

# Compute micro-average Precision-Recall curve and average precision
precision["micro"], recall["micro"], _ = precision_recall_curve(all_labels_bin.ravel(), all_outputs.ravel())
average_precision["micro"] = average_precision_score(all_labels_bin, all_outputs, average="micro")

# Plot Precision-Recall curves
plt.figure(figsize=(10, 8))

# Use the new method to get colormaps
colors = plt.colormaps['Set1'](np.linspace(0, 1, num_classes))

for i, color in zip(range(num_classes), colors):
    plt.plot(recall[i], precision[i], color=color, lw=2,
             label=f'Precision-Recall curve of class {i} (AP = {average_precision[i]:.2f})')

plt.plot(recall["micro"], precision["micro"], color='gold', lw=2,
         label=f'Micro-average Precision-Recall curve (AP = {average_precision["micro"]:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multiclass Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

# Print Average Precision (AP) for each class
for i in range(num_classes):
    print(f"Average Precision for class {i}: {average_precision[i]:.4f}")

print(f"Micro-average Precision: {average_precision['micro']:.4f}")

# %%
# Function to convert a tensor image to a NumPy array for visualization
def imshow(img):
    """Function to convert a tensor image to a NumPy array for visualization."""
    if isinstance(img, np.ndarray):
        npimg = img
    else:
        img = img / 2 + 0.5  # Unnormalize
        npimg = img.numpy()

    if npimg.shape[0] == 3:  # Check if the image has 3 channels
        return np.transpose(npimg, (1, 2, 0))  # Rearrange dimensions for matplotlib
    else:
        return npimg

# Define the classes

class_to_idx, classes = class_dict, class_names

# Set the model to evaluation mode
vit.eval()

# Get a batch of images
dataiter = iter(test_dataloader)
images, labels = next(dataiter)

# Move images and labels to the appropriate device
images, labels = images.to(device), labels.to(device)

# Get predictions
with torch.no_grad():
    outputs = vit(images)
    _, predicted = torch.max(outputs, 1)

# Convert to numpy for easy indexing
images = images.cpu().numpy()
labels = labels.cpu().numpy()
predicted = predicted.cpu().numpy()

# Plotting the images along with true and predicted labels
plt.figure(figsize=(30, 30))
number_images = (5, 5)
num_samples = min(number_images[0] * number_images[1], len(images))

for i in range(num_samples):
    plt.subplot(number_images[0], number_images[1], i + 1)
    plt.axis("off")
    
    true_label = classes[labels[i]]
    predicted_label = classes[predicted[i]]
    
    # Set title color based on whether prediction is correct
    color = "darkgreen" if true_label == predicted_label else "red"
    
    plt.title(f"True: {true_label}\nPredicted: {predicted_label}", color=color, fontsize=16)
    plt.imshow(imshow(images[i]))

plt.tight_layout()
plt.show()

# Calculate overall accuracy
accuracy = (predicted == labels).mean()
print(f"Overall accuracy: {accuracy:.2f}")

# %%
# Save everything you need to recreate and use the model
torch.save({
    'model_state_dict': vit.state_dict(),
    'model_config': {
        'img_size': 224,
        'in_channels': 3,
        'patch_size': 16,
        'num_transformer_layers': 12,
        'embedding_dim': 768,
        'mlp_size': 3072,
        'num_heads': 12,
        'num_classes': 9  # Your number of car classes
    }
}, 'vit_model_balance.pth')

# %%
# # Load the model
# checkpoint = torch.load('vit_model.pth', map_location='cpu')

# # Recreate model with exact same config
# model = ViT(**checkpoint['model_config'])

# # Load the trained weights
# model.load_state_dict(checkpoint['model_state_dict'])

# # Set to evaluation mode for inference
# model.eval()

# %%
