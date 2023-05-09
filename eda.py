import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io

# Path to the unzipped dataset
data_dir = 'www/data/'

# Path to the metadata file
metadata_path = os.path.join(data_dir, 'metadata.csv')

# Read metadata
metadata = pd.read_csv(metadata_path)

# Set image size
image_size = 64

# Print metadata head
print(metadata.head())

# Print metadata info
print(metadata.info())

# Check for missing values
print(metadata.isnull().sum())

# Visualize the distribution of different skin lesion types
plt.figure(figsize=(10, 6))
sns.countplot(data=metadata, x='dx', order=metadata['dx'].value_counts().index)
plt.title('Distribution of Skin Lesion Types')
plt.xlabel('Skin Lesion Type')
plt.ylabel('Frequency')
plt.show()

# Load some example images
example_images = []
for lesion_type in metadata['dx'].unique():
    example_image_path = metadata[metadata['dx'] == lesion_type].iloc[0]['image_id'] + '.jpg'
    example_image_dir = 'HAM10000_images_part_1' if 'ISIC_00' in example_image_path else 'HAM10000_images_part_2'
    example_image_full_path = os.path.join(data_dir, example_image_dir, example_image_path)
    example_image = io.imread(example_image_full_path)
    example_images.append(example_image)

# Plot example images
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for i, (image, lesion_type) in enumerate(zip(example_images, metadata['dx'].unique())):
    axes[i].imshow(image)
    axes[i].set_title(lesion_type)
    axes[i].axis('off')

plt.show()

# Get some statistics of the images
widths = []
heights = []
ratios = []

for idx, row in metadata.iterrows():
    image_path = row['image_id'] + '.jpg'
    image_dir = 'HAM10000_images_part_1' if 'ISIC_00' in image_path else 'HAM10000_images_part_2'
    image_full_path = os.path.join(data_dir, image_dir, image_path)
    image = io.imread(image_full_path)
    width, height, _ = image.shape
    widths.append(width)
    heights.append(height)
    ratios.append(float(width) / float(height))

metadata['width'] = widths
metadata['height'] = heights
metadata['ratio'] = ratios

print(metadata.describe())

# Plot the distributions of widths, heights and ratios
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

sns.histplot(data=metadata, x='width', kde=True, ax=ax1)
ax1.set_title('Distribution of Image Widths')

sns.histplot(data=metadata, x='height', kde=True, ax=ax2)
ax2.set_title('Distribution of Image Heights')

sns.histplot(data=metadata, x='ratio', kde=True, ax=ax3)
ax3.set_title('Distribution of Image Ratios')

plt.show()

# Plot the distributions of widths, heights and ratios for each skin lesion type
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

# Plot the distribution of widths for each skin lesion type
sns.histplot(data=metadata, x='width', hue='dx', kde=True, ax=ax1)
ax1.set_title('Distribution of Image Widths')

# Plot the distribution of heights for each skin lesion type
sns.histplot(data=metadata, x='height', hue='dx', kde=True, ax=ax2)
ax2.set_title('Distribution of Image Heights')
