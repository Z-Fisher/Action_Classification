# Two-Stream Human Action Classification performed using ConvNets.
View "dataset_curation.md" to find instructions for setting up the datasets.

## Full Dataset
Data sampled by every consecutive frame, without defining a set framerate or skipping frames.

## Sampled Dataset
Data sampled at 6 fps, skipping 1 second between frames

## Models
All necessary models can be found in this gdrive link: https://drive.google.com/drive/folders/1jeQpvKQttygRkUPmZcLOl1YYjefw0CH3?usp=sharing

## Instructions for Training and Testing

Run the following files to train spatial, temporal, and fuseNet on the full dataset. The code blocks have titles which explain their function.
- Spatial_Stream_Full.ipynb
- Temporal_Stream_Full.ipynb
- FuseNet_Full.ipynb

Run the following files to train spatial, temporal, and fuseNet on the sampled dataset. The code blocks have titles which explain their function.
- Spatial_Stream_Sampled.ipynb
- Temporal_Stream_Sampled.ipynb
- FuseNet_Sampled.ipynb


## Plotting Model Performance
Any model's performance (training loss, validation loss, accuracy) can be plotted using the plotting.py function. Make sure to update the path to model. The plotting code block at the end of each notebook file can also be used instead.