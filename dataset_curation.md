# How to structure your files before running extract_videos.py

- Place extract_videos.py in the same directory (now referred to as "root") as "hmdb51_org" and the splits folder
- Rename folder containing splits text files to "splits"
- Make a folder named "data" in the root directory
- Make folders named "test" and "train" in the "data" folder
- Make sure hmdb51 dataset folder is named "hmdb51_org" and that everything inside is extracted
- Run extract_videos.py to completion - should see action folders appearing in "test" and "train"

# How to get dataset batch files

- Define data folder and create folder within named "dataset_{}" [number of frames] and folders within named "test" and "train
- Run dataset.py
