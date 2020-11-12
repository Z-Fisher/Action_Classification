import os
import shutil

for file in os.listdir('splits'):
    foldername = file.split('.')[0]
    if not foldername[-1] == '1':
        continue
    foldername = foldername[:-12]

    path_test = "test/" + foldername
    path_train = "train/" + foldername

    if not os.path.isdir(path_test):
        os.mkdir(path_test)
    if not os.path.isdir(path_train):
        os.mkdir(path_train)

    train_list = []
    test_list = []
    with open("splits/" + file, "r") as file_r:
        for line in file_r:
            id = line[-3]
            if id == '0' or id == '1':
                train_list.append(line[:-4])
            if id == '2':
                test_list.append(line[:-4])

    for vid_file in test_list:
        path_src = 'hmdb51_org/' + foldername + "/" + vid_file
        path_dst = path_test + "/" + vid_file
        if not os.path.isfile(path_dst):
            shutil.copy2(path_src, path_dst)

    for vid_file in train_list:
        path_src = 'hmdb51_org/' + foldername + "/" + vid_file
        path_dst = path_train + "/" + vid_file
        if not os.path.isfile(path_dst):
            shutil.copy2(path_src, path_dst)
