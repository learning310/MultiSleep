import os
import torch
import numpy as np

data_dir = r"C:\Users\syfmark\Desktop\EEG\WorkSpace\SSL\data_preprocessing\sleep-edf\sleepEDF20_EOG_subjects"
output_dir = r"./data/eog/"
edf20_permutation = np.array(
    [14, 5, 4, 17, 8, 7, 19, 12, 0, 15, 16, 9, 11, 10, 3, 1, 6, 18, 2, 13]
)

os.makedirs(output_dir)
files = os.listdir(data_dir)
files = np.array([os.path.join(data_dir, i) for i in files])
files.sort()

# to have the same results as in the paper
# files = files[edf20_permutation]

len_train = int(len(files) * 0.8)

######## TRAINing files ##########
training_files = files[:len_train]
# load files
X_train = np.load(training_files[0])["x"]
y_train = np.load(training_files[0])["y"]

for np_file in training_files[1:]:
    X_train = np.vstack((X_train, np.load(np_file)["x"]))
    y_train = np.append(y_train, np.load(np_file)["y"])

data_save = dict()
data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_train)
torch.save(data_save, os.path.join(output_dir, "train.pt"))


######## TesT ##########
test_files = files[len_train:]
# load files
X_train = np.load(test_files[0])["x"]
y_train = np.load(test_files[0])["y"]

for np_file in test_files[1:]:
    X_train = np.vstack((X_train, np.load(np_file)["x"]))
    y_train = np.append(y_train, np.load(np_file)["y"])

data_save = dict()
data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
data_save["labels"] = torch.from_numpy(y_train)
torch.save(data_save, os.path.join(output_dir, "test.pt"))
