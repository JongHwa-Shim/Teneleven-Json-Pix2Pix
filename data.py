from os.path import join

from dataset import DatasetFromFolder, JsonDatasetTest, JsonDatasetTrain, RPLANDataset, RPLANDataset_test
from dataset import transform

def get_training_set(root_dir):
    train_dir = join(root_dir, "train")

    # return DatasetFromFolder(train_dir, direction)
    return JsonDatasetTrain(train_dir, transform = transform)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")

    # return DatasetFromFolder(test_dir, direction)
    return JsonDatasetTest(test_dir)
