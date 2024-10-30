r"""
DataLoader
===

DEPRECATED. USE FEATUREX INSTEAD.


Manages downloading, unzipping, and loading of train and test datasets.

This module exports:
    - DataLoader class

Usage
-----
To create an instance of the DataLoader class, do::

    >>> loader = DataLoader(train_path="videos/train/", test_path="videos/test/")

Constructor parameters are (in order):
    - `train_path` the path to the training data directory
    - `test_path` the path to the testing data directory
You can replace these in case you want to store or have the data in a different location.

To load the data, do::

    >>> loader.load_data()

If the data is not present, it will be downloaded and unzipped automatically.
"""

import os
import gdown
import zipfile


class DataLoader:
    def __init__(self, train_path="videos/train/", test_path="videos/test/"):
        self.train_gdrive_id = "1rkRaRlPB5INfNVf_GPhVar_hymUWG9bZ"
        self.test_gdrive_id = "1kBbC2xMBopRfr393lRm0kP-t3qpoqY-T"
        self.train_path = train_path
        self.test_path = test_path
        self.train_file = "train.zip"
        self.test_file = "test.zip"

    def load_data(self):
        # TODO
        # attempt to load one video from both test and train, if one fails, call i_dont_have_the_data
        # else call i_have_the_data
        return 0, 0

    def i_have_the_data(self):
        # TODO
        # return data as x matrix and y vector (labels)
        pass

    def i_dont_have_the_data(self):
        print("Datasets not present. Attempting to download.")
        self.download_data()
        self.unzip_data()

    # Unzips the downloaded data
    def unzip_data(self):
        print("Unzipping data.")
        for path, file in [
            (self.train_path, self.train_file),
            (self.test_path, self.test_file),
        ]:
            print(f"Unzipping {file} to {path}")
            zip_file_path = os.path.join(path, file)
            if os.path.isfile(zip_file_path):
                with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                    zip_ref.extractall(path)
                os.remove(zip_file_path)
        print("Unzipping complete.")

    # Downloads both train and test sets
    def download_data(self):
        if not os.path.exists(self.train_path):
            # Downloads train data
            os.makedirs(self.train_path)
            if not os.path.isfile(f"{self.train_path}{self.train_file}"):
                user_input = input(
                    "WARNING: About 14GB of free storage is required to download the train dataset. Proceed? (Y/n): "
                )
                if user_input.lower() != "y":
                    print("Download cancelled.")
                    return
                print("Downloading train data.")
                gdown.download(
                    id=self.train_gdrive_id,
                    output=f"{self.train_path}{self.train_file}",
                    quiet=False,
                )
                print("Train data downloaded.")
        else:
            print("Train data already present.")
        if not os.path.exists(self.test_path):
            # Downloads test data
            os.makedirs(self.test_path)
            if not os.path.isfile(f"{self.test_path}{self.test_file}"):
                print(
                    "WARNING: About 2GB of free storage is required to download the test dataset. Proceed? (Y/n)"
                )
                if user_input.lower() != "y":
                    print("Download cancelled.")
                    return
                print("Downloading test data.")
                gdown.download(
                    id=self.test_gdrive_id,
                    output=f"{self.test_path}{self.test_file}",
                    quiet=False,
                )
                print("Test data downloaded.")
        else:
            print("Test data already present.")
