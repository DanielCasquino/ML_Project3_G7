import os
import gdown


class DataLoader:
    def __init__(self):
        self.train_gdrive_id = "1rkRaRlPB5INfNVf_GPhVar_hymUWG9bZ"
        self.test_gdrive_id = "1kBbC2xMBopRfr393lRm0kP-t3qpoqY-T"
        self.train_path = "videos/train/"
        self.test_path = "videos/test/"
        self.train_file = "train.zip"
        self.test_file = "test.zip"

    def download_data(self):
        if not os.path.exists(self.train_path):
            # download and extract train data
            os.makedirs(self.train_path)
            if not os.path.isfile(f"{self.train_path}{self.train_file}"):
                gdown.download(
                    id=self.train_gdrive_id,
                    output=f"{self.train_path}{self.train_file}",
                    quiet=False,
                )
        if not os.path.exists(self.test_path):
            # download and extract test data
            os.makedirs(self.test_path)
            if not os.path.isfile(f"{self.test_path}{self.test_file}"):
                gdown.download(
                    id=self.test_gdrive_id,
                    output=f"{self.test_path}{self.test_file}",
                    quiet=False,
                )
