import os

import numpy as np
import pandas as pd
from pytubefix import YouTube

import cv2
from keras.api.applications.vgg16 import VGG16, preprocess_input


# no longer using this, we assume user already has the videos downloaded
def download_features(path, tmp_path="tmp/"):
    limit = 10
    i = 0

    df = pd.read_csv(path)
    youtube_ids = df["youtube_id"].tolist()
    labels = df["label"].tolist()

    # dictionary that maps each label to a number
    unique_labels = {str(label): i for (i, label) in enumerate(np.unique(labels))}

    x_result = []  # will hold the feature vectors
    y_result = []  # will hold entry labels

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    for index, youtube_id in enumerate(youtube_ids):
        if i == limit:
            break
        try:
            yt = YouTube("https://www.youtu.be/" + youtube_id)
            yt.check_availability()
        except Exception as e:
            print(f"Error with {youtube_id}: {e}")
            # youtube_ids.pop(index)
            # labels.pop(index)
            continue
        yt.streams.filter(progressive=True, file_extension="mp4").order_by(
            "resolution"
        ).desc().first().download(output_path=tmp_path, filename=youtube_id + ".mp4")

        feat = feature_vector(tmp_path + youtube_id + ".mp4")
        x_result.append(feat)  # appends a row of features
        y_result.append(unique_labels[labels[index]])
        i += 1
    return np.array(x_result), np.array(y_result)


def load_features_from_files(x_path="./x.csv", y_path="./y.csv", z_path="./z.csv"):
    x = pd.read_csv(x_path).values
    y = pd.read_csv(y_path).values.flatten()
    z = pd.read_csv(z_path)["label_name"].tolist()
    return x, y, z


def extract_features_to_files(
    csv_path="source/train_subset.csv",
    video_path="videos/train/train_subset",
    limit=None,
):
    # for testing purposes
    counter = 0

    df = pd.read_csv(csv_path)

    # sorts both file list and dataframe by youtube_id, so they match when assigning labels
    video_files = sorted([f for f in os.listdir(video_path)])
    df = df.sort_values(by="youtube_id").reset_index(drop=True)

    # dictionary that maps each label to a number
    unique_labels = {str(label): i for (i, label) in enumerate(np.unique(df["label"]))}

    # we convert it to a list, so cluster i has label at index i
    unique_labels = [label_name for label_name, _ in unique_labels.items()]

    x_result = []  # will hold the feature vectors
    y_result = []  # will hold entry labels
    
    for index, video_file in enumerate(video_files):
        if limit is not None and counter == limit:
            break
        try:
            print(f"Processing {index}: {video_file}")
            compound_path = os.path.join(video_path, video_file)

            video_features = feature_vector(compound_path, sampling_rate=8)

            # avoid corrupted videos
            if isinstance(video_features, np.float64) or isinstance(video_features, float):
                print(f"Video {video_file} is corrupted. Continuing with next video.")
                continue
            # this happened with some videos like 1azVHxhCCU0.mp4 and 1IQCtz7ZUzo.mp4, the program stopped and we lost like 2 hours of processing

            x_result.append(video_features)  # appends a row of features
            y_result.append(
                unique_labels.index(df["label"][index])
            )  # gets corresponding label
            counter += 1
        except Exception as e:
            print(f"Error with {video_file}: {e}")
            continue
    x_df = pd.DataFrame(x_result)
    y_df = pd.DataFrame(y_result, columns=["label"])
    z_df = pd.DataFrame(unique_labels, columns=["label_name"])

    x_df.to_csv("x.csv", index=False)
    y_df.to_csv("y.csv", index=False)
    z_df.to_csv("z.csv", index=False)


# sampling rate means how often we take a frame for feature extraction
def feature_vector(video_path, sampling_rate=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_counter = 0

    while cap.isOpened():  # grabs every sampling_rate-th frame
        ret, frame = cap.read()
        if not ret:
            break
        if frame_counter % sampling_rate == 0:
            frames.append(frame)  # adds it to array
        frame_counter += 1

    cap.release()
    frames = np.array(frames)
    model = VGG16(
        weights="imagenet", include_top=False, pooling="avg"
    )  # pooling makes this have WAY less features

    feature_vectors = []
    for frame in frames:  # elmer made this part so no clue bout what it does
        resized_frame = cv2.resize(frame, (224, 224))
        frame_array = np.expand_dims(resized_frame, axis=0)
        frame_array = preprocess_input(frame_array)

        features = model.predict(frame_array, verbose=0)
        feature_vectors.append(features.flatten())

    feature_vectors = np.array(feature_vectors)

    video_feature_vector = np.mean(feature_vectors, axis=0)
    return video_feature_vector


# removes everything after the _
def clean_video_names(video_path):
    videos = [f for f in os.listdir(video_path)]
    for video in videos:
        clean_name = video.split("_")[0]
        os.rename(
            os.path.join(video_path, video),
            os.path.join(video_path, clean_name + ".mp4"),
        )