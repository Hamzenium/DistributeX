from tensorflow import keras
from tensorflow.keras import layers
from flwr_datasets import FederatedDataset, IidPartitioner

def load_data(partition_id: int, num_partitions: int):
    partitioner = IidPartitioner(num_partitions=num_partitions)
    fds = FederatedDataset(
        dataset="mnist",
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
    return x_train, y_train, x_test, y_test

def load_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile("adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model