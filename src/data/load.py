import torch
import torchvision
from torch.utils.data import TensorDataset
# Testing
import argparse
import wandb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8, val_size=0.1, random_state=42):
    """
    # Load the Iris data and split into train/val/test sets
    """
      
    # Cargar el dataset Iris
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Convertir a tensores de PyTorch
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    # Primera división: train+val vs test
    test_size = 1.0 - train_size - val_size
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Segunda división: train vs val
    val_size_adjusted = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )

    training_set = TensorDataset(X_train, y_train)
    validation_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)
    datasets = [training_set, validation_set, test_set]
    return datasets

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="MLOps_Iris",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        datasets = load()  # separate code for loading the datasets
        names = ["training", "validation", "test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "iris-raw", type="dataset",
            description="raw Iris dataset, split into train/val/test",
            metadata={"source": "sklearn.datasets.load_iris",
                      "sizes": [len(dataset) for dataset in datasets],
                      "features": 4,
                      "classes": 3,
                      "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                      "target_names": ["setosa", "versicolor", "virginica"]})

        for name, data in zip(names, datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing
if __name__ == "__main__":
    load_and_log()