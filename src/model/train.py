import torch
import torch.nn.functional as F
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Import the model class from the main file
from src.Classifier import Classifier

import os
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def read(data_dir, split):
    """
    Read data from a directory and return a TensorDataset object.

    Args:
    - data_dir (str): The directory where the data is stored.
    - split (str): The name of the split to read (e.g. "train", "valid", "test").

    Returns:
    - dataset (TensorDataset): A TensorDataset object containing the data.
    """
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)



def train(model, train_loader, valid_loader, config):
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters())
    model.train()
    example_ct = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            example_ct += len(data)

            if batch_idx % config.batch_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    batch_idx / len(train_loader), loss.item()))
                
                train_log(loss, example_ct, epoch)

        # evaluate the model on the validation set at each epoch
        loss, accuracy = test(model, valid_loader)  
        test_log(loss, accuracy, example_ct, epoch)

    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # No need to flatten data for Iris (already flat)
            
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum')  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)
    # where the magic happens
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)
    # where the magic happens
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}/{accuracy:.3f}")

def evaluate(model, test_loader):
    """
    ## Evaluate the trained model
    """

    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(model, test_loader.dataset)

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions

def get_hardest_k_examples(model, testing_set, k=10):
    model.eval()
    loader = DataLoader(testing_set, 1, shuffle=False)

    # get the losses and predictions for each item in the dataset
    losses = None
    predictions = None
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            # No need to flatten data for Iris
            
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            
            if losses is None:
                losses = loss.view((1, 1))
                predictions = pred
            else:
                losses = torch.cat((losses, loss.view((1, 1))), 0)
                predictions = torch.cat((predictions, pred), 0)

    argsort_loss = torch.argsort(losses, dim=0).cpu()
    
    # Ensure we don't ask for more examples than we have
    k = min(k, len(testing_set))
    
    highest_k_losses = losses[argsort_loss[-k:]]
    hardest_k_examples = testing_set[argsort_loss[-k:]][0]
    true_labels = testing_set[argsort_loss[-k:]][1]
    predicted_labels = predictions[argsort_loss[-k:]]

    return highest_k_losses, hardest_k_examples, true_labels, predicted_labels


def train_and_log(config,experiment_id='99'):
    with wandb.init(
        project="MLOps_Iris", 
        name=f"Train-Model-ExecId-{args.IdExecution}-ExpId-{experiment_id}", 
        job_type="train-model", 
        config=config,
        tags=["iris", "training", "pytorch"]) as run:
        
        config = wandb.config
        
        data = run.use_artifact('iris-raw:latest')
        data_dir = data.download()

        training_dataset = read(data_dir, "training")
        validation_dataset = read(data_dir, "validation")

        train_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Use Iris classifier model artifact
        model_artifact = run.use_artifact("iris-classifier:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "initialized_model_iris-classifier.pth")
        model_config = model_artifact.metadata
        config.update(model_config)

        model = Classifier(**model_config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
 
 
        train(model, train_loader, validation_loader, config)

        # Save trained model
        model_artifact = wandb.Artifact(
            "iris-trained-model", type="model",
            description="Trained Neural Network model for Iris classification",
            metadata=dict(model_config))

        torch.save(model.state_dict(), "iris_trained_model.pth")
        model_artifact.add_file("iris_trained_model.pth")
        wandb.save("iris_trained_model.pth")

        run.log_artifact(model_artifact)

    return model

    
def evaluate_and_log(experiment_id='99',config=None,):
    
    with wandb.init(project="MLOps_Iris", 
                    name=f"Eval-Model-ExecId-{args.IdExecution}-ExpId-{experiment_id}", 
                    job_type="eval-model", 
                    config=config,
                    tags=["iris", "evaluation", "pytorch"]) as run:
        
        # Load test data
        data = run.use_artifact('iris-raw:latest')
        data_dir = data.download()
        testing_set = read(data_dir, "test")

        test_loader = DataLoader(testing_set, batch_size=32, shuffle=False)

        # Load trained model
        model_artifact = run.use_artifact("iris-trained-model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "iris_trained_model.pth")
        model_config = model_artifact.metadata

        model = Classifier(**model_config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        loss, accuracy, highest_losses, hardest_examples, true_labels, preds = evaluate(model, test_loader)

        # Log final metrics
        run.summary.update({"test_loss": float(loss), "test_accuracy": float(accuracy)})
                
        # Log hardest examples (feature vectors for Iris)
        iris_feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        iris_class_names = ["setosa", "versicolor", "virginica"]
        
        # Create table for hardest examples
        hardest_table = wandb.Table(columns=["features", "true_label", "predicted_label", "loss"])
        
        for example, true_label, pred_label, loss_val in zip(hardest_examples, true_labels, preds, highest_losses):
            features_str = ", ".join([f"{name}: {val:.2f}" for name, val in zip(iris_feature_names, example.cpu().numpy())])
            hardest_table.add_data(
                features_str,
                iris_class_names[int(true_label)],
                iris_class_names[int(pred_label)],
                float(loss_val)
            )
        
        wandb.log({"hardest_examples": hardest_table})

# Training configuration experiments
epochs = [50, 100, 150]
learning_rates = [0.001, 0.01, 0.1]

for id, (epoch, lr) in enumerate(zip(epochs, learning_rates)):
    print(f"\n🔥 Experiment {id+1}: Epochs={epoch}, LR={lr}")
    
    train_config = {
        "batch_size": 16,  
        "epochs": epoch,
        "batch_log_interval": 5,  
        "optimizer": "Adam",
        "learning_rate": lr
    }
    
    model = train_and_log(train_config, id)
    evaluate_and_log(id, train_config)    

"""    
train_config = {"batch_size": 128,
                "epochs": 5,
                "batch_log_interval": 25,
                "optimizer": "Adam"}

model = train_and_log(train_config)
evaluate_and_log()
"""