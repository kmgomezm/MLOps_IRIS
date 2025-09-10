
import torch

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

# Check if the directory "./model" exists
if not os.path.exists("./model"):
    # If it doesn't exist, create it
    os.makedirs("./model")

# Data parameters for Iris dataset
num_classes = 3  # setosa, versicolor, virginica
input_shape = 4  # sepal_length, sepal_width, petal_length, petal_width

def build_model_and_log(config, model, model_name="MLP", model_description="Simple MLP"):
    with wandb.init(project="MLOps_Iris", 
        name=f"initialize-Model-ExecId-{args.IdExecution}", 
        job_type="initialize-model", 
        config=config,
        tags=["iris", "model-initialization", "pytorch"]) as run:
        config = wandb.config

        model_artifact = wandb.Artifact(
            model_name, type="model",
            description=model_description,
            metadata=dict(config))

        name_artifact_model = f"initialized_model_{model_name}.pth"

        # Save model state dict
        torch.save(model.state_dict(), f"./model/{name_artifact_model}")
        
        # Add file to artifact
        model_artifact.add_file(f"./model/{name_artifact_model}")

        # Save to wandb
        wandb.save(name_artifact_model)

        # Log model summary information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.log({
            "model_total_parameters": total_params,
            "model_trainable_parameters": trainable_params,
            "model_input_shape": input_shape,
            "model_output_classes": num_classes
        })

        run.log_artifact(model_artifact)


# MLP Configuration for Iris dataset
model_config = {
    "input_shape": input_shape,      # 4 features for Iris
    "hidden_layer_1": 16,            # Smaller hidden layer for Iris
    "hidden_layer_2": 8,             # Smaller second hidden layer
    "num_classes": num_classes       # 3 classes for Iris
}

# Create model instance
model = Classifier(**model_config)

# Build and log the model
build_model_and_log(
    model_config, 
    model, 
    "iris-classifier", 
    "Neural Network Classifier for Iris dataset with 4 input features and 3 output classes"
)