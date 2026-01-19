import os
from numpy import mean
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

from time import time
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import logging

from sklearn.metrics import classification_report, accuracy_score
from settings import settings

logger = logging.getLogger()

matplotlib.use("Agg")


def run_emotion_recognition_model(model_class: nn.Module, num_epochs: int, batch_size: int):

    initialized_model = model_class()
    preprocess = initialized_model.weights.transforms()

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            preprocess,
        ]
    )

    test_transform = transforms.Compose([preprocess])

    dataset = datasets.ImageFolder(settings.DATASET_PATH, transform=None)
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    test_dataset.dataset.transform = test_transform

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=5)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # Configure device
    model = model_class().to(device)
    criterion = nn.CrossEntropyLoss()  # Specify the loss layer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_loss = []

    def train(model, loader, epoch_num):  # Train the model
        model.train()  # Set the model to training mode

        running_loss = []
        start = time()
        for batch, label in tqdm(loader):
            batch = batch.to(device)
            label = label.to(device)
            optimizer.zero_grad()  # Clear gradients from the previous iteration
            pred = model(batch)  # This will call Network.forward() that you implement
            loss = criterion(pred, label)  # Calculate the loss
            running_loss.append(loss.item())
            loss.backward()  # Backprop gradients to all tensors in the network
            optimizer.step()  # Update trainable weights
        logger.info("Epoch {} loss:{}".format(epoch_num + 1, mean(running_loss)))  # Print the average loss for this epoch
        train_loss.append(mean(running_loss))
        logger.info(f"Took {time()-start}s to run")

    def evaluate(model, loader):  # Evaluate accuracy on validation / test set
        model.eval()  # Set the model to evaluation mode
        running_loss = []

        all_preds = []
        all_labels = []
        running_loss = []

        with torch.no_grad():
            for batch, labels in tqdm(loader):
                batch = batch.to(device)
                labels = labels.to(device)

                preds = model(batch)
                loss = criterion(preds, labels)
                running_loss.append(loss.item())

                predicted = torch.argmax(preds, dim=1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        logger.info(f"Overall emotion accuracy: {acc:.4f}")

        
        report = classification_report(all_labels, all_preds, target_names=settings.LABELS, digits=4)
        logger.info("Emotion report:\n")
        logger.info(report)

        all_satisfaction_preds = [settings.EMOTION_TO_SATISFACTION[settings.LABELS[e]] for e in all_preds]
        all_satisfaction_labels = [settings.EMOTION_TO_SATISFACTION[settings.LABELS[e]] for e in all_labels]

        acc = accuracy_score(all_labels, all_preds)
        logger.info(f"Overall satisfaction accuracy: {acc:.4f}")

        satisfaction_report = classification_report(
            all_satisfaction_labels, all_satisfaction_preds, target_names=settings.SATISFACTION_LABELS, digits=4
        )
        logger.info("Satisfaction report:\n")
        logger.info(satisfaction_report)

    for epoch_num in range(num_epochs):
        train(model, trainloader, epoch_num)
        if (epoch_num+1) % 5 == 0:
            logger.info("Evaluate on validation set...")
            evaluate(model, valloader)
            logger.info("Evaluate on test set")
            evaluate(model, testloader)
