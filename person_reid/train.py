import argparse
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import CUHK03Dataset
from src.model import ReIDSiamese
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm


def read_data(dataset_path):
    path = os.path.join(dataset_path, "pairs.csv")
    data = pd.read_csv(path)

    return data


def train(dataset_path, output_path, num_epochs, batch_size, num_workers, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((160, 80)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Model
    model = ReIDSiamese().to(device)

    # Dataset and DataLoader
    dataset = CUHK03Dataset(
        data=read_data(dataset_path),
        image_folder_path=os.path.join(dataset_path, "archive", "images_labeled"),
        transform=transform,
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Loss and Optimizer
    criterion = nn.BCELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr)

    # Training loop

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for img1, img2, labels in tqdm(train_loader):
            img1, img2 = img1.to(device), img2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader)}")

    torch.save(
        model.state_dict(),
        os.path.join(
            output_path, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_siamese_model.pth"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Validate that dataset_path and output_path are valid directories
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        raise ValueError(f"Invalid dataset path: {args.dataset_path}")

    if not os.path.isdir(args.output_path):
        raise ValueError(f"Invalid output path: {args.output_path}")

    # Train the model
    train(
        args.dataset_path,
        args.output_path,
        args.num_epochs,
        args.batch_size,
        args.num_workers,
        args.lr,
    )
