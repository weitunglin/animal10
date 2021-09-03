import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchsummary import summary

import os
import argparse
import numpy as np
import pandas as pd
from vit_pytorch import ViT
from datetime import datetime
from dataset import Animal10Dataset
from model import Conv3Layer

step = 1
best_acc = 0

def calculate_accuracy(ground_truth, predictions):
    total_count = ground_truth.size(0)
    correct_count = (ground_truth == predictions).sum().item()
    return correct_count / total_count

def save_model(run_name, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join("~/ml/animal10/checkpoints/", f"{run_name}_checkpoint.bin")
    f = open(model_checkpoint, "w+")
    f.close()
    torch.save(model_to_save.state_dict(), model_checkpoint)

def train(net, dataset, criterion, optimizer, epoch, tb, args, device):
    global step
    running_loss = 0.0
    ground_truth = torch.Tensor([]).to(device)
    predictions = torch.Tensor([]).to(device)
    net.train()

    for i, (images, labels) in enumerate(dataset):
        optimizer.zero_grad()
        outputs = net(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        ground_truth = torch.cat((ground_truth, labels))
        for item in outputs:
            predictions = torch.cat((predictions, torch.argmax(item).unsqueeze(-1)))
        
        running_loss += loss.item()
        if i % 20 == 19:
            print(f"Loss: {running_loss / 20.0:>.3f} [{(i + 1) * len(images)}/{len(dataset) * len(images)}]") 
            tb.add_scalar(f"Training Loss (by global step)", running_loss / 20.0, step)
            step += 1
            running_loss = 0

    accuracy = calculate_accuracy(ground_truth, predictions)
    tb.add_scalar(f"Training Accuracy (by epoch)", accuracy, epoch)
    print(f"Training Accuracy (by epoch): {accuracy}")

def test(net, dataset, epoch, tb, args, device):
    global best_acc
    ground_truth = torch.Tensor([]).to(device)
    predictions = torch.Tensor([]).to(device)
    net.eval()

    for i, (images, labels) in enumerate(dataset):
        with torch.no_grad():
            outputs = net(images)
        
        ground_truth = torch.cat((ground_truth, labels))
        batch_predictions = torch.argmax(outputs, dim=-1)
        predictions = torch.cat((predictions, batch_predictions))
        if i % 20 == 0:
            batch_labels = labels
            print(f"accuracy: {calculate_accuracy(batch_labels, batch_predictions)}\t correct count: {(batch_labels == batch_predictions).sum().item()}")
            print(batch_labels)
            print(batch_predictions)

    accuracy = calculate_accuracy(ground_truth, predictions)
    tb.add_scalar(f"Test Accuracy (by epoch)", accuracy, epoch)
    print(f"Test Accuracy (by epoch): {accuracy}")

    if best_acc < accuracy:
        best_acc = accuracy
        save_model(args.run_name, net)
        print(f"New Best Accurary: {best_acc}")

def main():
    # parse arguments
    print("parsing arguments")
    parser = argparse.ArgumentParser(description="Aninal 10")
    parser.add_argument("--input", type=str, required=True, help="directory to input")
    parser.add_argument("--net", type=str, required=True, help="select the network")
    parser.add_argument("--epochs", type=int, required=True, help="epochs for training")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size for the dataset")
    parser.add_argument("--patch-size", type=int, default=4, help="patch size for vit network")
    parser.add_argument("--tb", type=bool, default=True, help="store result to tensor board")
    parser.add_argument("--run-name", type=str, default=datetime.now().strftime("%Y_%m_%d_%H:%M:%S"), help="run name for tensor board")
    parser.add_argument("--random-seed", type=int, default=42, help="designated random seed to enabel reproducity of the training process")

    args = parser.parse_args()
    print("----- args -----")
    print(args)

    # initialize tensorboard
    print("initialize tensor board")
    tb = SummaryWriter(f"runs/{args.run_name}") if args.tb else None

    # prepare dataset
    print("preparing dataset")
    translate = {"cane": "Dog", "cavallo": "Horse", "elefante": "Elephant", "farfalla": "Butterfly", "gallina": "Chicken", "gatto": "Cat", "mucca": "Cow", "pecora": "Sheep", "scoiattolo": "Squirrel", "ragno": "Spider"}
    categories = os.listdir(args.input)
    origin_data = np.array([])
    for category in categories:
        files = os.listdir(f"{args.input}/{category}")
        for f in files:
            origin_data = np.append(origin_data, [translate[category], f"{args.input}/{category}/{f}"], axis=0)
    origin_data = origin_data.reshape((-1, 2))

    data_df = pd.DataFrame(data=origin_data, columns=["label", "image_path"])
    print("----- dataset -----")
    print(data_df.head())
    print(data_df.describe())

    train_df = data_df.sample(frac=0.8, random_state=args.random_seed)
    print("----- train set -----")
    print(train_df.head())
    print(train_df.describe())
    print(train_df.groupby(["label"]).size())
    train_distribution = train_df.groupby("label").size()
    train_count = train_distribution.sum()
    train_distribution = train_count / (train_distribution * 10)
    test_df = data_df.drop(train_df.index)
    print("----- test set -----")
    print(test_df.head())
    print(test_df.describe())
    print(test_df.groupby(["label"]).size())

    train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    train_dataset = DataLoader(Animal10Dataset(train_df, transform=train_transform), batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dataset = DataLoader(Animal10Dataset(test_df, transform=test_transform), batch_size=args.batch_size, shuffle=False, num_workers=4)

    # initialize network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.net == "vit":
        net = ViT(
            image_size=100*100,
            patch_size=args.patch_size,
            num_classes=10,
            dim=512,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.net == "vit-timm":
        import timm
        net = timm.create_model("vit_small_patch16_224", pretrained=True)
        net.head = nn.Linear(net.head.in_features, 10)
    elif args.net == "conv":
        net = Conv3Layer(num_classes=10)
    
    net = net.to(deivce)
 
    print("----- model summary -----")
    print(summary(net, (3, 100, 100)))

    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(train_distribution))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
 
    sample_images = iter(test_dataset).next()[0]
    tb.add_image("animal 10", make_grid(sample_images))
    tb.add_graph(net, sample_images)

    for epoch in range(1, args.epochs + 1):
        print(f"----- Epoch {epoch} -----")
        train(net, train_dataset, criterion, optimizer, epoch, tb, args, device)
        test(net, test_dataset, epoch, tb, args, device)
        print(f"----- End of epoch {epoch} -----")

    global best_acc
    print(f"Best Accuracy: {best_acc}")

    test_ground_truth = torch.Tensor([]).to(device)
    test_predictions = torch.Tensor([]).to(device)
    with torch.no_grad():
        net.eval()
        for (images, labels) in test_dataset:
            outputs = net(images)
            test_ground_truth = torch.cat((test_ground_truth, labels))
            test_predictions = torch.cat((test_predictions, torch.argmax(outputs, dim=-1)))
    
    confusion_df = pd.crosstab(test_ground_truth, test_predictions, rownames=["Actual"], colnames=["Predicted"], margins=True)
    print("----- confusion matrix -----")
    print(confusion_df)

    tb.close()

if __name__ == "__main__":
    main()
