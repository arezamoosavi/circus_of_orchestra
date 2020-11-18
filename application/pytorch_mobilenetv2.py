import time
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, models, transforms
import numpy as np
import os

import logging

logging.basicConfig(
    filename="logs.log",
    level=logging.DEBUG,
    format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
)


# parameters
batch_size = 20
epochs = 60
workers = 4 if os.name == "nt" else 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = "~/dataset"

dataset = datasets.ImageFolder(
    data_dir,
    transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (255, 255, 255)),
        ]
    ),
)

class_names = dataset.classes
img_inds = np.arange(len(dataset))

split_rate = 5
from_indx = int(1 / split_rate * len(img_inds))

test_inds = img_inds[2 * from_indx : 3 * from_indx]

train_inds = [x for x in img_inds if x not in test_inds]

# from_indx = int(1/split_rate * len(img_inds))

train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds),
)
test_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(test_inds),
)

premodel = models.mobilenet_v2(pretrained=True)

## freeze the layers
for param in premodel.parameters():
    param.requires_grad = False

number_of_features = premodel.classifier[1].in_features

features = list(premodel.classifier.children())[:-1]  # Remove last layer
features.extend([torch.nn.Linear(number_of_features, len(class_names))])

premodel.classifier = torch.nn.Sequential(*features)
premodel = premodel.to(device)

# optimizer = optim.Adam(premodel.parameters(), lr=0.001)
optimizer = optim.SGD(premodel.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

# validation function
def validate(model, test_dataloader):
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    for _, data in enumerate(test_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        output = model(data)
        loss = criterion(output, target)

        val_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        val_running_correct += (preds == target).sum().item()

    val_loss = val_running_loss / len(test_dataloader.dataset)
    val_accuracy = 100.0 * val_running_correct / len(test_dataloader.dataset)

    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.2f}")

    return val_loss, val_accuracy


# training function
def fit(model, train_dataloader):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for _, data in enumerate(train_dataloader):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / len(train_dataloader.dataset)
    train_accuracy = 100.0 * train_running_correct / len(train_dataloader.dataset)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")

    return train_loss, train_accuracy


train_loss, train_accuracy = [], []
val_loss, val_accuracy = [], []

start = time.time()
logging.info("start of training....")

for epoch in range(epochs):
    logging.info(f"start of training for epoch {epoch}")

    train_epoch_loss, train_epoch_accuracy = fit(premodel, train_loader)
    logging.info(
        f"training for epoch {epoch} results: train_epoch_loss :  {train_epoch_loss}, train_epoch_accuracy : {train_epoch_accuracy}"
    )

    val_epoch_loss, val_epoch_accuracy = validate(premodel, test_loader)
    logging.info(
        f"validating for epoch {epoch} results: val_epoch_loss :  {val_epoch_loss}, val_epoch_accuracy : {val_epoch_accuracy}"
    )

    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
    val_loss.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

    logging.info(f"end of training for epoch {epoch}")

end = time.time()

logging.info("end of training....")


print((end - start) / 60, "minutes")

torch.save(premodel, "mobilenetv2")
