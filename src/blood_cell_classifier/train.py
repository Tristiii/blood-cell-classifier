from models import DenseNet121
from utils import get_model

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor

from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

from medmnist import BloodMNIST

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Import datasets
train_dataset = BloodMNIST(
    split='train',
    size=28,
    transform=ToTensor(),
    download=True
)

val_dataset = BloodMNIST(
    split='val',
    size=28,
    transform=ToTensor(),
    download=True
)

test_dataset = BloodMNIST(
    split='test',
    size=28,
    transform=ToTensor(),
    download=True
)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

torch.manual_seed(42)

# Creating model
model = DenseNet121().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

accuracy_fn = Accuracy(task='multiclass', num_classes=8).to(device)
precision_fn = Precision(task='multiclass', num_classes=8).to(device)
recall_fn = Recall(task='multiclass', num_classes=8).to(device)
f1_fn = F1Score(task='multiclass', num_classes=8).to(device)
auroc_fn = AUROC(task='multiclass', num_classes=8).to(device)

for epoch in range(30):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.squeeze(-1).long().to(device)
        y_pred = model(X_batch)

        loss     = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    total_loss /= len(train_dataloader)

    model.eval()
    with torch.inference_mode():
        for X_batch, y_batch in val_dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.squeeze(-1).long().to(device)
            y_pred = model(X_batch)

            accuracy_fn.update(y_pred, y_batch)
            precision_fn.update(y_pred, y_batch)
            recall_fn.update(y_pred, y_batch)
            f1_fn.update(y_pred, y_batch)

            auroc_fn.update(
                torch.softmax(y_pred, dim=1),
                y_batch
            )

    accuracy = accuracy_fn.compute()
    precision = precision_fn.compute()
    recall = recall_fn.compute()
    f1 = f1_fn.compute()
    auroc = auroc_fn.compute()

    accuracy_fn.reset()
    precision_fn.reset()
    recall_fn.reset()
    f1_fn.reset()
    auroc_fn.reset()

    print(
        f'Epoch {epoch} | '
        f'Loss (training): {total_loss:.4f} | '
        f'Accuracy: {accuracy:.1%} | '
        f'Precision: {precision:.1%} | '
        f'Recall: {recall:.1%} | '
        f'F1: {f1:.3f} | '
        f'AUROC: {auroc:.3f}'
    )

with torch.inference_mode():
    for X_batch, y_batch in test_dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.squeeze(-1).long().to(device)
        y_pred = model(X_batch)

        accuracy_fn.update(y_pred, y_batch)
        precision_fn.update(y_pred, y_batch)
        recall_fn.update(y_pred, y_batch)
        f1_fn.update(y_pred, y_batch)
        auroc_fn.update(torch.softmax(y_pred, dim=1), y_batch)

test_accuracy = accuracy_fn.compute()
test_precision = precision_fn.compute()
test_recall = recall_fn.compute()
test_f1 = f1_fn.compute()
test_auroc = auroc_fn.compute()

print(
    f'Test | '
    f'Accuracy: {test_accuracy:.1%} | '
    f'Precision: {test_precision:.1%} | '
    f'Recall: {test_recall:.1%} | '
    f'F1: {test_f1:.3f} | '
    f'AUROC: {test_auroc:.3f}'
)

# Save state dict of model
torch.save(model.state_dict(), get_model() / 'densenet121_state_dict.pth')