
import torch
from tqdm.auto import tqdm


def train_step(model, dataloader, loss_fn, optimizer, device):

    model.train()

    train_loss = 0
    correct = 0

    for X, y in tqdm(dataloader, desc="Training"):

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        y_pred = model(X)

        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)

        y_pred_class = y_pred.argmax(dim=1)
        correct += (y_pred_class == y).sum().item()

    train_loss /= len(dataloader.dataset)
    train_acc = correct / len(dataloader.dataset)

    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device):

    model.eval()

    test_loss = 0
    correct = 0

    with torch.inference_mode():

        for X, y in tqdm(dataloader, desc="Testing"):

            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)

            test_loss += loss.item() * X.size(0)

            y_pred_class = y_pred.argmax(dim=1)
            correct += (y_pred_class == y).sum().item()

    test_loss /= len(dataloader.dataset)
    test_acc = correct / len(dataloader.dataset)

    return test_loss, test_acc
