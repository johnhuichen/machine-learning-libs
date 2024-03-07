import torch
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import inspect
import copy


def store_attr():
    curframe = inspect.currentframe()
    prev_locals = curframe.f_back.f_locals
    for k, v in curframe.f_back.f_locals.items():
        if k != "self":
            setattr(prev_locals["self"], k, v)


class Architecture:
    def __init__(self, model, optimizer, loss_func):
        store_attr()


class DataloaderStore:
    def __init__(self, train_dataloader, val_dataloader):
        store_attr()


from tqdm import tqdm


class Trainer:
    def __init__(self, architecture, dataloader_store, metrics=list()):
        store_attr()

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.architecture.model = self.architecture.model.to(self.device)
        self.epoch = 1

    def start(self, num_epoch):
        for epoch in tqdm(range(num_epoch)):
            self.train()
            self.eval()
            self.epoch += 1

    def train(self):
        self.architecture.model.train()

        for xbatch, ybatch in self.dataloader_store.train_dataloader:
            xbatch = xbatch.to(self.device)
            ybatch = ybatch.to(self.device)
            pred = self.architecture.model(xbatch)
            loss = self.architecture.loss_func(pred, xbatch, ybatch)
            loss.backward()
            self.architecture.optimizer.step()
            self.architecture.optimizer.zero_grad()

    def eval(self):
        self.architecture.model.eval()

        with torch.no_grad():
            for xbatch, ybatch in self.dataloader_store.val_dataloader:
                xbatch = xbatch.to(self.device)
                ybatch = ybatch.to(self.device)
                pred = self.architecture.model(xbatch)

                for metric in metrics:
                    metric.accumulate(pred, xbatch, ybatch)

        print(f"epoch={self.epoch}")

        for metric in metrics:
            metric.report()


class Metric:
    def __init__(self, name, calc_func):
        store_attr()
        self.total = 0
        self.count = 0

    def accumulate(self, pred, xbatch, ybatch):
        n = len(xbatch)
        self.count += n
        self.total += self.calc_func(pred, xbatch, ybatch) * n

    def report(self):
        print(f"{self.name}: {self.total/self.count}")


model = MyModel()
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr)


def loss_func(pred, xbatch, ybatch):
    return F.mse_loss(pred, xbatch)


architecture = Architecture(model=model, optimizer=optimizer, loss_func=loss_func)

batch_size = 2000
train_data = torchvision.datasets.FashionMNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
val_data = torchvision.datasets.FashionMNIST(
    root="data", train=False, download=True, transform=ToTensor()
)
train_dataloader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=8
)
val_dataloader = DataLoader(
    val_data, batch_size=batch_size, shuffle=False, num_workers=8
)
dataloader_store = DataloaderStore(train_dataloader, val_dataloader)


def calc_loss(pred, xbatch, ybatch):
    return F.mse_loss(pred, xbatch)


metrics = [Metric("loss", calc_loss)]

trainer = Trainer(architecture, dataloader_store, metrics)
trainer.start(5)
