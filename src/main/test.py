def clf_segmentation():
    import os
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from catalyst import dl, metrics
    from torchvision.transforms import ToTensor
    from torchvision.datasets import FashionMNIST

    class ClassifyUnet(nn.Module):

        def __init__(self, in_channels, in_hw, out_features):
            super().__init__()
            self.encoder = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1), nn.Tanh())
            self.decoder = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
            self.clf = nn.Linear(in_channels * in_hw * in_hw, out_features)

        def forward(self, x):
            z = self.encoder(x)
            z_ = z.view(z.size(0), -1)
            y_hat = self.clf(z_)
            x_ = self.decoder(z)
            return y_hat, x_

    model = ClassifyUnet(1, 28, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

    loaders = {
        "train": DataLoader(FashionMNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32),
        "valid": DataLoader(FashionMNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32),
    }

    class CustomRunner(dl.Runner):
        def _handle_batch(self, batch):
            x, y = batch
            x_noise = (x + torch.rand_like(x)).clamp_(0, 1)
            y_hat, x_ = self.model(x_noise)

            loss_clf = F.cross_entropy(y_hat, y)
            iou = metrics.iou(x_, x).mean()
            loss_iou = 1 - iou
            loss = loss_clf + loss_iou
            accuracy01, accuracy03, accuracy05 = metrics.accuracy(y_hat, y, topk=(1, 3, 5))
            self.batch_metrics = {
                "loss_clf": loss_clf,
                "loss_iou": loss_iou,
                "loss": loss,
                "iou": iou,
                "accuracy01": accuracy01,
                "accuracy03": accuracy03,
                "accuracy05": accuracy05,
            }
            
            if self.is_train_loader:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

    runner = CustomRunner()
    runner.train(
        model=model, 
        optimizer=optimizer, 
        loaders=loaders, 
        verbose=False,
    )

def multilabel_clf():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from catalyst import dl

    # sample data
    num_samples, num_features, num_classes = int(1e4), int(1e1), 4
    X = torch.rand(num_samples, num_features)
    y = (torch.rand(num_samples, num_classes) > 0.5).to(torch.float32)

    # pytorch loaders
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    model = torch.nn.Linear(num_features, num_classes)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])

    # model training
    runner = dl.SupervisedRunner()
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        num_epochs=3,
        callbacks=[dl.MultiLabelAccuracyCallback(threshold=0.5)]
    )

def multilabel_segmentation():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from catalyst import dl, metrics
    from catalyst import utils

    # sample data
    num_samples, height, width, num_classes, num_classes_1 = int(1e4), int(128), int(128), 4, 5
    X = torch.rand(num_samples, 1, height, width)
    y_1 = (torch.rand(num_samples, num_classes, height, width) > 0.5).to(torch.float32)
    y_2 = (torch.rand(num_samples, ) * num_classes_1).to(torch.int64)    # pytorch loaders
    print(y_2.shape)
    dataset = TensorDataset(X, y_1, y_2)
    loader = DataLoader(dataset, batch_size=16, num_workers=1)
    loaders = {"train": loader, "valid": loader}

    # model, criterion, optimizer, scheduler
    class Unet(nn.Module):
        def __init__(self, in_channels, num_classes, num_classes_1, in_hw):
            super().__init__()
            self.encoder = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1), nn.Tanh())
            self.decoder = nn.Conv2d(in_channels, num_classes, 3, 1, 1)
            self.clf = nn.Linear(in_channels * in_hw * in_hw, num_classes_1)

        def forward(self, x):
            z = self.encoder(x)
            z_ = z.view(z.size(0), -1)
            y_hat_clf = self.clf(z_)
            x_ = self.decoder(z)
            return x_, y_hat_clf

    model = Unet(1, num_classes, num_classes_1, 128)
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion_1 = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2])
    
    class CustomRunner(dl.Runner):
        def _handle_batch(self, batch):
            x, y1, y2 = batch
            y1_hat, y2_hat = self.model(x)
            # print(y2_hat.shape)
            # print(y2.shape)
            loss_seg = criterion(y1_hat, y1)
            loss_clf = criterion_1(y2_hat, y2)
            loss = loss_seg + loss_clf

            iou = metrics.iou(y1_hat, y1).mean()
            # print(iou.shape)
            dice = metrics.dice(y1_hat, y1).mean()
            # print(dice.shape)
            acc1, acc3, acc5 = metrics.accuracy(y2_hat, y2, topk=(1, 3, 5))
            self.batch_metrics = {
                "loss_clf": loss_clf,
                "loss_seg": loss_seg,
                "loss": loss,
                "iou": iou,
                "dice": dice,
                "acc01": acc1,
                "acc03": acc3,
                "acc05": acc5
            }
            
            # if self.is_train_loader:
            #     loss.backward()
            #     self.optimizer.step()
            #     self.optimizer.zero_grad()

    # model training
    runner = CustomRunner(device=utils.get_device())
    runner.train(
        model=model,
        # criterion={
        #     'seg_loss': criterion,
        #     'clf_loss': criterion_1
        # },
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir="./logdir",
        num_epochs=3,
        callbacks={
        #     'seg_loss': dl.CriterionCallback(input_key='targets1', output_key='logits1', criterion_key='seg_loss', prefix='seg_loss'),
        #     'clf_loss': dl.CriterionCallback(input_key='targets2', output_key='logits2', criterion_key='clf_loss', prefix='clf_loss'),
            # 'loss': dl.CriterionCallback(prefix='loss', input_key=None, output_key=None, criterion_key="loss"),
            'optim': dl.OptimizerCallback(metric_key="loss", accumulation_steps=1, grad_clip_params=None)
        #     'acc': dl.AccuracyCallback(input_key='targets2', output_key='logits2', num_classes=num_classes_1),
            # 'iou': dl.IouCallback(per_class=True),
        #     'dice': dl.DiceCallback(per_class=True),
        },
        main_metric='loss',
        verbose=True
    )


def finetuning():
    import os
    from torch import nn, optim
    from torch.utils.data import DataLoader
    from catalyst import dl, utils
    from catalyst.contrib.datasets import MNIST
    from catalyst.data import ToTensor
    from catalyst.experiments import Experiment

    class CustomExperiment(Experiment):
        def __init__(self, logdir, device):
            super().__init__()
            self._logdir = logdir
            self._device = device

        def get_engine(self):
            return dl.DeviceEngine(self._device)

        def get_loggers(self):
            return {
                "console": dl.ConsoleLogger(),
                "csv": dl.CSVLogger(logdir=self._logdir),
                "tensorboard": dl.TensorboardLogger(logdir=self._logdir),
            }

        @property
        def stages(self):
            return ["train_freezed", "train_unfreezed"]

        def get_stage_len(self, stage: str) -> int:
            return 3

        def get_loaders(self, stage: str):
            loaders = {
                "train": DataLoader(
                    MNIST(os.getcwd(), train=True, download=True, transform=ToTensor()), batch_size=32
                ),
                "valid": DataLoader(
                    MNIST(os.getcwd(), train=False, download=True, transform=ToTensor()), batch_size=32
                ),
            }
            return loaders

        def get_model(self, stage: str):
            model = (
                self.model
                if self.model is not None
                else nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
            )
            if stage == "train_freezed":
                # freeze layer
                utils.set_requires_grad(model[1], False)
            else:
                utils.set_requires_grad(model, True)
            return model

        def get_criterion(self, stage: str):
            return nn.CrossEntropyLoss()

        def get_optimizer(self, stage: str, model):
            if stage == "train_freezed":
                return optim.Adam(model.parameters(), lr=1e-3)
            else:
                return optim.SGD(model.parameters(), lr=1e-1)

        def get_scheduler(self, stage: str, optimizer):
            return None

        def get_callbacks(self, stage: str):
            return {
                "criterion": dl.CriterionCallback(
                    metric_key="loss", input_key="logits", target_key="targets"
                ),
                "optimizer": dl.OptimizerCallback(metric_key="loss"),
                "checkpoint": dl.CheckpointCallback(
                    self._logdir, loader_key="valid", metric_key="loss", minimize=True, save_n_best=3
                ),
            }

        def _handle_batch(self, batch):
            x, y = batch
            logits = self.model(x)

            self.batch = {
                "features": x,
                "targets": y,
                "logits": logits,
            }

    runner = CustomRunner("./logs", "cpu")
    runner.run()


def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen

@multigen
def pred_generator():
    for i in range(10):
        yield i
def pred2_generator():
    for i in range(10):
        yield i*2

from typing import Callable
def print_num(generator: Callable):
    for time in range(10):
        print('Time', time)
        for i in generator:
            print(i)    
        print('-'*10)

def rgen (n):
    for elem in list:
        for times in range(n):
            yield elem

from itertools import chain, repeat

def repeated(iterable, n=1):
    items = chain.from_iterable(repeat(item, n) for item in iterable)
    for item in items:
        yield item

    # Or, in Python3.3 you could do:
    # yield from items



if __name__ == '__main__':
    # multilabel_segmentation()
    # print('First')
    # for a, b in zip(pred_generator(), pred2_generator()):
    #     print('a', a)
    #     print('b', b)
    # print('Second')
    # for a, b in zip(pred_generator(), pred2_generator()):
    #     print('a', a)
    #     print('b', b)
    # print_num(pred_generator())
    # l = list(range(10))
    # for i in repeated(l, 10):
    #     print(i)


    # @multigen
    # def myxrange(n):
    #     i = 0
    #     while i < n:
    #         yield i
    #         i += 1

    # m = myxrange(5)
    # print(list(m))
    # print(list(m))
    finetuning()


