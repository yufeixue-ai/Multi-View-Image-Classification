import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Round:
    def __call__ (self , pic): 
        return torch.round(pic)
binariser = Round()

class Dataloader:
    def __init__(self, config):
        self.batch_size = config.dataset.batch_size
        self.shuffle = config.dataset.shuffle
        self.download = config.dataset.download
        self.data_dir = f'{config.base.root_path}/{config.dataset.dir}'

        # dataset pre-processing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            binariser
        ])

    def load_data(self):
        train_dataset = datasets.MNIST(
            root=self.data_dir, train=True, transform=self.transform, download=self.download
        )
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )

        test_dataset = datasets.MNIST(
            root=self.data_dir, train=False, transform=self.transform, download=self.download
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=self.batch_size, shuffle=False
        )

        return train_loader, test_loader
