from emnist import EMNIST
from alexnet import AlexNet
from transformations import correct_rotation, random_transform
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable


class TrainNetwork():
    def __init__(self, dataset, batch_size, epochs, lr, lr_decay_epoch, momentum):
        assert (dataset == 'letters' or dataset == 'mnist')

        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_decay_epoch = lr_decay_epoch
        self.momentum = momentum

        # letters contains 27 classes, digits contains 10 classes
        num_classes = 27 if dataset == 'letters' else 10

        # Load pre learned AlexNet with changed number of output classes
        state_dict = torch.load('./trained_models/alexnet.pth')
        state_dict['classifier.6.weight'] = torch.zeros(num_classes, 4096)
        state_dict['classifier.6.bias'] = torch.zeros(num_classes)
        self.model = AlexNet(num_classes)
        self.model.load_state_dict(state_dict)

        # Use cuda if available
        if torch.cuda.is_available():
            self.model.cuda()

        # Load training dataset
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.train_loader = torch.utils.data.DataLoader(
            EMNIST('./data', dataset, download=True, transform=transforms.Compose([
                transforms.Lambda(correct_rotation),
                transforms.Lambda(random_transform),
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224, (0.9, 1.1), ratio=(0.9, 1.1)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
            ])),
            batch_size=batch_size, shuffle=True, **kwargs
        )

        # Optimizer and loss function
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        self.loss_fn = nn.CrossEntropyLoss()

    def reduce_learning_rate(self, epoch):
        """
        Reduce the learning rate by factor 0.1 every lr_decay_epoch
        :param optimizer: Optimizer containing the learning rate
        :param epoch: Current epoch
        :param init_lr: Initial learning rate
        :param lr_decay_epoch: Number of epochs until learning rate gets reduced
        :return: None
        """
        lr = self.lr * (0.1**(epoch // self.lr_decay_epoch))

        if epoch % self.lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, epoch):
        """
        Train the model for one epoch and save the result as a .pth file
        :param epoch: Current epoch
        :return: None
        """
        self.model.train()

        train_loss = 0
        train_correct = 0
        progress = None
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Get data and label
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            # Optimize using backpropagation
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            train_loss += loss.data[0]
            pred = output.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(target.data.view_as(pred)).sum()
            loss.backward()
            self.optimizer.step()

            # Print information about current step
            current_progress = int(100 * (batch_idx + 1) * self.batch_size / len(self.train_loader.dataset))
            if current_progress is not progress and current_progress % 5 == 0:
                progress = current_progress
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data), len(self.train_loader.dataset),
                    current_progress, loss.data[0]))

        train_loss /= (len(self.train_loader.dataset) / self.batch_size)
        train_correct /= len(self.train_loader.dataset)
        train_correct *= 100

        # Print information about current epoch
        print('Train Epoch: {} \tCorrect: {:3.2f}%\tAverage loss: {:.6f}'
              .format(epoch, train_correct, train_loss))

        # Save snapshot
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, './trained_models/{}_{}.pth'.format(self.dataset, epoch))

    def start(self):
        """
        Start training the network
        :return: None
        """
        for epoch in range(1, self.epochs + 1):
            self.reduce_learning_rate(epoch)
            self.train(epoch)