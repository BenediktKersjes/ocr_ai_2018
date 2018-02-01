from emnist import EMNIST
from alexnet import AlexNet
from transformations import correct_rotation
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable


class TestNetwork():
    def __init__(self, dataset, batch_size, epochs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs

        # letters contains 27 classes, digits contains 10 classes
        num_classes = 27 if dataset == 'letters' else 10

        # Load mdoel and use cuda if available
        self.model = AlexNet(num_classes)
        if torch.cuda.is_available():
            self.model.cuda()

        # Load testing dataset
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.test_loader = torch.utils.data.DataLoader(
            EMNIST('./data', dataset, download=True, transform=transforms.Compose([
                transforms.Lambda(correct_rotation),
                transforms.Resize((224, 224)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
            ]), train=False),
            batch_size=batch_size, shuffle=True, **kwargs
        )

        # Optimizer and loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def test(self, epoch):
        """
        Test the model for one epoch with a pre trained network
        :param epoch: Current epoch
        :return: None
        """
        # Load weights from trained model
        state_dict = torch.load('./trained_models/{}_{}.pth'.format(self.dataset, epoch),
                                map_location=lambda storage, loc: storage)['model']
        self.model.load_state_dict(state_dict)
        self.model.eval()

        test_loss = 0
        test_correct = 0
        progress = None
        for batch_idx, (data, target) in enumerate(self.test_loader):
            # Get data and label
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            #
            output = self.model(data)
            loss = self.loss_fn(output, target)
            test_loss += loss.data[0]
            pred = output.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.data.view_as(pred)).sum()

            # Print information about current step
            current_progress = int(100 * (batch_idx + 1) * self.batch_size / len(self.test_loader.dataset))
            if current_progress is not progress and current_progress % 5 == 0:
                progress = current_progress
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * len(data), len(self.test_loader.dataset),
                    current_progress, loss.data[0]))

        test_loss /= (len(self.test_loader.dataset) / self.batch_size)
        test_correct /= len(self.test_loader.dataset)
        test_correct *= 100

        # Print information about current epoch
        print('Test Epoch: {} \tCorrect: {:3.2f}%\tAverage loss: {:.6f}'
              .format(epoch, test_correct, test_loss))

    def start(self):
        """
        Start testing the network
        :return: None
        """
        for epoch in range(1, self.epochs + 1):
            self.test(epoch)