import argparse
from train_network import TrainNetwork
from test_network import TestNetwork
from showcase import LiveShowcase


def parse_arguments():
    parser = argparse.ArgumentParser('Artificial Intelligence Project: Optical character recognition')
    parser.add_argument('--showcase', action='store_const', const=True,
                        help='start the live showcase using a camera')
    parser.add_argument('--train', choices=['mnist', 'letters'],
                        help='train the network using the specified dataset')
    parser.add_argument('--test', choices=['mnist', 'letters'],
                        help='test the network using the specified dataset')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr-decay-epoch', type=int, default=25, metavar='D',
                        help='reduce the learning rate by factor 0.1 every D epochs (default: 25)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--model', type=str, default='./trained_models/letters_30.pth',
                        help='Path to the trained model used for the live showcase')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    if args.showcase:
        print('Starting the showcase')
        showcase = LiveShowcase(args.model)
        showcase.start()

    if args.train:
        print('Training the network using {} dataset'.format(args.train))
        training = TrainNetwork(args.train, args.batch_size, args.epochs, args.lr, args.lr_decay_epoch, args.momentum)
        training.start()

    if args.test:
        print('Testing the network using {} dataset'.format(args.test))
        training = TestNetwork(args.test, args.batch_size, args.epochs)
        training.start()