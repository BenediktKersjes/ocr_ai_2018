import os
import numpy as np
import argparse
import nltk.corpus
from PIL import Image
from torchvision import transforms
from emnist import EMNIST
from transformations import correct_rotation, random_transform


def parse_arguments():
    parser = argparse.ArgumentParser('Test image creator')
    parser.add_argument('--number-of-words', type=int, default=10, metavar='N',
                        help='number of words created (default: 10)')
    parser.add_argument('--max-word-length', type=int, default=10, metavar='N',
                        help='maximum word length generated (default: 10)')
    parser.add_argument('--output-height', type=int, default=224, metavar='N',
                        help='height of the output image (default: 224)')
    parser.add_argument('--space-mean', type=int, default=30, metavar='N',
                        help='mean space between characters (default: 30)')
    parser.add_argument('--space-std', type=int, default=20, metavar='N',
                        help='std deviation of the mean space between characters (default: 20)')
    parser.add_argument('--white-threshold', type=int, default=30, metavar='N',
                        help='threshold for finding bounding boxes (default: 30)')
    parser.add_argument('--max-train-data', type=int, default=5000, metavar='N',
                        help='maximum number of train data used for creating words (default: 5000)')
    parser.add_argument('--save-path', type=str, default='./../data/generated_words/', metavar='N',
                        help='path where the images are saved (default: ./../data/generated_words/)')
    return parser.parse_args()


class WordSampler (object):
    def __init__(self, max_word_length, output_height, space_mean, space_std, white_threshold, max_train_data):
        self.max_word_length = max_word_length
        self.output_height = output_height
        self.space_mean = space_mean
        self.space_std = space_std
        self.white_threshold = white_threshold

        # Load dataset
        self.train_data = EMNIST('./../dataset', 'letters', download=True, transform=transforms.Compose([
            transforms.Lambda(correct_rotation),
            transforms.Lambda(random_transform),
            transforms.Resize((224, 224)),
            transforms.Grayscale(3),
            transforms.ToTensor()]))

        # Map classes to actual characters
        self.classes = {}
        for char_nr in range(1, 27):
            self.classes[char_nr] = chr(char_nr + 64)

        # Define reversed labels
        self.classes_rev = {}
        for key, value in zip(self.classes.keys(), self.classes.values()):
            self.classes_rev[value] = key

        # Init dictionary with words <= max_word_length
        self.dictionary = [word for word in nltk.corpus.words.words() if len(word) <= max_word_length]

        # Fill data structure with characters from dataset
        self.characters = {}
        for label in self.classes:
            self.characters[label] = []
        for number, (data, target) in enumerate(self.train_data):
            if number > max_train_data:
                break
            self.characters[target].append(data)

    def sample_english_words(self, number_of_words):
        """
        Samples words from nltk database
        :param number_of_words: Number of words to be sampled
        :return: list with sampled words as list of (image, list of (label, bounding_box))
        """
        return [self.sample_english_word() for _ in range(number_of_words)]

    def sample_english_word(self):
        """
        Select random word form dictionary and sample this word
        :return: sampled word as (image, list of (label, bounding_box))
        """
        word_chosen = np.random.choice(self.dictionary)
        return self.sample_chosen_word(word_chosen)

    def sample_chosen_word(self, word_chosen):
        """
        Sample word
        :param word_chosen: string to be sampled
        :return: sampled word as (image, list of (label, bounding_box))
        """
        # Create word image with black background
        word_img = Image.new(mode='L',
                             size=(len(word_chosen) * self.output_height, self.output_height),
                             color=0)

        # Get space on left border of image
        space = self.get_rand_space()
        offset_x_pxl = space

        # Set characters into word image
        labels = []
        for character in word_chosen:
            # Turn character to upper since only upper characters are used in the EMNIST dataset
            character = character.upper()

            # Get class index of character
            class_idx_char = self.classes_rev[character]

            # Sample random character image from characters list
            rand_idx = np.random.randint(len(self.characters[class_idx_char]))
            img = self.characters[class_idx_char][rand_idx]

            # Transform character image to PIL image of correct size
            img = transforms.ToPILImage()(img)
            img = img.resize(size=(self.output_height, self.output_height), resample=Image.BICUBIC)

            # Get border of character in image
            character_box = self.get_min_max_white(img)
            width = character_box[2] - character_box[0]
            height = character_box[3] - character_box[1]

            # Paste character in the middle in y and with random space offset in x
            space_upper = int((self.output_height - height) / 2)
            word_img.paste(img.crop(character_box), (offset_x_pxl, space_upper))

            # Calculate the min and max x and y of character in image picture
            label_box = (offset_x_pxl, character_box[1], character_box[2] + offset_x_pxl, character_box[3])

            # Append label and box of character to labels of word
            labels.append((class_idx_char, label_box))

            # Get random space and update offset in x
            space = self.get_rand_space()
            offset_x_pxl += width + space

        # Resize word image
        word_img = word_img.crop(box=(0, 0, offset_x_pxl, self.output_height))
        return word_img, labels

    def get_rand_space(self):
        """
        Get random pixels as space between characters.
        :return: number of pixels
        """
        return max(0, int(np.random.normal(loc=self.space_mean, scale=self.space_std)))

    def get_min_max_white(self, image):
        """
        Get min and max x and y index of pixel above threshold
        :param image: image to be evaluated
        :return: tuple of (left, upper, right, lower)
        """
        image = np.asarray(image.convert("L"))

        # Get indices of pixels above threshold
        rows, cols = np.nonzero(image > self.white_threshold)

        # Find min and max of x and y
        box = np.min(cols), np.min(rows), np.max(cols), np.max(rows)
        return box

    def save_samples(self, samples, save_path):
        """
        Save sampled images and labels
        :param samples: list of (image, list of (label, bounding_box))
        :param save_path: path where to save the samples
        :return: None
        """
        # Create directory if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save images with labels
        with open(save_path + "labels.txt", "w") as label_file:
            for idx, (word_img, label) in enumerate(samples):
                file_name = '{:05d}.png'.format(idx)

                # Save image
                word_img.save(save_path + file_name, "PNG")

                # Save information about image: file_name; word; bounding_boxes
                label_file.write(file_name + "; ")
                for class_idx, _ in label:
                    label_file.write(self.classes[class_idx])
                label_file.write("; ")
                for _, char_box in label:
                    label_file.write("{}".format(char_box) + ", ")
                label_file.write(os.linesep)


if __name__ == "__main__":
    args = parse_arguments()
    sampler = WordSampler(args.max_word_length, args.output_height, args.space_mean,
                          args.space_std, args.white_threshold, args.max_train_data)
    sampled_words = sampler.sample_english_words(args.number_of_words)
    sampler.save_samples(sampled_words, args.save_path)