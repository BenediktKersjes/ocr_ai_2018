from collections import defaultdict
import Levenshtein
import argparse
import torch
from alexnet import AlexNet
import nltk
from procedure import *


def parse_arguments():
    parser = argparse.ArgumentParser('Test sample images')
    parser.add_argument('--path-to-model', type=str, default='./../trained_models/letters_30.pth', metavar='N',
                        help='input batch size for training (default: ./../trained_models/letters_30.pth)')
    parser.add_argument('--path-to-images', type=str, default='./../data/generated_words/', metavar='N',
                        help='input batch size for training (default: ./../data/generated_words/)')
    parser.add_argument('--save-path', type=str, default='./../data/generated_words/', metavar='N',
                        help='path where the results are saved (default: ./../data/generated_words/)')
    return parser.parse_args()


def test_sample_images(path_to_model, path_to_images, save_path):
    num_classes = 27

    # Load pre learned AlexNet
    state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)['model']
    model = AlexNet(num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    # Process every image
    dictionary = set(nltk.corpus.words.words())
    distances = defaultdict(lambda: defaultdict(lambda: 0))
    size_distances = defaultdict(lambda: defaultdict(lambda: 0))
    corrected_words = defaultdict(lambda: defaultdict(lambda: 0))
    with open('{}labels.txt'.format(path_to_images)) as f:
        for line in f:
            sections = line.split('; ')
            if len(sections) < 2:
                continue
            fname = sections[0]
            correct_word = sections[1]

            # Open image
            image = cv2.imread('{}{}'.format(path_to_images, fname))
            output = image

            # Find bounding boxes for each character
            image = preprocess_image(image)
            _, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)
            bounding_boxes = find_bounding_boxes(image)
            bounding_boxes = filter_bounding_boxes(image, bounding_boxes)

            # Find 5 most probable results
            subimages = extract_characters(image, bounding_boxes)
            results = classify_characters(model, subimages)
            results = results[:5]

            # Check if word can be corrected
            corrected_word = ''
            for word in results:
                if word[0].lower() in dictionary and corrected_word is '':
                    corrected_word = word[0]

            # Append to evaluation dicts for evaluation
            most_probable_word = results[0][0]
            distance = Levenshtein.distance(most_probable_word, correct_word)
            distances[len(correct_word)][distance] += 1
            size_distances[len(correct_word)][len(most_probable_word)] += 1

            corrected_words[len(correct_word)][0] += 1
            if corrected_word == correct_word:
                corrected_words[len(correct_word)][1] += 1

            # Print information about current progress
            print('Correct: {:12s}  Most probable: {:12s}  Corrected: {:12s}  Distance: {:1d}  Success: {}'.format(
                correct_word, most_probable_word, corrected_word, distance, corrected_word == correct_word
            ))

    #  Save results
    with open('{}/test_results_distance.txt'.format(save_path), 'w') as f:
        for size in sorted(distances):
            for distance in sorted(distances[size]):
                f.write('{};{};{}\n'.format(size, distance, distances[size][distance]))

    with open('{}/test_results_size.txt'.format(save_path), 'w') as f:
        for size in sorted(size_distances):
            for size_distance in sorted(size_distances[size]):
                f.write('{};{};{}\n'.format(size, size_distance, size_distances[size][size_distance]))

    with open('{}/test_results_corrected.txt'.format(save_path), 'w') as f:
        for key in sorted(corrected_words):
            for count in sorted(corrected_words[key]):
                f.write('{};{};{}\n'.format(key, count, corrected_words[key][count]))


if __name__ == "__main__":
    args = parse_arguments()
    test_sample_images(args.path_to_model, args.path_to_images, args.save_path)
