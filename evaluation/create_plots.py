import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib import rcParams
import re


if __name__ == "__main__":
    font = {'family': 'sans-serif', 'weight': 'normal', 'size': 20}
    matplotlib.rc('font', **font)
    rcParams.update({'figure.autolayout': True})
    marker_square = 's'
    marker_circle = '*'
    color = '#B5152B'

    distances = defaultdict(lambda: defaultdict(lambda: 0))
    with open('../data/generated_words/test_results_distance.txt') as f:
        for line in f:
            parts = line.split(';')
            distances[parts[0]][parts[1]] = parts[2]

    size_distances = defaultdict(lambda: defaultdict(lambda: 0))
    with open('../data/generated_words/test_results_size.txt') as f:
        for line in f:
            parts = line.split(';')
            size_distances[parts[0]][parts[1]] = parts[2]

    corrected_words = defaultdict(lambda: defaultdict(lambda: 0))
    with open('../data/generated_words/test_results_corrected.txt') as f:
        for line in f:
            parts = line.split(';')
            corrected_words[parts[0]][parts[1]] = parts[2]

    # Average edit distance by length
    avg_distances = []
    for size in distances:
        distance_sum = 0
        count = 0
        for distance in distances[size]:
            count += int(distances[size][distance])
            distance_sum += int(distances[size][distance]) * int(distance)
        avg_distances.append([int(size), float(distance_sum / count)])

    plt.figure(figsize=(10, 5))
    plt.xlabel('Word length in characters')
    plt.ylabel('Average Levenshtein distance')
    plt.plot(np.array(avg_distances)[:, 0], np.array(avg_distances)[:, 1], linestyle=':', marker=marker_square,
             color=color)
    plt.grid(True)
    plt.show()

    # edit distance <= 1
    distance_smaller_one = []
    overall_count = 0
    overall_total = 0
    for size in distances:
        count = 0
        total = 0
        for distance in distances[size]:
            if int(distance) <= 1:
                count += int(distances[size][distance])
                overall_count += int(distances[size][distance])
            total += int(distances[size][distance])
            overall_total += int(distances[size][distance])
        distance_smaller_one.append([int(size), float(count / total) * 100])

    print('Overall distance <= 1: ' + str(float(overall_count / overall_total)))

    # edit distance == 0
    distance_zero = []
    overall_count = 0
    overall_total = 0
    for size in distances:
        count = 0
        total = 0
        for distance in distances[size]:
            if int(distance) == 0:
                count += int(distances[size][distance])
                overall_count += int(distances[size][distance])
            total += int(distances[size][distance])
            overall_total += int(distances[size][distance])
        distance_zero.append([int(size), float(count / total) * 100])

    print('Overall distance = 0: ' + str(float(overall_count / overall_total)))

    plt.figure(figsize=(10, 5))
    plt.xlabel('Word length in characters')
    plt.ylabel('Predicted words [%]')
    handle_1, = plt.plot(np.array(distance_zero)[:, 0], np.array(distance_zero)[:, 1],
                         label='Levenshtein Distance = 0', linestyle=':', marker=marker_circle, color=color)
    handle_2, = plt.plot(np.array(distance_smaller_one)[:, 0], np.array(distance_smaller_one)[:, 1],
                         label='Levenshtein Distance ≤ 1', linestyle=':', marker=marker_square, color=color)
    plt.legend(handles=[handle_1, handle_2], loc=3)
    plt.grid(True)
    plt.show()

    # correct after correction
    corrected_words_ratio = []
    overall_count = 0
    overall_total = 0
    for size in corrected_words:
        overall_total += int(corrected_words[size]['0'])
        overall_count += int(corrected_words[size]['1'])
        corrected_words_ratio.append(
            [int(size), float(int(corrected_words[size]['1']) / int(corrected_words[size]['0'])) * 100])

    print('Correctly predicted: ' + str(float(overall_count / overall_total)))

    plt.figure(figsize=(10, 5))
    plt.xlabel('Word length in characters')
    plt.ylabel('Correctly predicted words [%]')
    plt.plot(np.array(corrected_words_ratio)[:, 0], np.array(corrected_words_ratio)[:, 1],
             linestyle=':', marker=marker_square, color=color)
    plt.grid(True)
    plt.show()

    # Read data from result files
    train_pattern = re.compile('Train Epoch: (\d+) \tCorrect: ([\d.]+)%\tAverage loss: ([\d.]+)\n')
    with open('train_result.txt', 'r') as f:
        content = f.read()
        elements = train_pattern.findall(content)
        epoch = [int(e[0]) for e in elements]
        train_correct = [float(e[1]) for e in elements]
        train_loss = [float(e[2]) for e in elements]

    test_pattern = re.compile('Test Epoch: (\d+) \tCorrect: ([\d.]+)%\tAverage loss: ([\d.]+)\n')
    with open('test_result.txt', 'r') as f:
        content = f.read()
        elements = test_pattern.findall(content)
        test_correct = [float(e[1]) for e in elements]
        test_loss = [float(e[2]) for e in elements]

    # plot data
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Correct labels [%]')
    ax1.plot(epoch, test_correct, 'o:', label='Testing', marker=marker_square, color=color)
    ax1.plot(epoch, train_correct, 'o:', label='Training', marker=marker_circle, color=color)

    plt.grid(True)
    ax1.set_yticks(range(70, 105, 5))
    plt.legend(loc=4)
    plt.show()
