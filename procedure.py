from operator import itemgetter
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
from torch import nn
import numpy as np
import cv2


def preprocess_image(image):
    """
    Pre process the image from rgb to binary
    :param image: rgb image
    :return: binary image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    ret, image = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)
    return image


def merge_bounding_boxes(box1, box2):
    """
    Merge two bounding boxes to recieve one which includes both other boxes
    :param box1: (min_x, min_y, width, height)
    :param box2: (min_x, min_y, width, height)
    :return: (min_x, min_y, width, height)
    """
    max_x = box1[0] + box1[2] if box1[0] + box1[2] > box2[0] + box2[2] else box2[0] + box2[2]
    max_y = box1[1] + box1[3] if box1[1] + box1[3] > box2[1] + box2[3] else box2[1] + box2[3]
    min_x = box2[0] if box2[0] < box1[0] else box1[0]
    min_y = box2[1] if box2[1] < box1[1] else box1[1]
    width = max_x - min_x
    height = max_y - min_y
    return min_x, min_y, width, height


def find_bounding_boxes(image, factor_horizontal=0.3, factor_vertical=1.0):
    """
    Find the bounding boxes for all characters
    :param image: binary image
    :param factor_horizontal: factor for combining boxes in horizontal direction
    :param factor_vertical: factor for combining boxes in vertical direction
    :return: list of bounding boxes containing the characters (min_x, min_y, width, height)
    """
    # Find connected components in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    stats = sorted(stats, key=itemgetter(0))

    # Merge boxes that probably belong to the same character
    bounding_boxes = []
    current_box = None
    for i in range(1, len(stats)):
        box = stats[i]
        if current_box is None:
            # Set first box to compare with
            current_box = box
        else:
            # Calculate the overlap in x direction
            # relative to the min width of both compared boxes
            overlap = current_box[0] + current_box[2] - box[0]
            min_width = box[2] if box[2] < current_box[2] else current_box[2]
            horizontal_overlap_factor = overlap / min_width

            # Calculate the distance between centers of characters in y direction
            # relative to max height of both compared boxes
            height_box = max(current_box[3], box[3])
            vertical_difference = ((current_box[1] + current_box[3] / 2)
                                   - (box[1] + box[3] / 2))
            vertical_space_quotient = abs(vertical_difference / height_box)

            # Check if boxes belong together
            if (horizontal_overlap_factor > factor_horizontal) and (vertical_space_quotient < factor_vertical):
                current_box = merge_bounding_boxes(current_box, box)
            else:
                # Add box to list and update current_box
                bounding_boxes.append(current_box)
                current_box = box
    if current_box is not None:
        bounding_boxes.append(current_box)

    return bounding_boxes


def filter_bounding_boxes(image, bounding_boxes, factor=0.05):
    """
    Sort out bounding boxes that are to small and so probably not a character
    :param image: binary image
    :param bounding_boxes: list of bounding boxes containing (min_x, min_y, width, height)
    :param factor: minimum height of the bounding box in relation to the full image
    :return: filtered list of bounding boxes containing (min_x, min_y, width, height)
    """
    return list(filter(lambda box: box[3] > image.shape[0] * factor, bounding_boxes))


def extract_characters(image, bounding_boxes):
    """
    Extract the characters from the image
    :param image: binary image
    :param bounding_boxes: bounding boxes containing the characters
    :return: list of images containing the characters
    """
    subimages = []
    for box in bounding_boxes:
        # Create subimage using the bounding box
        pil = Image.fromarray(image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])
        size = pil.size

        # Add padding to make image quadratic
        if size[0] > size[1]:
            pil = transforms.Pad((0, int((size[0] - size[1]) / 2)))(pil)
        else:
            pil = transforms.Pad((int((size[1] - size[0]) / 2), 0))(pil)

        # Resize image to match it with training set
        pil = transforms.Resize((28, 28))(pil)

        # Rescale padding
        padding = int(2. * pil.size[0] / 28.)
        pil = transforms.Pad((padding, padding))(pil)

        # Add subimage to list
        subimages.append(pil)

    return subimages


def classify_characters(model, subimages):
    """
    Classify the characters and return the probability
    :param model: model used for classification
    :param subimages: images containing the characters
    :return: all possible words and their probability
    """
    lut = [' ',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z'
           ]

    # Extract characters
    word_1 = []  # Contains all characters with the highest probability
    word_2 = []  # Contains all characters with the second highest probability
    for subimage in subimages:
        tensor = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.Grayscale(3), transforms.ToTensor()])(subimage)
        tensor.unsqueeze_(0)
        tensor = Variable(tensor)
        output = model(tensor)
        output = nn.Softmax(dim=1)(output)
        output = output.data[0].numpy()

        results = list(zip(lut, output))
        results = sorted(results, key=lambda x: x[1], reverse=True)
        word_1.append(results[0])
        word_2.append(results[1])

    # Create all possible words using the two most probable characters for each bounding box
    n = len(word_1)
    words = []
    combinations = [[int(x) for x in list("{0:0b}".format(i).zfill(n))] for i in range(0, 2 ** n)]
    for combination in combinations:
        word = [row[:] for row in word_1]
        for index, should_replace in zip(range(len(combination)), combination):
            if should_replace == 1:
                word[index] = word_2[index]
        words.append(word)

    words = list(map(lambda word: (''.join([w[0] for w in word]), np.prod([w[1] for w in word])), words))
    words = sorted(words, key=itemgetter(1), reverse=True)
    return words
