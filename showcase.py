import numpy as np
import cv2
import threading
import torch
from alexnet import AlexNet
import nltk
from procedure import *


class LiveShowcase:
    def __init__(self, path_to_model):
        num_classes = 27

        # Member variables
        self.status = 'Ready'
        self.last_words = None
        self.dictionary_set = set(nltk.corpus.words.words())

        # Load pre learned AlexNet
        state_dict = torch.load(path_to_model, map_location=lambda storage, loc: storage)['model']
        self.model = AlexNet(num_classes)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def process_image(self, image, bounding_boxes):
        """
        Process image to find and classify characters and build 5 most probable words
        :param image: rgb image
        :param bounding_boxes: list of bounding boxes containing characters (min_x, min_y, width, height)
        :return: None
        """
        self.status = 'Processing'

        # Find 5 most probable words
        subimages = extract_characters(image, bounding_boxes)
        words = classify_characters(self.model, subimages)
        self.last_words = words[:5]

        self.status = 'Ready'

    def start(self, max_bounding_boxes=10):
        """
        Start the live showcase using a camera
        :return: None
        """
        # Try to open a connection to the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Error: No camera found')
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        print('Press q to stop the live showcase')
        while True:
            # Capture frame-by-frame
            ret, image = cap.read()
            output = image

            # Find bounding boxes for each character
            image = preprocess_image(image)
            bounding_boxes = find_bounding_boxes(image)
            bounding_boxes = filter_bounding_boxes(image, bounding_boxes)
            for box in bounding_boxes:
                cv2.rectangle(output, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 0, 255), 2)

            # Process image if no other image is processed
            if self.status.__contains__('Ready'):
                if len(bounding_boxes) > max_bounding_boxes:
                    self.status = 'Ready [Warning: too many bounding boxes]'
                    self.last_words = None
                else:
                    thread = threading.Thread(target=self.process_image, args=(image, bounding_boxes), daemon=True)
                    thread.start()

            # Draw status bar with last recognized words
            cv2.putText(output, 'Status: {}'.format(self.status), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 1, cv2.LINE_AA)
            if self.last_words:
                for offset, word in zip(range(len(self.last_words)), self.last_words):
                    color = (0, 0, 0)
                    # Use green color if word is in dictionary
                    if word[0].lower() in self.dictionary_set:
                        color = (0, 255, 0)
                    cv2.putText(output, '{} ({:5.2f}%)'.format(word[0], 100 * word[1]),
                                (10, 20 + (offset + 1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color, 1, cv2.LINE_AA)

            # Draw bounding box around detected word
            if self.last_words and len(bounding_boxes) > 1:
                word = self.last_words[0]
                color = (0, 0, 255)
                # Use green color if word is in dictionary
                if word[0].lower() in self.dictionary_set:
                    color = (0, 255, 0)
                text = '{} ({:5.2f}%)'.format(word[0], 100 * word[1])
                padding = 10
                top_left = (np.min([b[0] for b in bounding_boxes]) - padding,
                            np.min([b[1] for b in bounding_boxes]) - padding)
                bottom_right = (np.max([b[0]+b[2] for b in bounding_boxes]) + padding,
                                np.max([b[1]+b[3] for b in bounding_boxes]) + padding)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(output, (top_left[0] - 1, top_left[1] - text_size[1] - 2 * padding),
                                          (top_left[0] + text_size[0] + 2 * padding, top_left[1]),
                              color, thickness=cv2.FILLED)
                cv2.putText(output, text, (top_left[0] + padding, top_left[1] - padding),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.rectangle(output, top_left, bottom_right, color, 2)

            # Display the resulting frame
            cv2.imshow('Image internal', image)
            cv2.imshow('Showcase', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

