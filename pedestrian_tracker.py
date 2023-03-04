# structure:
# |____pedestrian_tracker.py (simple pedestrian detector with bird's eye transform)
# |____Dataset
#     |____homography.txt (.txt file with homography matrix / same as 002.txt in original dataset)
#     |____video.avi (.avi video / same as 002.avi in original dataset)

import cv2 as cv
import numpy as np
from math import sqrt


class Configuration:
    def __init__(self):
        self.n = 25  # number of frames to track people
        self.dist_threshold = 15  # clusterization threshold in px
        self.scale_factor = 10  # bird's eye view image zoom
        self.horizontal_shift = 300  # bird's eye view horizontal shift
        self.vertical_shift = 100  # bird's eye view vertical shift


def homography_reader():
    homography_matrix = np.loadtxt('Dataset/homography.txt', delimiter=',').reshape((3, 3))  # read homography
    homography_matrix[0, ...] *= Configuration().scale_factor  # zoom output image
    homography_matrix[1, ...] *= Configuration().scale_factor

    for col in range(3):  # perform horizontal / vertical shift for best visualization
        homography_matrix[0, col] += Configuration().horizontal_shift * homography_matrix[2, col]
        homography_matrix[1, col] += Configuration().vertical_shift * homography_matrix[2, col]

    return homography_matrix


def calculate_distance(tracked, detected):  # calculate distance between tracked people and just detected
    return sqrt((tracked[0] - detected[0])**2 + (tracked[1] - detected[1])**2)


def adapt_bounding_box(bounding_box):  # detector's bounding box 15% reduction
    (x_coord, y_coord, width, height) = bounding_box
    x_coord += 0.075 * width
    y_coord += 0.025 * height
    width *= 0.85
    height *= 0.85
    return int(x_coord), int(y_coord), int(width), int(height)


# initialize default OpenCV people detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cap = cv.VideoCapture('Dataset/video.avi')  # load video from file
tracks = [[]]
while cap.isOpened():
    _, frame = cap.read()

    (regions, _) = hog.detectMultiScale(frame, padding=(20, 20), winStride=(8, 8), scale=1.05, groupThreshold=7)

    bounding_box_center = None
    is_detected = regions != ()  # True if detect at least one person
    saved_track = []
    if is_detected:
        for b_box in regions:
            x, y, w, h = adapt_bounding_box(b_box)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)  # draw rectangles
            bounding_box_center = [int(x + w / 2), int(y + h)]  # middle point of bounding box's bottom
            match = False
            if tracks == [[]]:
                tracks[0] = [bounding_box_center]
            else:
                for track in tracks:
                    current_track = track[0]
                    if calculate_distance(current_track, bounding_box_center) < Configuration().dist_threshold:
                        tracks[tracks.index(track)].insert(0, bounding_box_center)
                        saved_track.append(tracks.index(track))  # store all objects with updated coordinates
                        match = True
                if not match:
                    tracks.append([bounding_box_center])
                    saved_track.append(len(tracks) - 1)  # store just added objects
    else:
        tracks = [[]]

    _, image = cap.read()

    for track in tracks:
        if tracks.index(track) not in saved_track:
            tracks.pop(tracks.index(track))  # delete all track if they are not stored
        if len(track) > Configuration().n:
            tracks[tracks.index(track)].pop(-1)  # store tracks for only last n-frames
        if len(track) > 2:
            current_track = np.array(track, dtype=np.int32)
            pts = np.reshape(current_track, (-1, 1, 2))
            cv.polylines(frame, [pts], isClosed=False, color=(0, 0, 255), thickness=5)  # draw tracks on original image
            cv.polylines(image, [pts], isClosed=False, color=(0, 0, 255), thickness=7)  # ... bird's eye view image
    cv.imshow('original image', frame)

    # perform bird's eye view transform
    image = cv.warpPerspective(image, homography_reader(), (frame.shape[1], frame.shape[0]))
    cv.imshow("bird's eye view transformation", image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
