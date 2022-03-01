import cv2
import pandas as pd
import numpy as np

import config
from yolo import GUN_DETECTION_MODEL
from util.util import image_object_detection

# person_id frame_number x y score

KEYPOINT_FILE_FORMAT_STRING = "keypoints/{}.txt"
VIDEO_FORMAT_STRING = "../../Datasets/MoviesGuns/{}.mp4"


def read_keypoints_file(vid_number):
    keypoints_file = open(KEYPOINT_FILE_FORMAT_STRING.format(vid_number))

    full_text = keypoints_file.read()

    individual_people = full_text.split("\n\n")

    keypoints_dict = {'person_id': [], 'frame': [], 'xyc': []}

    for keypoint_set in individual_people:
        if keypoint_set == "":
            continue
        ind_kp = keypoint_set.split("\n")
        for point in ind_kp:
            p_id, frame, x, y = map(lambda x: int(round(float(x))),
                                    point.split(" ")[:-1])
            conf = float(point.split(" ")[-1])
            keypoints_dict['person_id'].append(p_id)
            keypoints_dict['frame'].append(frame)
            keypoints_dict['xyc'].append((x, y, conf))
    return pd.DataFrame(keypoints_dict)


def snap_to_frame(coords, frame_shape):
    if coords[0] < 0:
        coords = (0, coords[1])

    if coords[1] < 0:
        coords = (coords[0], 0)

    if coords[0] > frame_shape[0]:
        coords = (frame_shape[0], coords[1])

    if coords[1] > frame_shape[1]:
        coords = (coords[0], frame_shape[1])

    return coords


def extract_hand_regions(person_keypoints, frame_shape=(640, 480)):
    
    if person_keypoints.xyc.values[config.RIGHT_WRIST][2] > 0 and person_keypoints.xyc.values[config.RIGHT_ELBOW][2] > 0:
        right_wrist = person_keypoints.xyc.values[config.RIGHT_WRIST][:2]
        right_elbow = person_keypoints.xyc.values[config.RIGHT_ELBOW][:2]
    else:
        return []
    
    forearm_length = np.sqrt((right_wrist[0] - right_elbow[0]) ** 2 + (right_wrist[1] - right_elbow[1]) ** 2)

    if forearm_length > 20:

        top_left = (int(right_wrist[0] - forearm_length), int(right_wrist[1] - forearm_length))
        top_left = snap_to_frame(top_left, frame_shape)

        # bottom_right = (int(right_wrist[0] + forearm_length), int(right_wrist[1] + forearm_length))
        bottom_right = (int(right_wrist[0] + forearm_length // 3), int(right_wrist[1] + forearm_length // 3))
        bottom_right = snap_to_frame(bottom_right, frame_shape)

        return [top_left, bottom_right]
    else:
        return []


def annotate_image(frame, keypoints):
    people = np.unique(keypoints.person_id)
    curr_kp = 0
    for person in people:
        person_keypoints = keypoints[keypoints.person_id == person]
        hand_regions = extract_hand_regions(person_keypoints)
        # print(top_left, bottom_right)
        if len(hand_regions) > 0:
            top_left, bottom_right = hand_regions
            hand_image = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            frame = cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

            hand_image = cv2.resize(hand_image, (640, 480))
            hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
            _, hand_image = image_object_detection(GUN_DETECTION_MODEL, hand_image)
            cv2.imshow("Hand", hand_image)
        # print(len(person_keypoints.xyc))
        for coords in person_keypoints.xyc:
            x, y, conf = coords
            if curr_kp == 100:
                frame = cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
            curr_kp += 1
    return frame


def process_single_video(vid_number):
    """
    Process a single video
    """
    # Open video file
    cap = cv2.VideoCapture(VIDEO_FORMAT_STRING.format(vid_number))

    keypoints = read_keypoints_file(vid_number)
    # print(keypoints)
    # quit()

    frame_no = 0

    # Iterate over frames in video
    while cap.isOpened():
        frame_no += 1
        # Read frame and convert to RGB
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        curr_frame_kp = keypoints[keypoints.frame == frame_no]

        frame = annotate_image(frame, curr_frame_kp)

        # Convert back to BGR and show
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("CV2", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close video
    cap.release()


if __name__ == '__main__':
    process_single_video(1)
