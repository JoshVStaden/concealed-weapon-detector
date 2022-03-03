import cv2
import pandas as pd
import numpy as np

import config
from yolo import GUN_DETECTION_MODEL
from util.util import image_object_detection

KEYPOINT_FILE_FORMAT_STRING = "keypoints/{}.txt"
VIDEO_FORMAT_STRING = "../../Datasets/MoviesGuns/{}.mp4"


def read_keypoints_file(vid_number):
    """
    Reads and returns keypoints from keypoints/<vid_number>.txt
    Keypoints are stored as "person_id frame_no x y confidence"

    Returns: Dataframe of all keypoints
    """
    # Read in keypoints as keypoints/<vid_number>.txt
    keypoints_file = open(KEYPOINT_FILE_FORMAT_STRING.format(vid_number))

    # Individual people and frames are separated by blank lines
    full_text = keypoints_file.read()
    individual_people = full_text.split("\n\n")

    keypoints_dict = {'person_id': [], 'frame': [], 'xyc': []}
    for keypoint_set in individual_people:
        # Skip any blank lines
        if keypoint_set == "":
            continue

        # Extract all keypoints
        ind_kp = keypoint_set.split("\n")
        for point in ind_kp:
            # Convert to float, round to nearest decimal, convert to int
            p_id, frame, x, y = map(lambda x: int(round(float(x))),
                                    point.split(" ")[:-1])
            
            # Only convert confidence to float
            conf = float(point.split(" ")[-1])

            # Add to dictionary
            keypoints_dict['person_id'].append(p_id)
            keypoints_dict['frame'].append(frame)
            keypoints_dict['xyc'].append((x, y, conf))

    # Return dataframe
    return pd.DataFrame(keypoints_dict)


def snap_to_frame(coords, frame_shape):
    """
    Snap all coordinates within frame
    """
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
    """
    Searches for the hands, and returns regions around the hands

    Returns: [(xmin, ymin),(xmax, ymax)], a bounding box around the hands
    """
    # Looks for right wrist and right elbow
    # Only returns them if the confidence values are more than 0
    if person_keypoints.xyc.values[config.RIGHT_WRIST][2] > 0 and person_keypoints.xyc.values[config.RIGHT_ELBOW][2] > 0:
        right_wrist = person_keypoints.xyc.values[config.RIGHT_WRIST][:2]
        right_elbow = person_keypoints.xyc.values[config.RIGHT_ELBOW][:2]
    else:
        return []
    
    # Find distance from forearm to wrist
    forearm_length = np.sqrt((right_wrist[0] - right_elbow[0]) ** 2 + (right_wrist[1] - right_elbow[1]) ** 2)

    # If the forearm is too small, don't return anything
    if forearm_length > 20:

        # Snap coordinates within frame
        # Top left of box is defined with forearm length
        top_left = (int(right_wrist[0] - forearm_length), int(right_wrist[1] - forearm_length))
        top_left = snap_to_frame(top_left, frame_shape)

        # Bottom right of box is defined with forearm length divided by 3
        bottom_right = (int(right_wrist[0] + forearm_length), int(right_wrist[1] + forearm_length))
        bottom_right = snap_to_frame(bottom_right, frame_shape)

        return [top_left, bottom_right]
    else:
        return []


def hand_coords_to_image_coords(hand_coords_start, hand_box_size, hand_im_shape=(128, 128)):
    """
    Based on the size of the hand's bounding box, as well as the resized image, rescale
    the hand-coordinates to fit into the image.

    Returns: function(gun_coords) => corrected_coords: a function that converts the coordinates of the gun 
             to image coordinates
    """
    # Extract top left of hand's bounding box
    x_start, y_start = hand_coords_start

    # Rescale according to bounding box size and image size
    x_scale = hand_box_size[1] / hand_im_shape[1]
    y_scale = hand_box_size[0] / hand_im_shape[0]


    def _return_fun(gun_coords):
        ret_arr = []

        # Iterate over gun_coords
        for [(x_gun_min, y_gun_min), (x_gun_max, y_gun_max)] in gun_coords:
            # Rescale coords
            scaled_x_min = int(x_gun_min * x_scale)
            scaled_y_min = int(y_gun_min * y_scale)
            scaled_x_max = int(x_gun_max * x_scale)
            scaled_y_max = int(y_gun_max * y_scale)

            # Add to top left to fit in image
            ret_arr.append([(x_start + scaled_x_min, y_start + scaled_y_min), (x_start + scaled_x_max, y_start + scaled_y_max)])
        return ret_arr
    return _return_fun
        
def annotate_keypoints(frame, person_keypoints):
    """
    Take single person's keypoints and annotate image

    Returns: image annotation
    """
    for coords in person_keypoints.xyc:
        x, y, conf = coords
        frame = cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

    return frame



def annotate_image(frame, keypoints, annotate_kp=False, annotate_hand_bbox=False, annotate_gun=True, show_hand_image=True):
    """
    Takes in keypoints and an image, and annotates it.
    Current annotations: keypoints, right hand bounding box, gun found within right hand
    Keypoints must only contain those within the current frame

    Returns: the annotated image
    """
    # Get all people within current frame
    people = np.unique(keypoints.person_id)

    # Iterate over people
    for person in people:
        # Get current person's keypoints
        person_keypoints = keypoints[keypoints.person_id == person]

        # Find their right hand
        hand_regions = extract_hand_regions(person_keypoints)

        # If hand not visible, skip
        if len(hand_regions) > 0:
            # Extract the portion of the image containing the right hand
            top_left, bottom_right = hand_regions
            hand_image = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Draw rectangle around right hand
            if annotate_hand_bbox:
                frame = cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

            # Find the size of the bbox around the hand
            hand_size = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
            print(hand_size)

            # Resize hand image to (128, 128)
            hand_image = cv2.resize(hand_image, (64, 64))

            # Get a converter to global coordinates
            gun_to_image = hand_coords_to_image_coords(top_left, hand_size, hand_im_shape=(64, 64))
            hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)

            # Detect any guns within the image of the hand
            gun_coord, hand_image = image_object_detection(GUN_DETECTION_MODEL, hand_image)

            # If no guns, skip
            if annotate_gun and len(gun_coord) > 0:
                # Get coordinates, convert to global coords, draw rectange around gun
                min_gun, max_gun = gun_to_image(gun_coord)[0]
                frame = cv2.rectangle(frame, min_gun, max_gun, (255,255,255))
            
            # Show a separate image with the hand
            if show_hand_image:
                cv2.imshow("Hand", hand_image)
        
        # Annotate keypoints
        if annotate_kp:
            frame = annotate_keypoints(frame, person_keypoints)
            
    return frame


def process_single_video(vid_number):
    """
    Process a single video
    """
    # Open video file
    cap = cv2.VideoCapture(VIDEO_FORMAT_STRING.format(vid_number))

    keypoints = read_keypoints_file(vid_number)
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
