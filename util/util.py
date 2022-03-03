from unittest import result
import cv2
import numpy as np

def image_object_detection(model, frame, annotate=True):
    # brightnesses = np.sum(frame, axis=2) / (255)
    # frame[np.where(brightnesses < 0.1),:] = 0
    # frame = frame * 255
    # brightnesses = np.sum(frame, axis=0)
    # print(np.min(frame), np.max(frame))
    # quit()
    # frame[np.where(brightnesses < 0.2), :] = 0
    results = model([frame]).pandas().xyxy[0]

    coords = []
    conf_scores = [res.confidence for _, res in results.iterrows()]
    if len(conf_scores) == 0:
        return (coords, frame)
    max_conf = max(conf_scores)
    if max_conf > 0.0:
        for idx, res in results.iterrows():
            if res.confidence == max_conf:
                xmin = int(res.xmin)
                ymin = int(res.ymin)
                xmax = int(res.xmax)
                ymax = int(res.ymax)
                coords.append([(xmin, ymin), (xmax, ymax)])
                if annotate:
                    frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,255,255))
                    frame = cv2.putText(frame, f"{res['name']} {round(res.confidence, 2)}", (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))
                break
    return (coords, frame)





def show_video(model, filename=0):
    """
    Show a model making predictions on a video file.

    @param model: The object detector
    @param filename: The video file. Defaults to webcam
    """

    # define a video capture object
    vid = cv2.VideoCapture(filename)
    
    while(True):
        
        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        results = model([frame]).pandas().xyxy[0]

        for idx, res in results.iterrows():
            if res.confidence > 0.5:
                xmin = int(res.xmin)
                ymin = int(res.ymin)
                xmax = int(res.xmax)
                ymax = int(res.ymax)
                frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,255,255))
                frame = cv2.putText(frame, res['name'], (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))




    
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    