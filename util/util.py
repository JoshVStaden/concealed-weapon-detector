import cv2

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
    