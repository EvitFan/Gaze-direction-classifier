import cv2
from gaze_tracking import GazeTracking

#######################################################################
# initiate variables:

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
direction_X = "Idle"
direction_Y = "Idle"
blink = "No"

#######################################################################

while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    ################################################################
    # blinking detection:

    blink = "No"
    if gaze.is_blinking():
        blink = "Yes"
    ################################################################

 
    ################################################################
    # gaze detection:

    direction_X = gaze.left_idle_right()
    ################################################################

    
    ################################################################
    # display:
        
    cv2.putText(frame, "Horizontal direction : " + direction_X, (100, 350), font, 2, (0, 0, 255), 3)
    cv2.putText(frame, "Blink : " + blink, (160, 100), font, 3, (0, 255, 0), 3)
    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) == ord("q"):
        break
    ################################################################
   
webcam.release()
cv2.destroyAllWindows()
