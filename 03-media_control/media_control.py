import numpy as np
from keras.models import Sequential, load_model
import sys
import cv2
from cv2 import aruco
from pynput.keyboard import Key, Controller, KeyCode




# code from now on taken from Assignment 4
WIDTH = 100
HEIGHT = 100
SATURATION_THRESH = 38
BLUR_RADIUS = 9
MIN_HAND_POINT_DISTANCE = 0.085

mapping = {
    "rock": "play",
    "peace": "pause",
    "stop": "skip"
}


video_id = 0

if len(sys.argv) > 1:
    video_id = int(sys.argv[1])

# Define the ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
aruco_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# Create a video capture object for the webcam
cap = cv2.VideoCapture(video_id)
area_corners = np.array([[-1,-1],
                         [-1,-1],
                         [-1,-1],
                         [-1,-1]]) # dummy points for homography

def extract_area (frame:np.array, area_corners:np.array )->tuple[bool,np.ndarray]:
    WIDTH = frame.shape[1]
    HEIGHT = frame.shape[0]
    big_img_cornerpoints = np.array([[0     , 0],
                                    [WIDTH , 0],
                                    [WIDTH , HEIGHT],
                                    [0     , HEIGHT]])

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the frame
    corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(gray)
    # Check if marker is detected
    if ids is not None:
        # Draw lines along the sides of the marker
        aruco.drawDetectedMarkers(frame, corners)
        if len(ids) == 4:

            corners = np.array(corners)[ids.flatten().argsort()]
            for i, corner in enumerate(corners):
                area_corners[i] = corner[0][(i+2)%4]

    if not -1 in area_corners:
        # correct image
        homography, ret = cv2.findHomography(area_corners,big_img_cornerpoints)
        result_img = cv2.warpPerspective(frame, homography, (WIDTH, HEIGHT))
        return True,result_img
    else:
        return False, cv2.resize(frame, (WIDTH, HEIGHT))

def get_center_of_hand(frame:np.array)->tuple[np.ndarray, int]:

    #thresh = cv2.adaptiveThreshold(result_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 121, 2)
    #ret, thresh = cv2.threshold(result_img, 120, 255, cv2.THRESH_BINARY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    s = cv2.GaussianBlur(s, (BLUR_RADIUS,BLUR_RADIUS), 0)
    ret, s_thresh = cv2.threshold(s, SATURATION_THRESH, 255, cv2.THRESH_BINARY)
    if ret:
        #contours, hierarchy = cv2.findContours(s_thresh, 1, 2)
        dist = cv2.distanceTransform(s_thresh,cv2.DIST_L2,5)
        max_dist = np.max(dist)
        index = None
        if max_dist > WIDTH*MIN_HAND_POINT_DISTANCE:
            index  = np.unravel_index(dist.argmax(), dist.shape)[::-1]
        #dist = cv2.cvtColor(dist, cv2.COLOR_GRAY2BGR)
        
        #cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        #if not index is None:
        #    dist = cv2.circle(dist, index, 15, (0,0,255))
            
        #cv2.imshow("ASDF", dist)
        return index, max_dist
    else:
        return None, None
    
    # end of code from Assignment 4

cv2.namedWindow("DEBUG")

# finds hand, extracts ROI.
def crop_roi(frame:np.array, target_width:int, target_height:int)->np.ndarray|None:
    padding = 180
    point, distance = get_center_of_hand(frame)

    distance += padding
    if point is None or distance is None:
        return None
    frame_h, frame_w, channels = frame.shape
    center_x, center_y = point
    left_bound = int(max(0, center_x - distance))
    right_bound =  int(min(center_x + distance, frame_w))
    upper_bound =  int(max(0, center_y - distance))
    lower_bound =  int(min(center_y + distance, frame_h))

    cropped = frame[upper_bound:lower_bound, left_bound:right_bound]
    cropped = cv2.resize(cropped, (target_width, target_height))
    return cropped

keyboard = Controller()

model_path:str = 'gesture_recognition_for_media_control_1.keras'
model:Sequential = load_model(model_path)
print(model.summary())

GESTURES = ['rock', 'no_gesture', 'ok', 'like', 'dislike', 'peace']

status = None
is_playing = False

while True:
    ret, frame = cap.read()
    if ret and not frame is None:
        frame = np.array(frame)
        ret, area = extract_area(frame, area_corners)
        roi = crop_roi(area, 64,64)
        if not roi is None:
            cv2.imshow("DEBUG", roi)

            #convert ROI to be suitable input to NN
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = roi/255.
            #reshaped = roi.reshape(-1, 64, 64, 3)


            prediction = model.predict(np.array([roi]), verbose=0)
            predicted_gesture = GESTURES[np.argmax(prediction)]

            if predicted_gesture in mapping.keys():
                action = mapping[predicted_gesture]

                # trigger keypress only when the status changes
                if status != action:
                    status = action
                    
                    # press right keypress depending on status
                    if action == "play" and not is_playing:
                        keyboard.press(Key.space)
                        keyboard.release(Key.space)
                        is_playing = True
                        print("NOW PLAYING")
                    elif action == "pause" and is_playing:
                        keyboard.press(Key.space)
                        keyboard.release(Key.space)
                        is_playing = False
                        print("PAUSED")

                    elif action == "skip":
                        # unlike Play/Pause, skipping to the next track isn't really standardised between media players.
                        # this key combination is for skipping videos on YouTube.
                        # Depending on your media player, you may want to substitute this with the correct key combination
                        keyboard.press(Key.shift)
                        keyboard.press('n')
                        keyboard.release(Key.shift)
                        keyboard.release('n')


                
        

        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

