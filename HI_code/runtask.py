"""

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//       Too all users of this code:                                                  //
//       Please forgive this awful mess of code                                      //
//       I wrote this at 4am after working on other things since 8pm                //
//       Please be kind and if you want to make it look pretty please do           //
//       Just make sure to comment everything                                     //
///////////////////////////////////////////////////////////////////////////////////
///////
////
///
//
/


"""
# Module Imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os
import cv2 as cv
import numpy as np
import webbrowser
import time
import os
import threading




# creates alias for these classes
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode
ClassifierOptions = mp.tasks.components.processors.ClassifierOptions
# alias for drawing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
Htrue = False
Itrue = False





# absolute path to the gesture_recognizer TFlite file
model_path = os.getcwd() + "/exported_model/gesture_recognizer.task"
# base model recognizer options in common with other recognizers
base_opt= BaseOptions(model_asset_path=model_path)

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global Htrue
    global Itrue
    #sets browser choice for (webbrowser.open)
    browser = "firefox"
    #prints direct list data from mediapipe 
    print('gesture recognition result: {}'.format(result.gestures))
    #variable used to read list/string
    doot = ('gesture recognition result: {}'.format(result.gestures))
    #ddot variable used to find H within doot variable
    dot = 'H'
    #resulth used to see if H is in I
    resulth = dot in doot
    # prints if H has been seen by the mediapipe program 
    print(Htrue)
    # checks if H has been seen 
    if resulth == True:
        #sets Htrue to true
        Htrue=True
    #sets variable to resulti to see if there is an I in the string
    dang = 'I'
    resulti = (dang in doot) and (Htrue == True)
    if resulti == True:
        #if it sees I it sets Itrue to true
        Itrue = True
    #checks if both letters have been seen 
    if Htrue and Itrue == True:
        #resets values
        Htrue = False
        Itrue = False
        #opens webpage and tells terminal 
        print("welcome to autoslug")
        webbrowser.open('127.0.0.1')
    
       






    


    # annotated_image = draw_landmarks(frame, result)
    # image_result = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)
    # cv.imshow("frame", image_result)
    # cv.waitKey(1)

# sets up classifier options (not implemented, just here for future use)
custom_classifier_options = ClassifierOptions() # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/components/processors/ClassifierOptions

# sets up gesture recognizer options
options = GestureRecognizerOptions(
    base_options=base_opt,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    min_hand_detection_confidence=0.5
    )

# for some reason the documentation for anything mp.solutions is non existent
# this function does not work, To Do
def draw_landmarks(image, result: GestureRecognizerResult):
    annotated_image = np.copy(image)
    
    for hand_landmark in result.hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmark
          ])
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=hand_landmarks_proto,
            connections=mp.solutions.holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )
    return annotated_image
    



# Use OpenCV’s VideoCapture to start capturing from the webcam.
cap = cv.VideoCapture(0)

frame_timestamp_ms = 0
with GestureRecognizer.create_from_options(options) as recognizer:
    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    while cap.isOpened():
        # reads capture data and returns the frame
        ret, preprocessed_frame = cap.read()
        frame = cv.flip(preprocessed_frame, 1)
        if not ret:
            print("Empty Frame")
            break
                
        # checks if escape key (q) has been pressed every 200 ms
        if cv.waitKey(200) & 0xff == ord('q'):
            break
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        frame_timestamp_ms += 1
        # Send live image data to perform gesture recognition.
        # The results are accessible via the `result_callback` provided in
        # the `GestureRecognizerOptions` object.
        # The gesture recognizer must be created with the live stream mode.
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

# ends capture
cap.release()
cv.destroyAllWindows()

