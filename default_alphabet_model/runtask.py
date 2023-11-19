# Module Imports
import mediapipe as mp # https://google.github.io/mediapipe/getting_started/python.html
from mediapipe.tasks import python # .tasks.python is a module, https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/python
from mediapipe.tasks.python import vision # .tasks.python.vision is a module, https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/python/vision
from mediapipe import solutions # https://google.github.io/mediapipe/solutions/solutions.html
from mediapipe.framework.formats import landmark_pb2 # https://google.github.io/mediapipe/solutions/hands.html#normalized-landmark
import os # https://docs.python.org/3/library/os.html
import cv2 as cv # https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
import numpy as np # https://numpy.org/doc/stable/

# creates alias for these classes
BaseOptions = mp.tasks.BaseOptions # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/vision/GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/vision/GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/vision/GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/vision/RunningMode
ClassifierOptions = mp.tasks.components.processors.ClassifierOptions # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/components/processors/ClassifierOptions
# alias for drawing
mp_hands = mp.solutions.hands # https://google.github.io/mediapipe/solutions/hands.html
mp_drawing = mp.solutions.drawing_utils # https://google.github.io/mediapipe/solutions/drawing_utils.html
mp_drawing_styles = mp.solutions.drawing_styles # https://google.github.io/mediapipe/solutions/drawing_utils.html


# absolute path to the gesture_recognizer TFlite file
# TF Lite model path for the alphabet recognizer, which is used for the base model recognizer. 
# Base model recognizer is used to detect hands and crop the image.
# .getcwd() gets the current working directory
model_path = os.getcwd() + "/exported_model/gesture_recognizer.task" # string concatenation in python
# base model recognizer options in common with other recognizers
# model asset path is the path to the gesture_recognizer TFlite file
base_opt= BaseOptions(model_asset_path=model_path)

# Create a gesture recognizer instance with the live stream mode:
# result: what model predicted (gesturerecognizerresult)
# output_image: the image that was used to predict the result
# timestamp_ms: the time the image was taken
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if(result.gestures!=[]): # if a gesture is seen, then
        # format the result to be more readable
        # result is a list of tuples, where the first element is the gesture name and the second element is the confidence
        # .category_name gets the gesture name (like "A" or "B")
        print('gesture recognition result: {}'.format(result.gestures[0][0].category_name))
    # annotated_image = draw_landmarks(frame, result)
    # image_result = cv.cvtColor(annotated_image, cv.COLOR_RGB2BGR)
    # cv.imshow("frame", image_result)
    # cv.waitKey(1)

# sets up classifier options (not implemented, just here for future use)
custom_classifier_options = ClassifierOptions() # https://developers.google.com/mediapipe/api/solutions/python/mp/tasks/components/processors/ClassifierOptions

# sets up gesture recognizer options
options = GestureRecognizerOptions(
    base_options=base_opt, # base model recognizer options (as defined above)
    running_mode=VisionRunningMode.LIVE_STREAM, # live stream mode
    result_callback=print_result, # function that is called when a result is received
    min_hand_detection_confidence=0.5 # minimum confidence for hand detection
    )

# for some reason the documentation for anything mp.solutions is non existent
# this function does not work, To Do
def draw_landmarks(image, result: GestureRecognizerResult): # image is a numpy array, result is from the gesture recognizer from above
    annotated_image = np.copy(image) # annotated_image is a numpy array
    
    for hand_landmark in result.hand_landmarks:  # hand_landmarks is a list of lists of tuples
        # .NormalizedLandmarkList() creates a new landmark list, landmark_pb2 is a protobuf file
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList() # creates a new landmark list
        hand_landmarks_proto.landmark.extend([ # adds landmarks to the list
            # x, y, and z are the coordinates of the landmark
            # x and y are normalized, z is not
            # x and y are in the range [0, 1], z is in the range [-1, 1]
            # x and y are relative to the image, z is relative to the camera
            # for more info, see https://google.github.io/mediapipe/solutions/hands.html#normalized-landmark
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmark
          ])
        mp_drawing.draw_landmarks( # draws the landmarks on the image
            image=annotated_image, # the image to draw on
            landmark_list=hand_landmarks_proto, # the landmarks to draw
            connections=mp.solutions.holistic.HAND_CONNECTIONS, # the connections between the landmarks
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(), # the style of the landmarks, .get_default_hand_landmarks_style() is a function that returns a DrawingSpec object
            # drawing_spec is a DrawingSpec object, which is a protobuf file
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style() # the style of the connections, .get_default_hand_connections_style() is a function that returns a DrawingSpec object
            )
    return annotated_image # returns the image with the landmarks drawn on it
    



# Use OpenCV’s VideoCapture to start capturing from the webcam.
cap = cv.VideoCapture(0) # 0 is the default camera

frame_timestamp_ms = 0 # the time the frame was taken
with GestureRecognizer.create_from_options(options) as recognizer: # creates a gesture recognizer with the options defined above
    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    while cap.isOpened(): # while the capture is open
        # reads capture data and returns the frame
        ret, preprocessed_frame = cap.read() # ret is a boolean, preprocessed_frame is a numpy array, preprocessed_frame is the frame that was read
        frame = cv.flip(preprocessed_frame, 1) # flips the frame horizontally
        if not ret: # if the frame is empty, then
            print("Empty Frame") # prints "Empty Frame"
            break
                
        # checks if escape key (q) has been pressed every 200 ms
        if cv.waitKey(200) & 0xff == ord('q'): # if the escape key has been pressed, then break
            break
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) # creates a MediaPipe Image object from the frame, mp.ImageFormat.SRGB is the image format, frame is the image data

        frame_timestamp_ms += 1 # increments the frame timestamp by 1
        # Send live image data to perform gesture recognition.
        # The results are accessible via the `result_callback` provided in
        # the `GestureRecognizerOptions` object.
        # The gesture recognizer must be created with the live stream mode.
        recognizer.recognize_async(mp_image, frame_timestamp_ms) # recognizes the image asynchronously, mp_image is the image to recognize, frame_timestamp_ms is the time the image was taken

# ends capture
cap.release() # releases the capture
cv.destroyAllWindows() # destroys all windows

