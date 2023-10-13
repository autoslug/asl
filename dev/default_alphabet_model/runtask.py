# Module Imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import os
import cv2 as cv
import numpy as np

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
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)


# absolute path to the gesture_recognizer TFlite file
model_path = os.getcwd() + "/exported_model/gesture_recognizer.task"
# base model recognizer options in common with other recognizers
base_opt= BaseOptions(model_asset_path=model_path)

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    if(result.gestures!=[]):
        print('gesture recognition result: {}'.format(result.gestures[0][0].category_name))
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
# mine is set to one make sure you set your number to the correct number
cap = cv.VideoCapture(1)

frame_timestamp_ms = 0
with GestureRecognizer.create_from_options(options) as recognizer:
    # Create a loop to read the latest frame from the camera using VideoCapture#read()
    while cap.isOpened():
        # reads capture data and returns the frame
        ret, preprocessed_frame = cap.read()
        frame = cv.flip(preprocessed_frame, 1)
        framergb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        x, y, c = frame.shape
        result = hands.process(framergb)
        if not ret:
            print("Empty Frame")
            break
                
        # checks if escape key (q) has been pressed every 200 ms
        if cv.waitKey(200) & 0xff == ord('q'):
            break
        # Convert the frame received from OpenCV to a MediaPipe’s Image object.
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mp_drawing.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)

           
        frame_timestamp_ms += 1
        # Send live image data to perform gesture recognition.
        # The results are accessible via the `result_callback` provided in
        # the `GestureRecognizerOptions` object.
        # The gesture recognizer must be created with the live stream mode.
        recognizer.recognize_async(mp_image, frame_timestamp_ms)

        #overlays the text on the image
        cv.putText(frame, "Press q to quit", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)

        #shows image
        cv.imshow('MediaPipe Gesture Recognition', frame)
         


# ends capture
cap.release()
cv.destroyAllWindows()

