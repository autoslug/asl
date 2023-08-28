# import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    print('gesture recognition result: {}'.format(result))

# configuration/options/settings for mediapipe
# https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer/python#configuration_options
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'), # needs the trained model path here
    running_mode=VisionRunningMode.LIVE_STREAM, # live video input mode to reduce latency
    num_hands=2, # max number of hands that can be recognized
    result_callback=print_result # displays result (documentation isn't clear about this)
    )

with GestureRecognizer.create_from_options(options) as recognizer:
    # The detector is initialized. Use it here.
    # ...
    pass

# Use OpenCV’s VideoCapture to start capturing from the webcam.
# Create a loop to read the latest frame from the camera using VideoCapture#read()
# Convert the frame received from OpenCV to a MediaPipe’s Image object.
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

# Send live image data to perform gesture recognition.
# The results are accessible via the `result_callback` provided in
# the `GestureRecognizerOptions` object.
# The gesture recognizer must be created with the live stream mode.
recognizer.recognize_async(mp_image, frame_timestamp_ms)

