import tensorflow as tf
assert tf.__version__.startswith('2')
import os
from mediapipe_model_maker import gesture_recognizer
import matplotlib.pyplot as plt


# https://developers.google.com/mediapipe/solutions/customization/gesture_recognizer

# insert training and test data import here

# dataset path
dataset_path = os.path.dirname(os.path.abspath(__file__))  + "/Training Data/Gesture Image Pre-Processed Data/" # insert local path here
print(dataset_path)

# dataset labels
labels = []
for i in os.listdir(dataset_path):
  if os.path.isdir(os.path.join(dataset_path, i)):
    labels.append(i)
print(labels)

# splits dataset
data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=gesture_recognizer.HandDataPreprocessingParams(shuffle=True, min_detection_confidence=0.8)
)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# train model
hparams = gesture_recognizer.HParams(export_dir="exported_model")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# evaluates performance/loss
loss, acc = model.evaluate(test_data, batch_size=1)
print(f"Test loss:{loss}, Test accuracy:{acc}")

# export trained model
model.export_model() # insert steps here idk what, but it requires the model to be exported as a TFlite file
