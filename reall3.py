from keras.models import load_model
import cv2
import numpy as np
import tensorflow as tf

# Disable TensorFlow logging
tf.get_logger().setLevel('ERROR')

np.set_printoptions(suppress=True)

model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

a = 0

prev_label = ""
current_label = ""

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    a +=1

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    flipped = cv2.flip(image, 1)
    # Show the image in a window
    cv2.imshow("Webcam Image", flipped)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)


    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    # print('d')
    prediction = model.predict(image)#이부분에서 '1/1 [==============================] - 0s 32ms/step'이거 비슷한거 뜸
    # print('o')
    index = np.argmax(prediction)
    # print('100')
    class_name = class_names[index]
    # print('10000000')
    confidence_score = prediction[0][index]

    if prev_label == "":
        prev_label = class_name[2:]
        continue 
    
    if prev_label != current_label:
        print(">>> new command fired...")
        prev_label = class_name[2:]

    if a == 10:
        # print(class_name[2:], end="")
        a = 0
    # Print prediction and confidence score
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
