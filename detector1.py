from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2
import time

model = load_model("/Users/rahulbhadoria/Desktop/Smart Bin/model_checkpoint.keras")


from tensorflow.keras.preprocessing import image
import numpy as np

def predict_single_image(frame):

    frame = cv2.resize(frame, (150,150))

    # Convert the image to an array
    img_array = image.img_to_array(frame)
    
    # Expand the dimensions to match the input shape of the model (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image
    img_array = preprocess_input(img_array)
    
    # Predict the class of the image
    prediction = model.predict(img_array)
    
    print(prediction)
    
    if prediction[0] > 0.5:
        return "Class R"
    else:
        return "Class O"


# Open the default camera (usually the first camera connected to the computer)
cap = cv2.VideoCapture(0)

last_capture_time = time.time()

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    # Capture and display video frames from the camera
    while True:
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break  # Break the loop if no frame is captured

        current_time =  time.time() 

        # Display the frame
        cv2.imshow('Camera Frame', frame)

        if current_time - last_capture_time > 5:
            predict_single_image(frame)
            last_capture_time = current_time

        # Wait for 1 ms and break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera capture object
    cap.release()
    cv2.destroyAllWindows()

