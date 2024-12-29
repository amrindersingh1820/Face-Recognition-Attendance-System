import cv2
import numpy as np
import os
import pickle

# Initialize webcam
video = cv2.VideoCapture(0)  # 0 is for the primary webcam
if not video.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Load face detector
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Face data list
face_data = []
i = 0

# Directory to save data
os.makedirs('data', exist_ok=True)

# Get user input
name = input("Enter your name: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Unable to capture video.")
        break  # Exit if the video feed is not working

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop and resize the face image
        crop_img = frame[y:y + h, x:x + w]
        resized_image = cv2.resize(crop_img, (50, 50))

        # Add face data every 10th frame
        if len(face_data) < 50 and i % 10 == 0:
            face_data.append(resized_image)
            print(f"Collected {len(face_data)} face images.")
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    # Display the frame
    cv2.imshow("Face Collector", frame)

    # Exit loop if 'q' is pressed or 50 images are collected
    if cv2.waitKey(1) & 0xFF == ord('q') or len(face_data) >= 50:
        break

    i += 1

# Release resources
video.release()
cv2.destroyAllWindows()

# Save faces to a pickle file
face_data = np.array(face_data)
face_data = face_data.reshape((len(face_data), -1))

# Save data into pickle files
names_file = 'data/names.pkl'
face_data_file = 'data/face_data.pkl'

# Save name entries
if os.path.exists(names_file):
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
else:
    names = []

names += [name] * len(face_data)  # Add 50 entries of the name
with open(names_file, 'wb') as f:
    pickle.dump(names, f)

# Save face data
if os.path.exists(face_data_file):
    with open(face_data_file, 'rb') as f:
        existing_faces = pickle.load(f)
        face_data = np.append(existing_faces, face_data, axis=0)

with open(face_data_file, 'wb') as f:
    pickle.dump(face_data, f)

print("Face data and names have been saved successfully.")