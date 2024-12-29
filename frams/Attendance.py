import cv2
import numpy as np
import os
import csv
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load saved face data and labels
with open('data/names.pkl', 'rb') as w:
    LABEL = pickle.load(w)

with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABEL)

# Load background image and define column names
imgbackground = cv2.imread('bg.png')
COL_NAMES = ['Name', 'Time']

# Main loop
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
        timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M-%S')

        # Attendance-related operations
        attendance = [str(output[0]), str(timeStamp)]

        # Ensure the Attendance directory exists
        attendance_dir = "Attendance"
        os.makedirs(attendance_dir, exist_ok=True)  # Create the directory if it doesn't exist

        file_path = f"{attendance_dir}/Attendance_ {date}.csv"
        exist = os.path.exists(file_path)

        # Draw rectangles for the face detection
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)

        # Display the predicted name
        cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        if exist:
            with open(file_path, '+a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
                csvfile.close()
        else:
            with open(file_path, '+a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
                csvfile.close()

        # Resize the frame to match the target region in the background image
        resized_frame = cv2.resize(frame, (640, 480))
        imgbackground[162:162 + 480, 55:55 + 640] = resized_frame

        # Display the combined frame
        cv2.imshow("frame", imgbackground)

        # Handle key presses
        k = cv2.waitKey(1)
        if k == ord('o'):
            time.sleep(5)

        if k == ord('q'):
            break

# Release video
video.release()
cv2.destroyAllWindows()
