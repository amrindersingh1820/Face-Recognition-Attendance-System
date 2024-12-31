import cv2
import numpy as np
import os
import csv
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Load necessary data and models
def load_data():
    try:
        with open('data/names.pkl', 'rb') as w:
            labels = pickle.load(w)

        with open('data/face_data.pkl', 'rb') as f:
            faces = pickle.load(f)

        return labels, faces
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

# Initialize video capture and face detection
def initialize_camera():
    try:
        video = cv2.VideoCapture(0)
        facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if facedetect.empty():
            raise FileNotFoundError("Error loading haarcascade_frontalface_default.xml")
        return video, facedetect
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

# Ensure the Attendance directory exists
def ensure_directory():
    attendance_dir = "Attendance"
    os.makedirs(attendance_dir, exist_ok=True)
    return attendance_dir

# Log attendance, ensuring no duplicates
def log_attendance(file_path, col_names, attendance):
    already_logged = set()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            already_logged = {row[0] for row in reader if row}

    if attendance[0] not in already_logged:
        with open(file_path, 'a') as f:
            writer = csv.writer(f)
            if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
                writer.writerow(col_names)
            writer.writerow(attendance)

# Main loop
def main():
    labels, faces = load_data()
    video, facedetect = initialize_camera()

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)

    try:
        imgbackground = cv2.imread('bg.png')
        if imgbackground is None:
            raise FileNotFoundError("Error loading bg.png")

        col_names = ['Name', 'Time']
        attendance_dir = ensure_directory()

        while True:
            ret, frame = video.read()
            if not ret:
                print("Error: Unable to access camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x:x + w]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                output = knn.predict(resized_img)
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                # Attendance-related operations
                attendance = [str(output[0]), str(timeStamp)]
                file_path = f"{attendance_dir}/Attendance_{date}.csv"
                log_attendance(file_path, col_names, attendance)

                # Draw rectangles for the face detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)

                # Display the predicted name
                cv2.putText(frame, str(output[0]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display the date and time on the frame
            current_time = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
            cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Resize the frame to match the target region in the background image
            resized_frame = cv2.resize(frame, (640, 480))
            imgbackground[162:162 + 480, 55:55 + 640] = resized_frame

            # Display the combined frame
            cv2.imshow("Attendance System", imgbackground)

            # Handle key presses
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("Exiting...")
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
