""" Experiment with face detection and image filtering using OpenCV """

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
kernel = np.ones((21, 21), 'uint8')

cap = cv2.VideoCapture(0)
print(cap.isOpened())
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
        cv2.line(frame, (int(x+.25*w), int(y+.75*h)), (int(x+.75*w), int(y+.75*h)), (0, 0, 255), thickness=10)
        cv2.circle(frame, (int(x+.25*w), int(y+.4*h)), 20, (189, 103, 35), thickness=10)
        cv2.circle(frame, (int(x+.75*w), int(y+.4*h)), 20, (189, 103, 35), thickness=10)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
