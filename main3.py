import pathlib
import cv2
import numpy as np

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / 'data' / 'haarcascade_frontalface_default.xml'
clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = frame[y:y+h, x:x+w]

        # Convert face ROI to HSV color space
        hsv_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

        # Calculate average skin hue
        avg_hue = np.average(hsv_roi[:, :, 0])

        # Determine skin color based on hue range
        if 0 <= avg_hue < 15:
            skin_color = "Reddish"
        elif 15 <= avg_hue < 30:
            skin_color = "Yellowish"
        elif 30 <= avg_hue < 90:
            skin_color = "Light"
        elif 90 <= avg_hue < 150:
            skin_color = "Tan"
        else:
            skin_color = "Dark"

        # Draw rectangle and put skin color text above
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, skin_color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
        break

camera.release()
cv2.destroyAllWindows()