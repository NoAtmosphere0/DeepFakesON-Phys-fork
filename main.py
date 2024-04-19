import threading

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 0 is the default camera

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0 # counter for the number of frames

face_macth = False # flag to check if the face is matched or not

reference_img = cv2.imread('reference.jpg') # read the reference image

def check_face(frame):
    global face_macth
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_macth = True
        else:
            face_macth = False       
    except ValueError:
        face_macth = False  
while True:
    ret, frame = cap.read() # read the frame from the camera
    
    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start() # add a coma at the end of the argument to make it a tuple
            except ValueError:
                pass # ignore the error
        counter += 1
        
        if face_macth:
            cv2.putText(frame, "Face Matched", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Face Not Matched", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
        cv2.imshow('video', frame) # show the frame in a window
    
    key = cv2.waitKey(1) 
    if key == ord('q'): # if the user press 'q' break the loop
        break

cv2.destroyAllWindows() # close the camera window
    