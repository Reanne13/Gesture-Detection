import mediapipe as mp  # Importing the Mediapipe library for hand and face landmarks detection
import numpy as np  # Importing the Numpy library for numerical operations
import cv2  # Importing the OpenCV library for image processing

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Prompt the user to enter a name for the data
name = input("Enter the name of the data: ")

# Create instances of the MediaPipe Holistic and Hands solutions
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()

# Create an instance of the MediaPipe DrawingUtils utility for drawing landmarks
drawing = mp.solutions.drawing_utils

X = []  # Create an empty list to store landmark data for each frame
data_size = 0  # Initialize the number of frames captured to zero

# start the webcam feed
while True:
    # Create an empty list to store the normalized landmark coordinates for the current frame
    lst = []

    # read a frame from the webcam
    _, frm = cap.read()

    # flip the frame horizontally to mirror it
    frm = cv2.flip(frm, 1)

    # process the frame with the holistic model
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # if face landmarks are detected, extract the features
    if res.face_landmarks:
        # calculate the relative positions of each face landmark with respect to the second landmark (the nose)
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        # if left hand landmarks are detected, extract the features
        if res.left_hand_landmarks:
            # calculate the relative positions of each left hand landmark with respect to the eighth landmark (the index finger tip)
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        # if no left hand landmarks are detected, append 0 for each feature
        else:
            for i in range(42):
                lst.append(0.0)

        # if right hand landmarks are detected, extract the features
        if res.right_hand_landmarks:
            # calculate the relative positions of each right hand landmark with respect to the eighth landmark (the index finger tip)
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        # if no right hand landmarks are detected, append 0 for each feature
        else:
            for i in range(42):
                lst.append(0.0)

        # Add the list of landmark coordinates for the current frame to the list of landmark data for all frames
        X.append(lst)

        # Increment the number of frames captured
        data_size += 1

    # Draw the facial and hand landmarks on the frame using the DrawingUtils utility
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # put the data size counter in the top left corner of the frame
    cv2.putText(frm, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    # Display the current frame in a window named "window"
    cv2.imshow("window", frm)

    # Wait for a key event for 1 millisecond. If the key pressed is the "Esc" key (keycode 27) or the number of samples is greater than 99, then break the loop
    if cv2.waitKey(1) == 27 or data_size > 99:
        cv2.destroyAllWindows()  # Close all the open windows
        cap.release()  # Release the capture object
        break

# Save the data in the numpy array X as a numpy binary file with the given filename
np.save(f"{name}.npy", np.array(X))

# Print the shape of the numpy array X
print(np.array(X).shape)
