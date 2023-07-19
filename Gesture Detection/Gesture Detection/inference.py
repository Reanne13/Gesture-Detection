import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 

# Load the pre-trained Keras model
model  = load_model("model.h5")
# Load the labels
label = np.load("labels.npy")

# Initialize mediapipe holistic and hands solutions
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()

# Initialize the drawing utilities
drawing = mp.solutions.drawing_utils

# Open the default camera
cap = cv2.VideoCapture(0)

# Main loop
while True:
	lst = []

	# Read frame from the camera
	_, frm = cap.read()

	# Flip the frame horizontally
	frm = cv2.flip(frm, 1)

	# Process the frame with mediapipe holistic
	res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

	# Extract facial landmarks and calculate their relative positions
	if res.face_landmarks:
		for i in res.face_landmarks.landmark:
			lst.append(i.x - res.face_landmarks.landmark[1].x)
			lst.append(i.y - res.face_landmarks.landmark[1].y)

		# Extract left hand landmarks and calculate their relative positions
		if res.left_hand_landmarks:
			for i in res.left_hand_landmarks.landmark:
				lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
		else:
			# If no left hand landmarks are detected, append zeros
			for i in range(42):
				lst.append(0.0)

		# Extract right hand landmarks and calculate their relative positions
		if res.right_hand_landmarks:
			for i in res.right_hand_landmarks.landmark:
				lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
				lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
		else:
			# If no right hand landmarks are detected, append zeros
			for i in range(42):
				lst.append(0.0)

		# Reshape the list as input for the model
		lst = np.array(lst).reshape(1, -1)

		# Predict the gesture label using the model
		pred = label[np.argmax(model.predict(lst))]

		# Print the predicted gesture label
		print(pred)

		# Draw the predicted gesture label on the frame
		cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

	# Draw the face landmarks, left hand landmarks, and right hand landmarks on the frame
	drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

	# Show the frame in a window
	cv2.imshow("window", frm)

	# Break the loop if the 'Esc' key is pressed
	if cv2.waitKey(1) == 27:
		cv2.destroyAllWindows()
		cap.release()
		break
