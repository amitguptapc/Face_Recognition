# Recognise Faces using KNN algorithm

# sequence of operation :
# 1. load the training data (numpy arrays of all the persons)
		# x-values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use KNN to find the prediction of face (int)
# 5. map the predicted id to name of the user
# 6. Display the predictions on the screen - bounding box and name

import numpy as np
import cv2
import os

# Begin of KNN Algorithm
def distance(v1, v2):
	# Eucledian Distance
	return np.sqrt(((v1-v2)**2).sum())

def KNN(train, test, k=5):
	dist = []
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
# End of KNN Algorithm


# Initilize camera
cap = cv2.VideoCapture(0)
# Face detection using Haarcascade Classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = "Data/"
face_data = []
label = []

class_id = 0 # labels for given files
names = {} # mapping the ids with names

# Data Preparation
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4] # creates a label b/w class and name
        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # create labels for the class
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)

face_dataset = np.concatenate(face_data,axis = 0)
face_labels = np.concatenate(label,axis = 0).reshape((-1,1))
# Combinig dataset and labels into a single matrix
trainset = np.concatenate((face_dataset,face_labels),axis = 1)

# Testing the Classifier
while True:
    ret,frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h = face

        # get face region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        # predicted label
        out = KNN(trainset,face_section.flatten())

        # Display the name
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    cv2.imshow("Faces",frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
