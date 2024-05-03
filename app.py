import streamlit as st
import cv2
import os
import numpy as np
import pandas as pd
import csv
import uuid

st.title("Face Recognition System")

inp = st.text_input("Do  you want to append name? yes/no")
name=st.text_input("Enter name")
img_name=st.text_input("Enter image name")





if st.button("continue"):
    
    data=pd.read_csv("nameStored.csv")

    subjects=[]
    for _ in data.loc[0]:
        subjects.append(_)


    if inp == "yes":
        subjects.append(name)

    with open('nameStored.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(subjects)
        writer.writerow(subjects)

    data=pd.read_csv("nameStored.csv")
    subjects=[]
    for _ in data.loc[0]:
        subjects.append(_)

    if inp == "yes":
        # Setup paths
        PHOTO_PATH = os.path.join('training-data',f's{len(subjects)-1}')
        # Make the directories
        os.makedirs(PHOTO_PATH)

        os.path.join(PHOTO_PATH, '{}.jpg'.format(uuid.uuid1()))

        # Establish a connection to the webcam
        cap = cv2.VideoCapture(0)
        while cap.isOpened(): 
            ret, frame = cap.read()
        
            # Cut down frame to 250x250px
            frame = frame[120:120+250,200:200+250, :]
            
            # Collect PHOTO 
            if 0XFF == ord('a'):
                # Create the unique file path 
                imgname = os.path.join(PHOTO_PATH, '{}.jpg'.format(uuid.uuid1()))
                # Write out anchor image
                cv2.imwrite(imgname, frame)
            
            # Show image back to screen
            cv2.imshow('Image Collection', frame)
            
            # Breaking gracefully
            if 0XFF == ord('q'):
                break
                
        # Release the webcam
        cap.release()
        


    #function to detect face using OpenCV
    def detect_face(img):
        #convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #load OpenCV face detector, I am using LBP which is fast
        #there is also a more accurate but slow Haar classifier
        face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

        #let's detect multiscale (some images may be closer to camera than others) images
        #result is a list of faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
        
        #if no faces are detected then return original img
        if (len(faces) == 0):
            return None, None
        
        #under the assumption that there will be only one face,
        #extract the face area
        (x, y, w, h) = faces[0]
        
        #return only the face part of the image
        return gray[y:y+w, x:x+h], faces[0]


    #this function will read all persons' training images, detect face from each image
    #and will return two lists of exactly same size, one list 
    # of faces and another list of labels for each face
    def prepare_training_data(data_folder_path):
        
        #------STEP-1--------
        #get the directories (one directory for each subject) in data folder
        dirs = os.listdir(data_folder_path)
        
        #list to hold all subject faces
        faces = []
        #list to hold labels for all subjects
        labels = []
        
        #let's go through each directory and read images within it
        for dir_name in dirs:
            
            #our subject directories start with letter 's' so
            #ignore any non-relevant directories if any
            if not dir_name.startswith("s"):
                continue;
                
            #------STEP-2--------
            #extract label number of subject from dir_name
            #format of dir name = slabel
            #, so removing letter 's' from dir_name will give us label
            label = int(dir_name.replace("s", ""))
            
            #build path of directory containin images for current subject subject
            #sample subject_dir_path = "training-data/s1"
            subject_dir_path = data_folder_path + "/" + dir_name
            
            #get the images names that are inside the given subject directory
            subject_images_names = os.listdir(subject_dir_path)
            
            #------STEP-3--------
            #go through each image name, read image, 
            #detect face and add face to list of faces
            for image_name in subject_images_names:
                
                #ignore system files like .DS_Store
                if image_name.startswith("."):
                    continue;
                
                #build image path
                #sample image path = training-data/s1/1.pgm
                image_path = subject_dir_path + "/" + image_name

                #read image
                image = cv2.imread(image_path)
                
                #display an image window to show the image 
                st.image(image, caption="Training on image...")
                
                
                #detect face
                face, rect = detect_face(image)
                
                
                #------STEP-4--------
                #for the purpose of this tutorial
                #we will ignore faces that are not detected
                if face is not None:
                    #add face to list of faces
                    faces.append(face)
                    #add label for this face
                    labels.append(label)
                
        
        
        return faces, labels

    #let's first prepare our training data
    #data will be in two lists of same size
    #one list will contain all the faces
    #and other list will contain respective labels for each face
    st.write("Preparing data...")
    faces, labels = prepare_training_data("training-data")
    st.write("Data prepared")

    #print total faces and labels
    st.write("Total faces: ", len(faces))
    st.write("Total labels: ", len(labels))

    #create our LBPH face recognizer 
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    #train our face recognizer of our training faces
    face_recognizer.train(faces, np.array(labels))

    #function to draw rectangle on image 
    #according to given (x, y) coordinates and 
    #given width and heigh
    def draw_rectangle(img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    #function to draw text on give image starting from
    #passed (x, y) coordinates. 
    def draw_text(img, text, x, y):
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


    #this function recognizes the person in image passed
    #and draws a rectangle around detected face with name of the 
    #subject
    def predict(test_img):
        #make a copy of the image as we don't want to chang original image
        img = test_img.copy()
        #detect face from the image
        face, rect = detect_face(img)

        #predict the image using our face recognizer 
        label= face_recognizer.predict(face)
        print(label[0])
        #get name of respective label returned by face recognizer
        label_text = subjects[label[0]]
        
        #draw a rectangle around face detected
        draw_rectangle(img, rect)
        #draw name of predicted person
        draw_text(img, label_text, rect[0], rect[1]-5)
        
        return img

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[120:120+250,200:200+250, :]
        
        cv2.imshow('Verification', frame)
        
        # Verification trigger
        if 0xFF == ord('v'):
            # Save input image to application_data/input_image folder 
    #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #         h, s, v = cv2.split(hsv)

    #         lim = 255 - 10
    #         v[v > lim] = 255
    #         v[v <= lim] -= 10
            
    #         final_hsv = cv2.merge((h, s, v))
    #         img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            st.write("image is captured")
            
            cv2.imwrite(os.path.join('test-data',f'{img_name}.jpg'), frame)
            # Run verification
            
        
        if 0xFF == ord('q'):
            break
    cap.release()
    


    st.write("Predicting images...")

    #load test images
    test_img1 = cv2.imread(f"test-data/{img_name}.jpg")






    #perform a prediction
    predicted_img1 = predict(test_img1)

    st.write("Prediction complete")

    #display both images
    cv2.imshow(subjects[len(subjects)-1], predicted_img1)

    
    

