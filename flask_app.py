from flask import Flask, render_template, request, jsonify
import cv2
import os
import numpy as np
import pandas as pd
import csv
import uuid

app = Flask(__name__,static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/continue', methods=['POST'])
def continue_process():
    inp = request.form['inp']
    name = request.form['name']
    img_name = request.form['img_name']
    
    data = pd.read_csv("nameStored.csv")
    subjects = list(data.loc[0])

    if inp == "yes":
        subjects.append(name)

    with open('nameStored.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(subjects)
        writer.writerow(subjects)

    data = pd.read_csv("nameStored.csv")
    subjects = list(data.loc[0])
    

    if inp == "yes":
        PHOTO_PATH = os.path.join('training-data', f's{len(subjects)-1}')
        os.makedirs(PHOTO_PATH)
        os.path.join(PHOTO_PATH, '{}.jpg'.format(uuid.uuid1()))

        

        cap = cv2.VideoCapture(0)
        while cap.isOpened(): 
            ret, frame = cap.read()
        
            frame = frame[120:120+250,200:200+250, :]
            
            if cv2.waitKey(1) & 0XFF == ord('a'):
                imgname = os.path.join(PHOTO_PATH, '{}.jpg'.format(uuid.uuid1()))
                cv2.imwrite(imgname, frame)
            
            cv2.imshow('Image Collection', frame)
            
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
    def detect_face(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        if len(faces) == 0:
            return None, None
        
        (x, y, w, h) = faces[0]
        return gray[y:y+w, x:x+h], faces[0]
    
    def prepare_training_data(data_folder_path):
        dirs = os.listdir(data_folder_path)
        faces = []
        labels = []
        
        for dir_name in dirs:
            if not dir_name.startswith("s"):
                continue
            label = int(dir_name.replace("s", ""))
            subject_dir_path = os.path.join(data_folder_path, dir_name)
            subject_images_names = os.listdir(subject_dir_path)
            
            for image_name in subject_images_names:
                if image_name.startswith("."):
                    continue
                image_path = os.path.join(subject_dir_path, image_name)
                image = cv2.imread(image_path)
                face, rect = detect_face(image)
                
                if face is not None:
                    faces.append(face)
                    labels.append(label)
                    
        return faces, labels

    
    def predict_face(test_img):
        img = test_img.copy()
        face, rect = detect_face(img)
        label = face_recognizer.predict(face)
        label_text = subjects[label[0]]
        
        draw_rectangle(img, rect)
        draw_text(img, label_text, rect[0], rect[1]-5)
        
        return img
    
    faces, labels = prepare_training_data("training-data")
    
    # Create LBPH face recognizer
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    def draw_rectangle(img, rect):
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

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

    def instr1():
            return "Adjust the camera for capturing image. /n Press 'v' to capture image. /n Then press q for close the camera"

    instr1()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[120:120+250,200:200+250, :]
        
        cv2.imshow('Verification', frame)
        
        # Verification trigger
        if cv2.waitKey(10) & 0xFF == ord('v'):
            # Save input image to application_data/input_image folder 
    #         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #         h, s, v = cv2.split(hsv)

    #         lim = 255 - 10
    #         v[v > lim] = 255
    #         v[v <= lim] -= 10
            
    #         final_hsv = cv2.merge((h, s, v))
    #         img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            def print2():
                return "image is captured."
            print2()
            cv2.imwrite(os.path.join('test-data',f'{img_name}.jpg'), frame)
            # Run verification
            
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    

    #load test images
    test_img1 = cv2.imread(f"test-data/{request.form['img_name']}.jpg")






    #perform a prediction
    predicted_img1 = predict(test_img1)

    

    #display both images
    cv2.imshow(subjects[len(subjects)-1], predicted_img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

    return "Click on back arrow to check again. 😊"








if __name__ == '__main__':
    app.run(debug=True)
