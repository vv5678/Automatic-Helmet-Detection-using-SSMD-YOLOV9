from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import numpy as np
import cv2 as cv
import subprocess
import time
import os
from yoloDetection import detectObject, displayImage
import sys
from time import sleep
from tkinter import messagebox
from keras.models import load_model
import imutils
import time
import pickle
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Helmet Detection") #designing main screen
main.geometry("800x700")

global filename
global loaded_model

global class_labels
global cnn_model
global cnn_layer_names

CLASSES_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES_NAMES), 3))

ssmd_net = cv.dnn.readNetFromCaffe('Models/SSMD_deploy.prototxt.txt', 'Models/SSMD_deploy.caffemodel')

loaded_model = load_model('Models/new_helmet_model.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


frame_count = 0 
frame_count_out=0  

confThreshold = 0.5  
nmsThreshold = 0.4   
inpWidth = 416       
inpHeight = 416      
global option

classesFile = "Models/obj.names";
classes = None

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')


modelConfiguration = "Models/yolov3-obj.cfg";
modelWeights = "Models/yolov3-obj_2400.weights";

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def loadModels(): #function to load yolov3 model weight and class labels
    global class_labels
    global cnn_model
    global cnn_layer_names
    class_labels = open('yolov3model/yolov3-labels').read().strip().split('\n') #reading labels from yolov3 model
    print(str(class_labels)+" == "+str(len(class_labels)))
    cnn_model = cv.dnn.readNetFromDarknet('yolov3model/yolov3.cfg', 'yolov3model/yolov3.weights') #reading model
    cnn_layer_names = cnn_model.getLayerNames() #getting layers from cnn model
    cnn_layer_names = [cnn_layer_names[i[0] - 1] for i in cnn_model.getUnconnectedOutLayers()] #assigning all layers
    textarea.insert(END,'Yolo & SSMD models loaded')
        

def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askopenfilename(initialdir="bikes")
    textarea.delete('1.0', END)
    textarea.insert(END,str(filename)+' loaded')
    #messagebox.showinfo("File Information", "image file loaded")
    


def detectBike():
    global option
    option = 0
    indexno = 0
    label_colors = (0,255,0)
    try:
        image = cv.imread(filename)
        image_height, image_width = image.shape[:2]
    except:
        raise 'Invalid image path'
    finally:
        image, ops = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels,indexno)
        if ops == 1:
            displayImage(image,0)#display image with detected objects label
            option = 1
        else:
            displayImage(image,0)
            
    
def drawPred(classId, conf, left, top, right, bottom,frame,option):
    global frame_count
    #cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    label_name,label_conf = label.split(':')
    print(label_name+" "+str(conf)+" "+str(option))
    if label_name == 'Helmet' and conf > 0.50:
        if option == 0 and conf >= 0.65:
            cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            frame_count+=1
        if option == 0 and conf < 0.65:
            cv.putText(frame, "Helmet Not detected", (10, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
            frame_count+=1
        if option == 1:
            cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
            frame_count+=1    
    
        
    if(frame_count> 0):
        return frame_count

def postprocess(frame, outs, option):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    global frame_count_out
    frame_count_out=0
    classIds = []
    confidences = []
    boxes = []
    classIds = []
    confidences = []
    boxes = []
    cc = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                #print(classIds)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    count_person=0 # for counting the classes in this loop.
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        frame_count_out = drawPred(classIds[i], confidences[i], left, top, left + width, top + height,frame,option)
        my_class='Helmet'      
        unknown_class = classes[classId]
        if my_class == unknown_class:
            count_person += 1
    print(frame_count_out)
    if count_person == 0 and option == 1:
        cv.putText(frame, "Helmet Not detected", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
    if count_person >= 1 and option == 0:
        #path = 'test_out/'
        #cv.imwrite(str(path)+str(cc)+".jpg", frame)     # writing to folder.
        #cc = cc + 1
        cv.imshow('img',frame)
        cv.waitKey(50)


def yoloDetection():
    textarea.delete('1.0', END)
    frame = cv.imread(filename)
    frame_count =0
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    postprocess(frame, outs,0)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    print(label)
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    

def ssdDetection():
    
    frame = cv.imread(filename)
    frame = imutils.resize(frame, width=600, height=600)
    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    ssmd_net.setInput(blob)
    detections = ssmd_net.forward()  # getting the detections from the network
    persons = []
    person_roi = []
    motorbi = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                persons.append((startX, startY, endX, endY))

            if idx == 14:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                motorbi.append((startX, startY, endX, endY))

    xsdiff = 0
    xediff = 0
    ysdiff = 0
    yediff = 0
    p = ()
    for i in motorbi:
        mi = float("Inf")
        for j in range(len(persons)):
            xsdiff = abs(i[0] - persons[j][0])
            xediff = abs(i[2] - persons[j][2])
            ysdiff = abs(i[1] - persons[j][1])
            yediff = abs(i[3] - persons[j][3])
            if (xsdiff+xediff+ysdiff+yediff) < mi:
                mi = xsdiff+xediff+ysdiff+yediff
                p = persons[j]
                # r = person_roi[j]


        if len(p) != 0:
            label = "{}".format(CLASSES_NAMES[14])
            print("[INFO] {}".format(label))
            cv.rectangle(frame, (i[0], i[1]), (i[2], i[3]), COLORS[14], 2)
            y = i[1] - 15 if i[1] - 15 > 15 else i[1] + 15
            cv.putText(frame, label, (i[0], y), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[14], 2)   
            label = "{}".format(CLASSES_NAMES[15])
            print("[INFO] {}".format(label))
            cv.rectangle(frame, (p[0], p[1]), (p[2], p[3]), COLORS[15], 2)
            y = p[1] - 15 if p[1] - 15 > 15 else p[1] + 15
            roi = frame[p[1]:p[1]+(p[3]-p[1])//4, p[0]:p[2]]
            print(roi)
            if len(roi) != 0:
                img_array = cv.resize(roi, (50,50))
                gray_img = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
                img = np.array(gray_img).reshape(1, 50, 50, 1)
                img = img/255.0
                prediction = loaded_model.predict_proba([img])
                print("SSMD " +str(round(prediction[0][0],2)))
                if round(prediction[0][0],2) > 0.60:
                    cv.rectangle(frame, (p[0], p[1]), (p[0]+(p[2]-p[0]), p[1]+(p[3]-p[1])//4), COLORS[0], 2)
                    cv.putText(frame, "helmet "+str(round(prediction[0][0],2)), (p[0], y), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)
                else:
                    cv.putText(frame, "Helmet Not detected", (p[0], y), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)

    cv.imshow('Frame', frame)  # Displaying the frame
    cv.waitKey(0)
    cv.destroyAllWindows()

def graph():
    f = open('Models/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    plt.plot(data['loss'])
    plt.plot(data['val_loss'])
    plt.title('SSMD & Yolo Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['SSMD Loss', 'Yolo Loss'], loc='upper right')
    plt.show()    

def accuracyGraph():
    f = open('Models/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    plt.plot(data['accuracy'])
    plt.plot(data['val_accuracy'])
    plt.title('SSMD & Yolo Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['SSMD', 'Yolo'], loc='lower right')
    plt.show()
    
font = ('times', 16, 'bold')
title = Label(main, text='Helmet Detection using Single Shot Multibox Detection & YOLOV3 algorithm', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 14, 'bold')

model = Button(main, text="Generate & Load SSMD & Yolo Model", command=loadModels)
model.place(x=200,y=100)
model.config(font=font1)

uploadButton = Button(main, text="Upload Image", command=upload)
uploadButton.place(x=200,y=150)
uploadButton.config(font=font1)  

yoloButton = Button(main, text="Detect Helmet using YoloV3", command=yoloDetection)
yoloButton.place(x=200,y=200)
yoloButton.config(font=font1) 

ssdButton = Button(main, text="Detect Helmet using SSMD", command=ssdDetection)
ssdButton.place(x=200,y=250)
ssdButton.config(font=font1)

graphButton = Button(main, text="Loss Comparison Graph", command=graph)
graphButton.place(x=200,y=300)
graphButton.config(font=font1) 

exitapp = Button(main, text="Accuracy Comparison Graph", command=accuracyGraph)
exitapp.place(x=450,y=300)
exitapp.config(font=font1)

font1 = ('times', 12, 'bold')
textarea=Text(main,height=15,width=60)
scroll=Scrollbar(textarea)
textarea.configure(yscrollcommand=scroll.set)
textarea.place(x=10,y=350)
textarea.config(font=font1)

main.config(bg='light coral')
main.mainloop()
