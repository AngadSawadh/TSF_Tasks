import numpy as np
import cv2 as cv
from .params import MIN_CONF
from .params import NMS_THRESH

def detectpeople(frame, net, ln, personIdx=0):

    (H,W) = frame.shape[:2]
    final_results = []
    #it takes a frame(s)/image(s) and converts it into a four dimensional ndarray with order (N,C,H,W)
    #it divides each pixel value by 255 to normalize the image
    #it resizes all the input image(s) to 3*608*608
    #it changes channel order from BGR to RGB
    blob = cv.dnn.blobFromImage(frame,scalefactor=1/255,size=(416,416),swapRB=True,crop=False)

    #net is an object of class dnn.Net
    # here we set the input for the neural network 
    net.setInput(blob)
    #we need to take the output of YOLO output layer ln
    layerOutput = net.forward(ln) 

    #preparing outputs
    #layerOutput will be of shape (N,3,52,52,4+1+num_classes) or (N,3,26,26,4+1+num_classes) or (N,3,13,13,4+1+num_classes)
    bboxes = []
    centroids = []
    confidences = []

    for outputs in layerOutput:
        for detections in outputs:
            score_class = detections[5:]
            class_id = np.argmax(score_class)
            confidence = score_class[class_id]

            if class_id == personIdx and confidence > MIN_CONF:
                box = detections[:4]*np.array([W,H,W,H])
                (x_center,y_center,width,height) = box.astype("int")

                x_left = int(x_center - (width/2))
                y_left = int(y_center - (height/2))

                bboxes.append([x_left,y_left,int(width),int(height)])

                centroids.append([x_center,y_center])

                confidences.append(float(confidence))

    idx = cv.dnn.NMSBoxes(bboxes,confidences,MIN_CONF,NMS_THRESH)    

    if len(idx)>0:
        for i in idx.flatten():
            (x,y) = (bboxes[i][0],bboxes[i][1])
            (w,h) = (bboxes[i][2],bboxes[i][3])
            r = [confidences[i],(x,y,x+w,y+h),centroids[i]]
            final_results.append(r)

    return final_results

