from utils.detector import detectpeople
import numpy as np
import argparse
import cv2 as cv
import os
from scipy.spatial import distance as dist
from utils.params import MODEL_DIR
from utils.params import USE_GPU
from utils.params import MIN_DISTANCE
from calibrate import eagle_eye
from calibrate import top_plotter
from calibrate import calibrator


ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",type=str,default="",help="path to input file")
ap.add_argument("-o","--output",type=str,default="",help="path to output file")
ap.add_argument("-d","--display",type=int,default=1,help="want to display the video?")
args = vars(ap.parse_args())

model_path = os.path.join(MODEL_DIR,"yolov3.weights")
config_path = os.path.join(MODEL_DIR,"yolov3.cfg")
class_names_path = os.path.join(MODEL_DIR,"coco.names")

LABELS = open(class_names_path).read().strip().split("\n")
top_points,rect_points,top_view_img = calibrator(r"G:\TSF_Tasks\Social Distancing Detector\calibratecam.jpg")
matrix = eagle_eye(top_points,rect_points)


print("[INFO] loading YOLO parameters...")
net = cv.dnn.readNetFromDarknet(config_path, model_path)

# if USE_GPU:
#     print("[INFO] setting preferable backend and target to CUDA...")
#     net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
#     net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

print("[INFO] Processing the input...")
vs = cv.VideoCapture(args["input"] if args["input"] else 0)
writer = None
writer_top = None

while True:
    (success,frame) = vs.read()

    if not success:
        break

    (H,W) = frame.shape[:2]
    top_view = np.copy(top_view_img)

    results = detectpeople(frame,net,ln,personIdx=LABELS.index("person"))
    #results --> [[confidence,(bbox),[centroids]],....Ndetections]

    violate = set()

    if len(results)>1:
        centroids = np.array([result[2] for result in results])
        distances = dist.cdist(centroids,centroids,metric="euclidean")

        for i in range(distances.shape[0]):
            for j in range(i+1,distances.shape[1]):
                if distances[i][j]<MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)

    for (i, (conf, bbox, centroid)) in enumerate(results):
        (x1,y1,x2,y2) = bbox
        (c1,c2) = centroid
        px,py = top_plotter(c1,c2,matrix)
        color = (0,255,0)

        if i in violate:
            color = (0,0,255)

        cv.rectangle(frame,(x1,y1),(x2,y2),color,2)   
        cv.circle(frame,(c1,c2),3,color,1)
        cv.circle(top_view,(px,py),5,color,-1)

    
    text = "Total Violations:{}".format(len(violate))
    cv.putText(frame,text,(0,70),cv.FONT_HERSHEY_SIMPLEX,0.85,(255,0,0),2)

    if args["display"]>0:
        cv.imshow("Video",top_view)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    if args["output"] != "" and writer is None:
        fourcc = cv.VideoWriter_fourcc(*"MJPG")
        writer = cv.VideoWriter(args["output"], fourcc, 25,(frame.shape[1], frame.shape[0]), True)
        writer_top = cv.VideoWriter(r"G:\TSF_Tasks\Social Distancing Detector\final_output_top.avi", fourcc, 25,(top_view.shape[1],top_view.shape[0]), True)

    if writer is not None:
        writer.write(frame)
        writer_top.write(top_view)
