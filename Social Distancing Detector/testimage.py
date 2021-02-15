import cv2 as cv
import numpy as np
from utils.detector import detectpeople
from utils.params import MIN_DISTANCE
from utils.params import MODEL_DIR
from scipy.spatial import distance as dist
import os

img = cv.imread(r"G:\TSF_Tasks\Social Distancing Detector\calibratecam.jpg")
(H,W) = img.shape[:2]

pts1 = np.float32([[int(0.22*W),int(0.44*H)],[int(0.36*W),int(0.42*H)],[int(W),int(0.8*H)],[int(0.7*W),int(H)]])
pts2 = np.float32([[0,0],[0.3*W,0],[0.3*W,H],[0,H]])
matrix = cv.getPerspectiveTransform(pts1,pts2)

top_view = np.zeros((H,int(0.3*W),3))
cv.imshow("Blank",top_view)

cv.imshow("Image",img)

model_path = os.path.join(MODEL_DIR,"yolov3.weights")
config_path = os.path.join(MODEL_DIR,"yolov3.cfg")
class_names_path = os.path.join(MODEL_DIR,"coco.names")

LABELS = open(class_names_path).read().strip().split("\n")

print("[INFO] loading YOLO parameters...")
net = cv.dnn.readNetFromDarknet(config_path, model_path)

# if USE_GPU:
#     print("[INFO] setting preferable backend and target to CUDA...")
#     net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
#     net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

violate = set()

results = detectpeople(img,net,ln,personIdx=LABELS.index("person"))
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
    color = (0,255,0)

    if i in violate:
        color = (0,0,255)

    cv.rectangle(img,(x1,y1),(x2,y2),color,2)   
    cv.circle(img,(c1,c2),3,color,1) 
    px = int((matrix[0][0]*c1 + matrix[0][1]*c2 + matrix[0][2]) / ((matrix[2][0]*c1 + matrix[2][1]*c2 + matrix[2][2])))
    py = int((matrix[1][0]*c1 + matrix[1][1]*c2 + matrix[1][2]) / ((matrix[2][0]*c1 + matrix[2][1]*c2 + matrix[2][2])))  
    cv.circle(top_view,(px,py),3,color,-1)

    
text = "Total Violations:{}".format(len(violate))
cv.putText(img,text,(0,0),cv.FONT_HERSHEY_SIMPLEX,0.85,(255,0,0),2)

cv.imshow("Image",img)
cv.imshow("Top_view",top_view)
cv.waitKey(0)