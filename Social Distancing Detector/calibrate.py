import cv2 as cv
import numpy as np

# cv.circle(img,(int(0.22*W),int(0.44*H)),2,(0,0,255),-1)
# cv.circle(img,(int(0.36*W),int(0.42*H)),2,(0,0,255),-1)
# cv.circle(img,(int(W),int(0.8*H)),2,(0,0,255),-1)
# cv.circle(img,(int(0.7*W),int(H)),2,(0,0,255),-1)

# pts1 = np.float32([[int(0.22*W),int(0.44*H)],[int(0.36*W),int(0.42*H)],[int(W),int(0.8*H)],[int(0.7*W),int(H)]])
# pts2 = np.float32([[0,0],[0.3*W,0],[0.3*W,H],[0,H]])

# matrix = cv.getPerspectiveTransform(pts1,pts2)
# result = cv.warpPerspective(img,matrix,(int(0.3*W),H))

# cv.imshow("Image",img)
# cv.imshow("Transformed",result)
# cv.waitKey(0)

def calibrator(imgpath):
    cal_img = cv.imread(imgpath)
    (H,W) = cal_img.shape[:2]
    
    top_view_rect_points = [[0,0],
                            [0.3*W,0],
                            [0.3*W,H],
                            [0,H]]
    
    rect_points = [[int(0.22*W),int(0.44*H)],
                   [int(0.36*W),int(0.42*H)],
                   [int(W),int(0.8*H)],
                   [int(0.7*W),int(H)]]
    
    top_view = np.zeros((H,int(0.3*W),3),dtype=np.uint8)

    return top_view_rect_points,rect_points,top_view


def eagle_eye(top_view_rect_points,rect_points):
    pts1 = np.float32(rect_points)
    pts2 = np.float32(top_view_rect_points)
    matrix = cv.getPerspectiveTransform(pts1,pts2)
    return matrix

def top_plotter(c1,c2,matrix):
    px = int((matrix[0][0]*c1 + matrix[0][1]*c2 + matrix[0][2]) / ((matrix[2][0]*c1 + matrix[2][1]*c2 + matrix[2][2])))
    py = int((matrix[1][0]*c1 + matrix[1][1]*c2 + matrix[1][2]) / ((matrix[2][0]*c1 + matrix[2][1]*c2 + matrix[2][2]))) 
    return px,py


