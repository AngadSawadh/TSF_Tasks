from collections import Counter
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import utils
import identifierengine
import cv2 as cv

ap = argparse.ArgumentParser()
ap.add_argument("-i","--input",type=str,default="",help="path to the input file")
ap.add_argument("-o","--output",type=str,default="",help="path to the output file")
ap.add_argument("-k","--clusters",type=int,default=-1,help="enter the number of clusters if you don't want machine to decide it")
ap.add_argument("-d","--display",type=int,default=1,help="want to display the results(piechart)?")
args = vars(ap.parse_args())

img = utils.get_image(args["input"],filter=cv.COLOR_BGR2RGB)
img_data = utils.data_prep(img)

if (args["clusters"]==-1):
    inertia_metric = identifierengine.detectcolors(15,img_data)
    cluster_num = identifierengine.bestclusternumber(inertia_metric,1)

else:
    cluster_num = args["clusters"]

model = KMeans(n_clusters=cluster_num,init="k-means++")
labels = model.fit_predict(img_data)

counts = Counter(labels)
center_colors = model.cluster_centers_

ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [utils.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
rgb_colors = [ordered_colors[i] for i in counts.keys()]

plt.figure(figsize=(8,6))
plt.pie(counts.values(),labels=hex_colors,colors=hex_colors)

if (len(args["output"])>0):
    try:
        plt.savefig(args["output"],format="jpg")
    except:
        print("Enter a valid address for saving the output result.")

if(args["display"]==1):
    plt.show()


