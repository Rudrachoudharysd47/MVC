import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image
import PIL.ImageOps
x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')["labels"]
print(pd.Series(y).value_counts)
classes = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","v","W","X","Y","Z"]
nclasses = len(classes)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state = 9,train_size = 7500 , test_size = 2500)
xtrainscaled = xtrain/255
xtestscaled = xtest/255
clf = LogisticRegression(solver = "saga",multi_class = "multinomial").fit(xtrainscaled,ytrain)
def get_prediction(image):
        image_PIL = Image.open(image)
        image_bw = image_PIL.convert("L")
        image_bw_resize = image_bw.resize((28,28),Image.ANTIALIAS)
        pixelfilter = 20
        minpixel = np.percentile(image_bw_resize,pixelfilter)
        image_bw_resize_inverted_scaled = np.clip(image_bw_resize - minpixel , 0 , 255)
        max_pixel = np.max(image_bw_resize)
        image_bw_resize_inverted_scaled = np.asarray(image_bw_resize_inverted_scaled)/max_pixel
        testsample = np.asarray(image_bw_resize_inverted_scaled).reshape(1,784)
        testprediction = clf.predict(testsample)
        return testprediction[0]



