import tensorflow as tf
import time
import numpy as np
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import HyperLPRLite as pr
import cv2
def Hello():
    for i in range(10):
        start=time.time()
        a = tf.constant(2)
        b = tf.constant(2)
        c = tf.add(a, b)
        with tf.Session() as sess:
            d=sess.run(c)
        end=time.time()
        t=(end-start)*1000
        print(t)
        e=np.zeros((1,100))
        print(e)
    return d
def Add(a, b):
    return a+b
if __name__=='__main__':
    Hello()
