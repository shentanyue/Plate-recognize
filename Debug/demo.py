from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import HyperLPRLite as pr
import cv2
import numpy as np
import os
import time

fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)

CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z',
         '港', '学', '使', '警', '澳', '挂', '军', '北', '南', '广',
         '沈', '兰', '成', '济', '海', '民', '航', '空',
         ]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

def SpeedTest(grr):
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    model.SimpleRecognizePlateByE2E(grr)
    #t0 = time.time()
    for x in range(2):
        model.SimpleRecognizePlateByE2E(grr)
    #t = (time.time() - t0)/2.0
    #print ('Image size:'+ str(grr.shape[1])+"x"+str(grr.shape[0])+"need"+str(round(t*1000,2))+"ms")

    



def drawRectBox(image,rect,addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2,cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0]+1), int(rect[1]-16)), addText.encode('utf-8').decode("utf-8"), (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex

def encodeLabel(s):
    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        label[i] = CHARS_DICT[c]
    return label

def parseLine(line):
    parts = line.split(':')
    label = encodeLabel(parts[0].strip().upper())
    return label




#grr = cv2.imread("images_rec/7_.jpg")
#for filename in os.listdir(r"./file"): 
model = pr.LPR("model/cascade.xml","model/model12.h5","model/ocr_plate_all_gru.h5")
def run_demo():
    for i in range(1):
        dir = './test_data_clear'         
        for rt, dirs, files in os.walk(dir):
            for filename in files:
            #print(filename)
            #grr=cv2.imread("./test_data/%s.jpg"%(filename))
                #print(filename)
                grr=cv2.imdecode(np.fromfile("./test_data_clear/%s"%(filename),dtype=np.uint8),cv2.IMREAD_COLOR)
    
                #grr=cv2.imread(filename)    
                start=time.time()
                for pstr,confidence,rect in model.SimpleRecognizePlateByE2E(grr):
                    if confidence>0:
                        image = drawRectBox(grr, rect, pstr+" "+str(round(confidence,3)))
                        end=(time.time()-start)*1000
                        print(rect)
                        print(pstr)
                        print(confidence)
                        print(end)
                        print('\n')

                        
                        #cv2.imshow('grr',grr)
                        #cv2.waitKey(0)








                    
#               
#sum1=np.sum(count)
#print(sum1)
#

                #print ("plate_str:")
                #print (pstr)
                #print ("plate_confidence")
                #print (confidence)
    #SpeedTest(grr)            
    #cv2.imshow("image",image)
    #cv2.waitKey(0)


 #with open('./test_data/label.txt') as f:
#                    for line in f:
#                        labels=parseLine(line)
#                        labels=np.float32(labels)
#                        print(labels)
#                        #label_path = './test_data/label.txt'
#                        #tem_label = np.loadtxt(label_path)
#                        #row, col = tem_label.shape
#                        for j in range(7):
#                            count=np.ones((100,1))
#                            if pstr[i][j] != labels[i][j]:
#                                count[i] = 0
#                                break


#with open(self._label_file) as f:
#            for line in f:
#                filename, label = parseLine(line)
#                self.filenames.append(filename)
#                self.labels.append(label)
#                self._num_examples += 1
#        self.labels = np.float32(self.labels)




