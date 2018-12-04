from keras import backend as K
from keras.models import *
from keras.layers import *
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import os
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

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

chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
             u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
             u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
             u"Y", u"Z",u"港",u"学",u"使",u"警",u"澳",u"挂",u"军",u"北",u"南",u"广",u"沈",u"兰",u"成",u"济",u"海",u"民",u"航",u"空"
             ]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

######################################
# TODO: set the gpu memory using fraction #
#####################################
def get_session(gpu_fraction=0.3):
    """
    This function is to allocate GPU memory a specific fraction
    Assume that you have 6GB of GPU memory and want to allocate ~2GB
    """

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session(0.6))  # using 60% of total GPU Memory
os.system("nvidia-smi")  # Execute the command (a string) in a subshell

class recognition:
    def __init__(self, model_path):
        #model_path = "model/ocr_plate_all_gru.h5"
        self.modelSeqRec = self.model_seq_rec(model_path)

    def fastdecode(self, y_pred):
        results = ""
        confidence = 0.0
        table_pred = y_pred.reshape(-1, len(chars)+1)
        res = table_pred.argmax(axis=1)
        for i,one in enumerate(res):
            if one<len(chars) and (i==0 or (one!=res[i-1])):
                results+= chars[one]
                confidence+=table_pred[i][one]
        confidence/= len(results)
        return results, confidence

    def model_seq_rec(self, model_path):
        width, height, n_len, n_class = 164, 48, 7, len(chars)+ 1
        rnn_size = 256
        input_tensor = Input((164, 48, 3))
        x = input_tensor
        base_conv = 32
        for i in range(3):
            x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        conv_shape = x.get_shape()
        x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
        x = Dense(32)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
        gru1_merged = add([gru_1, gru_1b])
        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
        x = concatenate([gru_2, gru_2b])
        x = Dropout(0.25)(x)
        x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
        base_model = Model(inputs=input_tensor, outputs=x)
        base_model.load_weights(model_path)
        return base_model

    def recognizeOne(self, src):
        x_tempx = src
        x_temp = cv2.resize(x_tempx, (164, 48))
        x_temp = x_temp.transpose(1, 0, 2)
        y_pred = self.modelSeqRec.predict(np.array([x_temp]))
        y_pred = y_pred[:, 2:, :]
        return self.fastdecode(y_pred)

    def SimpleRecognizePlateByE2E(self, image, rect_refine):
        res_set = []
        w = rect_refine[3]
        h = rect_refine[2]
        image = image[0:w, 0:h]
        res, confidence = self.recognizeOne(image)
        res_set.append([res, confidence])
        return res_set

    def drawRectBox(self, image, rect, addText):
        cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0,0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(image, (int(rect[0]-1), int(rect[1])-16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                      cv2.LINE_AA)
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)
        draw.text((int(rect[0]+1), int(rect[1]-16)), addText.encode('utf-8').decode("utf-8"), (255, 255, 255), font=fontC)
        imagex = np.array(img)
        return imagex

    def Accuracy(self, filename, pstr):
        length1 = len(filename)-4
        length2 = len(pstr)
        if length1 != length2:
            return 0
        sum = 0
        for i in range(length1):
            if filename[i] == pstr[i]:
                sum += 1
        result = sum/length1
        return result

    def arrayreset(array):
        # for i inrange(array.shape[1]/3):
        #     pass
        a = array[:, 0:len(array[0] - 2):3]
        b = array[:, 1:len(array[0] - 2):3]
        c = array[:, 2:len(array[0] - 2):3]
        a = a[:, :, None]
        b = b[:, :, None]
        c = c[:, :, None]
        m = np.concatenate((a, b, c), axis=2)
        return m

    def run_demo(self,filename,rect_refine):
        # dir = './test_data_clear'
        # for rt, dirs, files in os.walk(dir):
        #     for filename in files:
        #filename = "/home/shenty/eclipse-workspace/test/src/test_data_clear/京SHM8QX.jpg"
        #rect_refine = [0, 0, 220, 72]
        #grr = arrayreset(image)
        #grr = cv2.imdecode(np.fromfile("%s" %(filename), dtype=np.uint8), cv2.IMREAD_COLOR)
        grr = cv2.imdecode(np.fromfile("./Generating_data/plate/%s" % (filename), dtype=np.uint8), cv2.IMREAD_COLOR)
        # start = time.time()
        for pstr, confidence in self.SimpleRecognizePlateByE2E(grr, rect_refine):
            if confidence > 0:
                grr = self.drawRectBox(grr, rect_refine, pstr+" "+str(round(confidence,3)))
                #end = (time.time()-start)*1000
                accuracy = self.Accuracy(filename, pstr)
                print(pstr)
                print(confidence)
                print(accuracy)
                # print(end)
                print('\n')
        return accuracy;

if __name__=='__main__':
    rect=[0,0,272,72]
    model=recognition("model/ocr_plate_all_gru.h5")
    dir = './Generating_data/plate'
    i = 0
    p = 0
    for rt, dirs, files in os.walk(dir):
        for filename in files:
            a = model.run_demo(filename, rect)
            p += 1
            if a == 1:
                i += 1
    sum = i/p
    print(sum)


