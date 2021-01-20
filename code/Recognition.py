'''
    Umpire detection and signal recognition
'''

import os
import cv2
from keras import applications
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
import numpy as np
import pickle
from keras.models import Model
import time
import shutil
import tensorflow as tf
from collections import deque
from google.colab.patches import cv2_imshow


class Infer:
    def __init__(self, params):
        self._seq_len = params['SEQ_LEN']
        self._img_width = params['IMG_WIDTH']
        self._img_height = params['IMG_HEIGHT']

        self._labels = params['LABELS']
        self._n_class = len(self._labels)
        self._checkpoint_path = params['CHECKPOINT_PATH']

        self._img_frames = deque(maxlen=self._seq_len)
        self._standby_frames = -1
        self._current_prediction = ""
        self._ump_in_buffer = False
        self._non_ump = 0
        #self._camera = cv2.VideoCapture(0)
        self._create_graph()

    def _create_graph(self):
        graph1 = tf.Graph()
        with graph1.as_default():
            session1 = tf.Session()
            with session1.as_default():
                self._sess = session1
                self._sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
                tf.saved_model.loader.load(self._sess, [tf.saved_model.tag_constants.SERVING], self._checkpoint_path)

                logits_op = self._sess.graph.get_tensor_by_name('logits:0')
                self._predictions_op = tf.argmax(logits_op, axis=1)
                self._data_X_op = self._sess.graph.get_tensor_by_name('data_X:0')

    def _process_image(self, img):
        img = cv2.resize(img, (self._img_width, self._img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Our sequence length (60) of images in one video is huge
        # To fill the buffer quickly, fill six images.

        if len(self._img_frames) < self._seq_len:
            self._img_frames.append(img)
            self._img_frames.append(img)
            self._img_frames.append(img)
            self._img_frames.append(img)
            self._img_frames.append(img)
            self._img_frames.append(img)
        else:
            self._img_frames.append(img)
            self._standby_frames += 1

        if len(self._img_frames) == self._seq_len and self._standby_frames in range(0, 10):
            print(self._current_prediction)
            return self._current_prediction
        elif len(self._img_frames) == self._seq_len:
            self._standby_frames = 0
            print('Buffer full now')
            img_array = np.array(self._img_frames, dtype=np.float32)
            img_array /= 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = self._sess.run(self._predictions_op, feed_dict={self._data_X_op: img_array})[0]
            prediction = self._labels[prediction]
            self._current_prediction = prediction
            print(prediction)
            return str(prediction)

        return None

    def infer_video(self):
        font = cv2.FONT_HERSHEY_SIMPLEX

        video = input('Enter video directory: ')
        cap = cv2.VideoCapture(video)
        print('Reading video..')

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('/content/Umpire/output.avi', fourcc, fps, size)

        j = 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret != True:
                break

            img1 = cv2.resize(frame, (224, 224))
            x = image.img_to_array(img1)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Feature Extraction Step
            features = model.predict(x)  #VGG19 Model

            # Classification Step
            predicted_values = loaded_model1.predict(features.reshape(1, -1))
            print('Frame', j)
            print(predicted_values)
            prediction_score = loaded_model1.decision_function(features.reshape(1, -1))
            print(prediction_score)

            if prediction_score >= 0.2:
                print('Umpire')
                info = self._process_image(frame)
                print(info)
                if info is not None:
                    cv2.putText(frame, info, (20, 100), font, 3, (255, 255, 255), 4, cv2.LINE_AA)

                self._ump_in_buffer = True
                self._non_ump = 0
                
            elif self._ump_in_buffer is True:
                self._non_ump+=1

            if self._non_ump >=5:
                self._img_frames.clear()
				self._standby_frames = -1
                print('Buffer empty')
                self._ump_in_buffer = False
                self._non_ump = 0


            cv2.imwrite(f"{dir}/{j}.jpg", frame)
            j += 1
            out.write(frame)
        
        out.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    params = {
        'SEQ_LEN': 60,
        'IMG_WIDTH': 224,
        'IMG_HEIGHT': 224,

        # List all the labels you have trained your model on. Do not change the order.
        'LABELS': ['Four', 'NoBall', 'Out',
                   'Six', 'Wide', 'NoSignal'],

        'CHECKPOINT_PATH': os.path.abspath('/content/Umpire/code/vgg19/model2_pb')
    }

    start_time = time.time()

    with tf.variable_scope('Keras'):
        base_model = applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                                              pooling=None, classes=1000)
        model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)

        # Project Model- loading the savedFER_vgg19fc2_model2net_ck_transfer_only_svm
        loaded_model1 = pickle.load(
            open("/content/Umpire/code/vgg19/FER_vgg19fc1_model1net_ck_transfer_only_svm.sav", "rb"))

    # Setting directory for output frames
    dir = "/content/Umpire/Output Frames"
    shutil.rmtree(dir, ignore_errors=True)
    os.mkdir(dir)

    i = Infer(params)
    i.infer_video()
