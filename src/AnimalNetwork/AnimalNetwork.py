import tensorflow as tf
import os

class AnimalNetwork:
    def __init__(self):
        
        #enlaces de dataset para entrenar y probar
        dir =  '/kaggle/input/animals-detection-images-dataset/'
        self.train_dir = os.path.join(dir, 'train')
        self.test_dir = os.path.join(dir, 'test')

    def loadNetwork(self):
        return False

    def createNetwork(self):
        capa = tf.keras.layers.Dense(units=1, input_shape=[1])
        
        return True