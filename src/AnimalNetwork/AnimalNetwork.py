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
        #Dense -> que todas las neuronas del layer anterior estan conectadas a todas las del layer actual
        #Flatten -> cque convierte data de 2D (como una foto de mxn dimensiones) en un arreglo 1D, osea q toda el valor de cada pixel y lo pone en un arreglo que puede usar de input para la red neuronal
        self.model = tf.keras.Sequential([
            #toma imagenes de 224x224 de 3 colores primos, y las pasa por 70 filtros de 3x3 aleatorios
            tf.keras.layers.Conv2D(70, (3,3), activation = 'relu', input_shape=(224,224,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(80, (3,3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(100, (3,3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(80, activation='softmax')
            ])
        
        #compilacion con funciones de perdida, optimizador y metrica de accuracy
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        #modificaciones a las imagenes
        modifiers = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)

        training = modifiers.flow_from_directory(
            self.train_dir,
            target_size = (224,224),
            batch_size = 32,
            class_mode = 'categorical'
        )

        testing = modifiers.flow_from_directory(
            self.test_dir,
            target_size = (224,224),
            batch_size = 32,
            class_mode = 'categorical'
        )
        history = self.model.fit(
            training,
            steps_per_epoch = training.batch_size,
            epochs = 30,
            validation_data = testing,
            validation_steps = testing.batch_size
        ) 
        return True

    def classify(self, dir):
        print("to be implemented")
    
