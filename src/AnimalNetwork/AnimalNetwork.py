import tensorflow as tf
import numpy as np
import os
import deeplake
import tkinter as tk
from pathlib import Path
import sys
from tensorflow.keras.layers import LeakyReLU
from tkinter import filedialog
from tkinter import messagebox

class AnimalNetwork:
    def __init__(self):
        
        #enlaces de dataset para entrenar y probar
        dir = '/kaggle/input/animals-detection-images-dataset/'
        self.train_dir = os.path.join(dir, 'train')
        self.test_dir = os.path.join(dir, 'test')

        #enlaces para datasets que no me mata la memoria de la compu
        ds = deeplake.load("hub://activeloop/animal10n-test")
        
        #dataset de entrenamiento
        self.dataset = ds.tensorflow().map(lambda sample: (
                tf.image.resize(tf.cast(tf.expand_dims(sample['images'], axis=0) , tf.uint8), (64, 64)) / 255.0,  
                tf.one_hot(tf.cast(sample['labels'], tf.int32), depth=10)
        ))     

        print("Dataset: ")
        first_element = next(iter(self.dataset.take(1)))
        image, label = first_element
        print("Image:")
        print(image)
        print("Label:")
        print(label)
        #dataset de prueba
        ts = deeplake.load("hub://activeloop/animal10n-test")
        self.testset = ts.tensorflow().map(lambda sample: (
                tf.image.resize(tf.cast(tf.expand_dims(sample['images'], axis=0) , tf.uint8), (64, 64)) / 255.0,  
                tf.one_hot(tf.cast(sample['labels'], tf.int32), depth=10)    
        ))
        print("Test Dataset: ")
        print(self.testset)

    def loadNetwork(self): #le deseo la muerte a esta libreria basura que nunca agarra donde quiero
        base_dir = Path(os.getcwd()) 
        save_path = base_dir / "SavedFiles" / "model.h5"

        print(f"Looking for model at: {save_path}")

        if not save_path.exists():
            print(f"Model file not found at {save_path}, creating new network...")
            self.createNetwork()
            return False

        try:
            self.model = tf.keras.models.load_model(save_path, 
            custom_objects={
                "LeakyReLU": LeakyReLU,
                "swish": tf.nn.swish
            })            
            print("Network loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading network: {e}")
            print("Creating new network...")
            self.createNetwork()
            return False

    def createNetwork(self):
        
        #Dense -> que todas las neuronas del layer anterior estan conectadas a todas las del layer actual
        #Flatten -> cque convierte data de 2D (como una foto de mxn dimensiones) en un arreglo 1D, osea q toda el valor de cada pixel y lo pone en un arreglo que puede usar de input para la red neuronal
        self.model = tf.keras.Sequential([
          
            tf.keras.layers.Conv2D(32, (3,3),padding = 'same',  activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(64,64,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3),  activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding = 'same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(128, (3,3),  activation=tf.keras.layers.LeakyReLU(alpha=0.1),padding = 'same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(512, (3,3), activation=tf.nn.swish),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.SpatialDropout2D(0.2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation=tf.nn.swish),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation=tf.nn.swish),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation=tf.nn.swish),
          
            #tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
            ])
        
        #compilacion con funciones de perdida, optimizador y metrica de accuracy
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'])
        
        #modificaciones a las imagenes
        #modifiers = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
        
        history = self.model.fit(
            self.dataset,
            epochs = 40,      
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1)]
        )

        #guardado de la red
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_path = os.path.join(base_dir, "SavedFiles", "mi_modelo.h5")
        self.model.save(save_path)
        print("Network saved")
        return True
    
    def classifyTestDeeplake(self):
        # Get a batch of test images and labels
        for x_batch, y_batch in self.testset.take(10):  # Taking only 5 samples for testing
         
            # Normalize image
            x_batch = x_batch

            # Make predictions
            print(x_batch)
            print(y_batch)
            predictions = self.model.predict(x_batch)
            predicted_class = np.argmax(predictions, axis=1)

            print(f"Predicted Animal: {predicted_class[0]}")
            print("-" * 50)

    def uploadImage(self):
        flag = True

        while flag:
            root = tk.Tk()
            root.withdraw() #esconde la ventana de Tkinter
            root.attributes('-topmost', True) #asegura que la ventana del File Chooser esté adelante
        
            img_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.PNG;*.JPG;*.JPEG")]
            )

            if img_path:
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64, 3))  # Prepocesamiento de Imagen - Carga la imagen y la redimensiona
                img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convierte la imagen a un array       
                img_array = np.expand_dims(img_array, axis=0)  # expande la dimensión del batch (de 3D a 4D)
                img_array = img_array / 255.0 # Normaliza la imagen (de 0 a 1) para hacer la predicción
        
                predictions = self.model.predict(img_array)  # hace la predicción
                predicted_class = np.argmax(predictions, axis=1) # obtiene el índice de la clase de la predicción
                print(f"Prediction: {predicted_class}")

                # definimos el nombre de los animales 
                class_labels = ["Cat", "Lynx", "Wolf", "Coyote", "Cheetah", "Jaguar", 
                                "Chimpanzee", "Orangutan", "Hamster", "Guinea pig"]
        
                print(f"Predicted Animal: {class_labels[predicted_class[0]]}")
                print("-" * 50)
            else:
                print("No image selected.")

            response = messagebox.askyesno("Confirmation", "Do you want to try again?")

            if response:
                flag = True
            else:
                flag = False


        