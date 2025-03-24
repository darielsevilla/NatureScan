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
        self.train_dir = 'C:/Users/darie/Downloads/raw-img/'
        #self.train_dir = os.path.join(dir, 'train')
        #self.test_dir = os.path.join(dir, 'test')

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
        base_dir = Path(os.getcwd())  
        save_path = base_dir / "SavedFiles" / "model.h5"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        #enlaces para datasets que no me mata la memoria de la compu
        #dataset de entrenamiento
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
        )

        ##test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

        # Loading the data
        self.dataset = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(64, 64), 
            class_mode='categorical'  
        )
  

        def augment(image, label):
            image = tf.image.resize(image, [128, 128])  # Keep high res initially
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.05)
            
            image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
            image = tf.image.adjust_contrast(image, contrast_factor=2.0)
            image = tf.image.resize(image, [64, 64])  

                
            return image, label

        #self.dataset = self.dataset.map(augment)
        
        #Dense -> que todas las neuronas del layer anterior estan conectadas a todas las del layer actual
        #Flatten -> cque convierte data de 2D (como una foto de mxn dimensiones) en un arreglo 1D, osea q toda el valor de cada pixel y lo pone en un arreglo que puede usar de input para la red neuronal
        self.model = tf.keras.Sequential([
          
            tf.keras.layers.Conv2D(32, (3,3),padding = 'same',  activation=tf.keras.layers.LeakyReLU(alpha=0.1), input_shape=(64,64,3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3,3),padding = 'same',  activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D(64, (3,3),  activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding = 'same'),
            tf.keras.layers.BatchNormalization(),
             tf.keras.layers.Conv2D(64, (3,3),  activation=tf.keras.layers.LeakyReLU(alpha=0.1), padding = 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.Conv2D(128, (3,3),  activation=tf.keras.layers.LeakyReLU(alpha=0.1),padding = 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3,3),  activation=tf.keras.layers.LeakyReLU(alpha=0.1),padding = 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),
            
            tf.keras.layers.Conv2D(256, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.1),padding = 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(256, (3,3), activation=tf.keras.layers.LeakyReLU(alpha=0.1),padding = 'same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2,2),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(512, activation=tf.nn.swish),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation=tf.nn.swish),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation=tf.nn.swish),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation='softmax')  
          
            ])
        
        
        #compilacion con funciones de perdida, optimizador y metrica de accuracy
       
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0003), metrics=['accuracy'])
        
        #modificaciones a las imagenes
        #modifiers = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)
        
        history = self.model.fit(
            self.dataset,
            epochs = 30, 
            #steps_per_epoch=5000,     
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1)]
        )

        #guardado de la red
        self.model.save(save_path)
        print("Network saved at: {save_path} ")
        return True
    
    

    def uploadImage(self):
        flag = True
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        )

        while flag:
            root = tk.Tk()
            root.withdraw() #esconde la ventana de Tkinter
            root.attributes('-topmost', True) #asegura que la ventana del File Chooser esté adelante
        
            img_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.PNG;*.JPG;*.JPEG")]
            )

            if img_path:
                print("path")
                print(img_path)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64,64,3)) 
                print("img")
                print(img)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                #img_array = tf.image.convert_image_dtype(img_array, tf.float32)  
                img_array = tf.cast(img_array, tf.uint8)
    
    # Normalize the image to [0, 1]
                img_array = np.expand_dims(img_array, axis=0)  
                                

                img_array = img_array / 255.0 # Normaliza la imagen (de 0 a 1) para hacer la predicción
                print("img_array")   
                print(img_array)
                predictions = self.model.predict(img_array)  # hace la predicción
                print(predictions)
                predicted_class = np.argmax(predictions, axis=1) # obtiene el índice de la clase de la predicción
                print(f"Prediction: {predicted_class}")

                # definimos el nombre de los animales 
                #class_labels = ["Cat", "Lynx", "Wolf", "Coyote", "Cheetah", "Jaguar", 
                #                "Chimpanzee", "Orangutan", "Hamster", "Guinea pig"]
                class_labels = ["Perro", "Caballo", "Elefante", "Mariposa", "Gallina", "Gato", 
                                "Vaca", "Oveja", "Araña", "Ardilla"]
                print(f"Predicted Animal: {class_labels[predicted_class[0]]}")
                print("-" * 50)
            else:
                print("No image selected.")

            response = messagebox.askyesno("Confirmation", "Do you want to try again?")

            if response:
                flag = True
            else:
                flag = False


        