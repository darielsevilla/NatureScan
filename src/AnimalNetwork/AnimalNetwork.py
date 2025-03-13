import tensorflow as tf
import numpy as np
import os
import deeplake

class AnimalNetwork:
    def __init__(self):
        
        #enlaces de dataset para entrenar y probar
        dir =  '/kaggle/input/animals-detection-images-dataset/'
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
    def loadNetwork(self):
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
            tf.keras.layers.Conv2D(256, (3,3), activation=tf.nn.swish),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(300, activation=tf.nn.swish),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(300, activation=tf.nn.swish),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(500, activation=tf.nn.swish),
            #tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
            ])
        
        #compilacion con funciones de perdida, optimizador y metrica de accuracy
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        
        #modificaciones a las imagenes
        #modifiers = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255)

        #training = modifiers.flow_from_directory(
        #    self.dataset,
        #    target_size = (224,224),
        #    batch_size = 32,
        #    class_mode = 'categorical'
        #)

        #testing = modifiers.flow_from_directory(
        #    self.test_dir,
        #    target_size = (224,224),
        #    batch_size = 32,
        #    class_mode = 'categorical'
        #)
        history = self.model.fit(
            self.dataset,
            epochs = 35,
            callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1)]
        ) 
        return True

    
    
    def classifyTest(self):
        image_paths = []
        for root, dirs, files in os.walk(self.test_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')): 
                    image_paths.append(os.path.join(root, file))
        
        image_paths = image_paths[:5]

        for img_path in image_paths:
            
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))  # Load the image
            img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert image to array
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize image
            
            # Make a prediction
            predictions = self.model.predict(img_array)
            
            # Get the predicted class
            predicted_class = np.argmax(predictions, axis=1)
            
            # Retrieve the class labels from the model
            class_labels = list(self.model.class_indices.keys())  # Retrieve the class labels
            
            # Print the image path and predicted class label
            print(f"Image: {img_path}")
            print(f"Predicted class: {class_labels[predicted_class[0]]}")
            print("-" * 50)
    
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

            print(f"Predicted Class: {predicted_class[0]}")
            print("-" * 50)