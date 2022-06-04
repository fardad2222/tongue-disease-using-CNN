```python
import tensorflow as tf
import matplotlib.pyplot as plt
#import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.utils import to_categorical

#from keras.utils import to_categorical
from keras.preprocessing import image

import pandas as pd

from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical
from tqdm import tqdm
```

img = image.load_img("C:/Users/Fardad/Desktop/dataa/basedata/train/Healthy/12876P.JPG")
plt.imshow(img)


```python
train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)
```

train_dataset = train.flow_from_directory("C:/Users/Fardad/Desktop/dataa/basedata/train/",
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = "binary")

validation_dataset = train.flow_from_directory("C:/Users/Fardad/Desktop/dataa/basedata/validation/",
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = "binary")

train_dataset.class_indices

train_dataset.classes


```python
model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape =(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512,activation="relu"),
                                    ##
                                    tf.keras.layers.Dense(1,activation="sigmoid")
                                    ])
```

model.compile(loss = "binary_crossentropy",
             optimizer = tf.keras.optimizers.RMSprop(lr=0.001), 
             metrics=["acc"])

modell_fit = model.fit(train_dataset,
                     steps_per_epoch = 3,
                     epochs = 30,
                     validation_data = validation_dataset)

dir_path = "C:/Users/Fardad/Desktop/tongue diseases"

for i in os.listdir(dir_path):
    img = image.load_img(dir_path+'//'+ i, target_size=(200,200))
    plt.imshow(img)
    plt.show()
    
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    val = model.predict(images)
    if val == 0:
        print("you are healthy")
    else:
        print("you are unhealthy")
    


```python

```
