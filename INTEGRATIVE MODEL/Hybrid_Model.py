

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate
from sklearn.preprocessing import LabelEncoder
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import scale
from keras.models import Sequential
from sklearn import preprocessing
from tensorflow.keras import Model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import models
import pandas as pd
import numpy as np



################################################################################


from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense
from keras.models import Model


################################################################################


def CNN():
    model = Sequential()

    model.add(Conv2D(128, (2, 2), activation='relu', input_shape=(224,224,1)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(16, activation='sigmoid'))

    return(model)


def MobileNet():
    
    mobileNet = tf.keras.applications.mobilenet.MobileNet()
    x = mobileNet.layers[-3].output
    model.add(Dense(units=64, activation='softmax'))(x)

    
def MLP(X_train):

    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1], )))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(16, activation='relu'))

    return(model)


input_shape = (224, 224)
batch_size = 8

train_ds = image_dataset_from_directory(
    
    "/directory/training",
    label_mode="binary",
    shuffle=True,
    subset=None,
    image_size=input_shape,
    batch_size=batch_size)

test_ds = image_dataset_from_directory(
    
    "/directory/testing",
    label_mode="binary",
    shuffle=True,
    subset=None,
    image_size=input_shape,
    batch_size=batch_size)


def process(path, label):

    df = pd.read_csv(path, low_memory=False, index_col=0)
    norm = np.linalg.norm(df)
    normal_array = df/norm
    df = normal_array.transpose()
    df['Y'] = label

    return(df)

def processed():

    R = process('/directory/PI/csv', 'R')
    S = process('/directory/PI.csv', 'S')
    df_all = R.append(pd.DataFrame(data = S), ignore_index=True)

    return(df_all)


df_PI = processed()
df_PI = shuffle(df_PI)

encoder = LabelEncoder()
df_counts['Y'] = encoder.fit_transform(df_counts['Y'])

x = df_PI.drop(['Y'], axis=1)
y = df_PI['Y']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


mlp_model = MLP(X_train)
mobileNet_model = MobileNet()
#cnn_model = CNN()

combinedInput = concatenate([mlp_model.output, MobileNet.output])

x_layer = Dense(9, activation="relu")(combinedInput)
x_layer = Dense(1, activation="sigmoid")(x_layer)

model = Model(inputs=[mlp_model.input, cnn_model.input], outputs=x_layer)
plot_model(model, to_file='demo.pdf', show_shapes=True)


model.compile(optimizer=keras.optimizers.Adam(1e-3),
                loss="binary_crossentropy",
                metrics=["accuracy"])

model.fit([X_train, train_np], y_train.to_numpy(), epochs=500, batch_size=8)

unseen = model.evaluate([X_test, val_ds], y_test)
seen = model.evaluate([X_train, train_ds], y_train)