# -*- coding: utf-8 -*-
"""
Created on Thu May 18 20:37:08 2023

@author: Georg
"""

"""
Quellen für CNNs:

https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/




Quelldaten:
https://www.cs.toronto.edu/~kriz/cifar.html


Zum Einlesen wäre auch folgendes möglich:
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)



Im CIFAR Datensatz sind 50000 Farbbilder im Format 32x32 zum Trainieren der
Daten enthalten, dazu noch 10000 Farbbilder zum Trainieren.
Der Trainingnsdatensatz enthält jeweils genau 60000 Bilder aus den Kategoriern
0 airplane
1 automobile
2 bird
3 cat
4 deer
5 dog
6 frog
7 horse
8 ship
9 truck

Die Daten sind in Dateien unterteilt. In jeder Datei ist ein Dictionary
enthalten. Alle Strings darin sind binär, auch die Keys. Dies ist beim Zugriff
zu beachten.

Dieser Trainignsdatensatz ist in 5 Dateien aufgeteilt, die jeweils 5000 Bilder
enthalten - annähernd gleichverteilt auf die Kategorien.

Der Testdatensatz beinhaltet genau 1000 Bilder von jeder Kategorie.

Format des Dictionarys der Trainings- und Test-Daten
Key             Daten
data            Liste aus 10000 Einträgen, jedes Element ist eine Liste aus
                3072 Integer, die ersten 1024 Zahlen entsprechen den Rot-Werten
                die nächsten 1024 den Grön-Werten und die letzten den
                Blauwerten -> uint8
labels          Die jeweilige Kategorie des entsprechenden Bildes als Zahl
                zwischen 0 und 9 (siehe oben)
batch_label     Beschreibung des Datensatzes
file_names      Dateinamen der einzelnen Bilder

"""
# %% Importe

import os
import pickle
import cv2
import matplotlib.image as mpimg

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tqdm

import tensorflow.keras.activations as acts
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.initializers as initializers
import keras.utils.np_utils as np_utils
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras import layers

from sklearn.metrics import confusion_matrix

from keras_tuner.tuners import Hyperband
from keras_tuner import HyperParameters


img_counter = 10


# %% Daten importieren

folder = "data"

file_1 = "data_batch_1"
file_2 = "data_batch_2"
file_3 = "data_batch_3"
file_4 = "data_batch_4"
file_5 = "data_batch_5"
file_test = "test_batch"
file_meta = "batches.meta"

train_file_list =[file_1, file_2, file_3, file_4, file_5]

# %% Funktionen für das Laden und Vorbereiten der Bilder,
# Kategorien und Target-Daten

def unpickle(folder, file):
    """ liest die Datei ein und extragiert das Dict"""
    file_path = os.path.join(folder, file)
    with open(file_path, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def show_img(dict, i):
    """"zeigt das i-te Bild aus dem dictionary dict"""
    pic = np.array(dict[b"data"][i])
    pic = pic.reshape((3,32,32))
    pic = pic.transpose(1,2,0)

    # Bild anzeigen
    plt.imshow(pic)
    plt.show()


def dicts_2_arrs(dict):
    """x-vals werden in ein numpy-Array der entsprechenden Form gebracht,
    die Werte selbst in float umgewandelt und normalisiert
    die y-vals werden in ein array geschrieben"""
    x_vals = np.array(dict[b"data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_vals = x_vals.astype(np.float32)
    x_vals /= 255.0
    y_vals = to_categorical(np.array(dict[b"labels"]))
    return x_vals, y_vals


def cats_2_vecs(dict):
    """Die Kategoerien der Bilder werden ausgelesen und in ein Array aus
    Einheitsvektoren umgeschrieben"""
    return to_categorical(dict[b"labels"])


# Liste der Kategorien mit entsprechenden Ziffern
cat_dir = unpickle(folder, file_meta)
cat_dir[b"num_vis"]
cat_list = [s.decode("utf8") for s in cat_dir[b"label_names"]]
for num, val in enumerate(cat_list):
    print(num, val)


batch_1_dict = unpickle(folder, file_1)
batch_1_dict[b"data"]
batch_1_dict[b"data"][0]

batch_1_target = cats_2_vecs(batch_1_dict)


# Ein paar Bilder anzeigen
for i in range(10):
    show_img(batch_1_dict, i)



# %% Trainingsdaten alle in zwei Arrays (trainX und trainy) packen

shape_X = (1,32,32,3)
shape_Y = (1,10)

train_X = np.empty(shape_X, dtype=np.float32)
train_Y = np.empty(shape_Y, dtype=np.uint8)

for file in tqdm.tqdm(train_file_list):
    dict = unpickle(folder, file)
    x, y = dicts_2_arrs(dict)
    train_X = np.concatenate((train_X, x), axis=0)
    train_Y = np.concatenate((train_Y, y), axis=0)

train_X = train_X[1:50001]
train_Y = train_Y[1:50001]

# %% Testdaten ebenfalls in arrays umandeln

test_dict = unpickle(folder, file_test)
test_X, test_Y = dicts_2_arrs(test_dict)


# %% CNN aufbauen - Parameter

width = 32
height = 32
depth = 3

num_classes = 10

train_size, test_size = train_X.shape[0] , test_X.shape[0]

# %%# Hyperparameter


# =============================================================================
# HYPERPARAMETER
# =============================================================================


epochs = 50
batch_size = 100
lr = 0.0003
optimizer = optimizers.Adam(learning_rate = lr)
# optimizer = optimizers.Adam() # default lr = 0.001

# %% Das eigentliche Model 1

model_name = "M1"

model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))
# kernel_size nur eine Zahl -> Quadratisch
# bei strides analog
# Bei erster Schicht immer noch input_shape mit angeben
# passing = "same" erhält die Größe der Matrix:
#   Vorteil: Details am Rand werden auch beachtet,
#   Nachteil: Mehr Rechnleistung notwendig

model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.25))


model.add(layers.Flatten())  # aufdröseln in einen Spaltenvektor
model.add(layers.Dense(128))  # "Normale" Layer
model.add(layers.Activation("relu"))
model.add(layers.Dense(num_classes))
model.add(layers.Activation("softmax"))  # für WSK


model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

# %% Model 2

model_name = "M2"

model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))

model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.25))


model.add(layers.Flatten())  # aufdröseln in einen Spaltenvektor
model.add(layers.Dense(512))  # "Normale" Layer
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128))  # "Normale" Layer
model.add(layers.Activation("relu"))
model.add(layers.Dense(num_classes))
model.add(layers.Activation("softmax"))  # für WSK


model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])



# %% Hypertuning Model 2

# Definition der Hyperparameter-Bereiche
hp = HyperParameters()
# hp.Choice('pool_size', values=[(2, 2), (3, 3)])
# hp.Choice('optimizer', values=['adam', 'sgd'])
hp.Float('dropout', min_value=0.2, max_value=0.4, step=0.05)
hp.Int('filters1', min_value=26, max_value=40, step = 2)
hp.Int('filters2', min_value=26, max_value=40, step = 2)


def build_model(hp):
    model = models.Sequential()

    model.add(layers.Conv2D(filters=hp.get('filters1'), kernel_size=3, strides=1, padding="same",
                            input_shape=(width, height, depth)))

    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(layers.Dropout(hp.get("dropout")))

    model.add(layers.Conv2D(filters=hp.get('filters2'), kernel_size=3, strides=1, padding="same",
                            input_shape=(width, height, depth)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(layers.Dropout(hp.get("dropout")))


    model.add(layers.Flatten())  # aufdröseln in einen Spaltenvektor
    model.add(layers.Dense(512))  # "Normale" Layer
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(hp.get("dropout")))
    model.add(layers.Dense(128))  # "Normale" Layer
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation("softmax"))  # für WSK


    # model.summary()

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate = 0.0003),
        metrics=["accuracy"])

    return model

# Erstellung des Tuners und Durchführung der Suche
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=100,
    hyperparameters=hp,
    directory='tuning',
    project_name='M2')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
tuner.search(x=train_X, y=train_Y, validation_data=(test_X, test_Y), epochs=10, callbacks=[stop_early])

# Abrufen der besten Hyperparameter-Kombination
best_hp = tuner.get_best_hyperparameters()[0]


print("Beste Hyperparameter:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Erstellen des finalen Modells mit den besten Hyperparametern
final_model = tuner.hypermodel.build(best_hp)


# %% Model 2 nach Hypertuning

model_name = "M2"

model = models.Sequential()

model.add(layers.Conv2D(filters=36, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))

model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters=38, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.25))


model.add(layers.Flatten())  # aufdröseln in einen Spaltenvektor
model.add(layers.Dense(512))  # "Normale" Layer
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128))  # "Normale" Layer
model.add(layers.Activation("relu"))
model.add(layers.Dense(num_classes))
model.add(layers.Activation("softmax"))  # für WSK




model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate = 0.0003),
    metrics=["accuracy"])

# %% Model 3

model_name = "M3"


model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())  # aufdröseln in einen Spaltenvektor
model.add(layers.Dense(256))  # "Normale" Layer
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(128))  # "Normale" Layer
model.add(layers.Activation("relu"))
model.add(layers.Dense(num_classes))
model.add(layers.Activation("softmax"))  # für WSK


model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])


# %% Hypertuning mit Model 3

# Definition der Hyperparameter-Bereiche
hp = HyperParameters()
hp.Float('dropout', min_value=0.2, max_value=0.4, step=0.05)
hp.Int('filters1', min_value=26, max_value=40, step = 2)
hp.Int('filters2', min_value=26, max_value=40, step = 2)
hp.Int('filters3', min_value=26, max_value=40, step = 2)


def build_model(hp):
    model = models.Sequential()

    model.add(layers.Conv2D(filters=hp.get('filters1'), kernel_size=3, strides=1, padding="same",
                            input_shape=(width, height, depth)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(layers.Dropout(hp.get('dropout')))

    model.add(layers.Conv2D(filters=hp.get('filters2'), kernel_size=3, strides=1, padding="same",
                            input_shape=(width, height, depth)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(layers.Dropout(hp.get('dropout')))

    model.add(layers.Conv2D(filters=hp.get('filters3'), kernel_size=3, strides=1, padding="same",
                            input_shape=(width, height, depth)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
    model.add(layers.Dropout(hp.get('dropout')))

    model.add(layers.Flatten())  # aufdröseln in einen Spaltenvektor
    model.add(layers.Dense(256))  # "Normale" Layer
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(hp.get('dropout')))
    model.add(layers.Dense(128))  # "Normale" Layer
    model.add(layers.Activation("relu"))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation("softmax"))  # für WSK

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.Adam(learning_rate = 0.0007),
        metrics=["accuracy"])

    return model


# Erstellung des Tuners und Durchführung der Suche
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=100,
    hyperparameters=hp,
    directory='tuning',
    project_name='M3')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
tuner.search(x=train_X, y=train_Y, validation_data=(test_X, test_Y), epochs=10, callbacks=[stop_early])

# Abrufen der besten Hyperparameter-Kombination
best_hp = tuner.get_best_hyperparameters()[0]


print("Beste Hyperparameter:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Erstellen des finalen Modells mit den besten Hyperparametern
final_model = tuner.hypermodel.build(best_hp)


# %% Model 3 nach Hypertuning


model_name = "M3"


model = models.Sequential()

model.add(layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(filters=40, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(filters=38, kernel_size=3, strides=1, padding="same",
                        input_shape=(width, height, depth)))
model.add(layers.Activation("relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid"))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())  # aufdröseln in einen Spaltenvektor
model.add(layers.Dense(256))  # "Normale" Layer
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128))  # "Normale" Layer
model.add(layers.Activation("relu"))
model.add(layers.Dense(num_classes))
model.add(layers.Activation("softmax"))  # für WSK

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate = 0.0007),
    metrics=["accuracy"])

# %% LeNet-5

model_name="LeNet-5"


model = models.Sequential()
model.add(layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(120, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])


# %% LeNet-5 - Version 2

model_name="LeNet-5-2"


model = models.Sequential()
model.add(layers.Conv2D(35, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.30))
model.add(layers.Conv2D(18, kernel_size=(3, 3), activation='relu'))
model.add(layers.AveragePooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.30))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.30))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])


# %% Le-Net Version 2 Hyper Tuning

# Definition der Hyperparameter-Bereiche
hp = HyperParameters()
# hp.Choice('pool_size', values=[(2, 2), (3, 3)])
hp.Choice('optimizer', values=['adam', 'sgd'])
hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.05)
hp.Int('filters1', min_value=30, max_value=40, step = 2)
hp.Int('filters2', min_value=16, max_value=26, step = 2)

def build_model(hp):
    model = models.Sequential()
    model.add(layers.Conv2D(hp.get("filters1"), kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(hp.get("dropout")))
    model.add(layers.Conv2D(hp.get("filters2"), kernel_size=(3, 3), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(hp.get("dropout")))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.30))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"])

    return model

# Erstellung des Tuners und Durchführung der Suche
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=100,
    hyperparameters=hp,
    directory='tuning',
    project_name='LeNet-2')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
tuner.search(x=train_X, y=train_Y, validation_data=(test_X, test_Y), epochs=10, callbacks=[stop_early])

# Abrufen der besten Hyperparameter-Kombination
best_hp = tuner.get_best_hyperparameters()[0]


print("Beste Hyperparameter:")
for param, value in best_hp.values.items():
    print(f"{param}: {value}")

# Erstellen des finalen Modells mit den besten Hyperparametern
final_model = tuner.hypermodel.build(best_hp)


# %% MiniVGG

model_name = "MiniVGG"

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate = 0.0003),
    metrics=["accuracy"])



# %% MiniVGG 2

model_name = "MiniVGG-2"

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Dropout(0.30))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.30))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.30))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.30))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizers.Adam(learning_rate = 0.0003),
    metrics=["accuracy"])




# %% Training und Loggen der Trainingsdaten

# =============================================================================
# TRAINING
# =============================================================================

history = model.fit(x=train_X, y=train_Y, verbose=1, batch_size=batch_size,
                    epochs=epochs, validation_data=[test_X, test_Y])

# %% Speichern des Modells

model.save(model_name)

# %% Laden eines Modells

# model_name = "M1"
# model_name = "M2"
model_name = "M3"

model = models.load_model(model_name)

print(model.summary())


# %% Plots

# Zugriff auf die Accuracy-Werte
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot erstellen
fig, axs = plt.subplots(1, 2)

# Plot der Accuracy-Werte
axs[0].plot(train_accuracy, label='Train Accuracy')
axs[0].plot(val_accuracy, label='Validation Accuracy')
# axs[0].set_ylim(0.94, 1)
axs[0].set_xlabel('Epochen')
axs[0].set_title('Accuracy')

axs[0].legend()


axs[1].plot(train_loss, label='Train Loss')
axs[1].plot(val_loss, label='Validation Loss')
# axs[1].set_ylim(0, 0.2)
axs[1].set_xlabel('Epochen')
axs[1].set_title('Loss')
axs[1].legend()

fig.suptitle("M: " + model_name + ", Eps: " + str(epochs) +", Bt : " + str(batch_size) + ", LR: "+ str(lr))
plt.tight_layout()
plt.show()

graph_folder = ".//graph//"
graph_name = str(img_counter).zfill(2)+" M"+model_name + " E" + str(epochs) +" Bt" + str(batch_size) + " L" + str(lr) +".png"

graph_file = os.path.join(graph_folder, graph_name)

fig.savefig(graph_file, dpi=600)
img_counter += 1

# %% Confusion Matrix

predictions = model.predict(test_X)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(test_Y, axis=1)

# Erstellen der Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)

print(cm)

# %% Test
# Test the CNN
score = model.evaluate(test_X, test_Y, verbose=1)
print("Score: ", score)


# %% Einlesen eines jpg-Bildes und Umwandeln in ein Numpy-Array

images = ["Porsche-klein.png", "Playmobil-klein.png","Hund1-klein.jpg",
          "Hund2-klein.jpg","Hirsch-klein.jpg", "Lotti-klein.jpg"]


image_path = "Bilder"

categories = ['Flugzeug', 'Auto', 'Vogel', 'Katze', 'Hirsch', 'Hund', 'Frosch', 'Pferd', 'Schiff', 'LKW']

def show_image(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


def process_image(image_path):
    image = cv2.imread(image_path) # Bild einlesen
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Bild in das RGB-Farbformat konvertieren
    normalized_image = rgb_image.astype('float32') / 255.0  # Bild in einen Float32-Array umwandeln und dann normalisieren
    reshaped_image = np.expand_dims(normalized_image, axis=0)  # Bild in die Shape des CIFAR-10-Datensatzes bringen
    return reshaped_image


def predict_image(image_path, model):
    img = process_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return predicted_class


for image in images:
    image_con = os.path.join(image_path, image)
    show_image(image_con)

    processed_image = process_image(image_con)

    predicted_class = predict_image(image_con, model)
    predicted_label = categories[predicted_class]
    print(image, ":\t", predicted_label)
