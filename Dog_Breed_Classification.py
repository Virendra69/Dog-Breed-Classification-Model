# We will create and train a model that takes the image of a dog a classifies as to a breed.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tensorflow as tf
from tensorflow import keras

images_ds = tf.data.Dataset.list_files(
    "E:\VS Code Programs\Python_Codes\Deep_Learning\Projects\Dog Breed Classification\dog-breed-identification\Train/*", shuffle=False)


def get_labels(file_path):
    import os
    label = tf.strings.split(file_path, os.path.sep)[-1]
    label = tf.strings.split(label, ".")[0]
    label = label.numpy()
    return label


df = pd.read_csv(
    'E:\VS Code Programs\Python_Codes\Deep_Learning\Projects\dog-breed-identification\labels.csv')

le = LabelEncoder()
df['breed'] = le.fit_transform(df['breed'])

X = []
y = []

for file_path in images_ds:
    label = get_labels(file_path)
    label = tf.compat.as_str_any(label)
    y.append(int(df['breed'][df.id == label]))

    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img)
    img = tf.image.resize(img, [128, 128])
    img = img/255
    X.append(img)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, shuffle=False, test_size=0.2)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu',
                        input_shape=(128, 128, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(120, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5)
model.evaluate(X_test, y_test)
