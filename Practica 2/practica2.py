import os
import numpy
import matplotlib.pyplot as plot
from PIL import Image, UnidentifiedImageError
from keras import Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, SimpleRNN, InputLayer
from tensorflow import keras
from keras.models import Sequential
from keras.optimizers import gradient_descent_v2, adam_v2
from keras.layers.core import Dense, Activation
from keras.regularizers import l1, l2
import seaborn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(HEIGHT, WIDTH) = (128, 128)


def plot_results(y_pred, y_test, hist):
    seaborn.heatmap(confusion_matrix(y_test, y_pred.astype(int)), annot=True)
    plot.xlabel("Valores verdaderos")
    plot.ylabel("Valores predichos")
    plot.show()

    plot.plot(hist.history["accuracy"])
    plot.plot(hist.history["val_accuracy"])
    plot.xlabel("Epoch")
    plot.ylabel("Accuracy")
    plot.legend(["Train", "Test"])
    plot.show()


#############################################
#############################################

def NeuralNetwork(x_train_arg, y_train_arg, x_test_arg, y_test_arg, x_val_arg, y_val_arg, type, epochs):
    x_train = x_train_arg.copy()
    y_train = y_train_arg.copy()
    x_test = x_test_arg.copy()
    y_test = y_test_arg.copy()
    x_val = x_val_arg.copy()
    y_val = y_val_arg.copy()

    #############################################
    # Convolutive NN
    #############################################
    if type == "1":
        model = Sequential([
            InputLayer((HEIGHT, WIDTH, 3)),
            # Capa 1
            # Red neuronal convolutiva con mascara 3x3
            Conv2D(32, activation="relu", kernel_size=3),  # , kernel_regularizer=l2(0.01)),
            # Capa 2
            Conv2D(64, activation="relu", kernel_size=3),  # , kernel_regularizer=l2(0.01)),
            # Capa 3
            # Conv2D(128, activation="relu", kernel_size=3),  # , kernel_regularizer=l2(0.01)),
            # Capa 4
            # Conv2D(128, activation="relu", kernel_size=3),  # kernel_regularizer=l2(0.01)),
            # Capa 5
            # Conv2D(1500, activation="relu", kernel_size=3),  # , kernel_regularizer=l2(0.01)),
            MaxPooling2D(pool_size=(2, 2)),
            # Para evitar el sobreajuste se eliminan nodos aleatoriamente
            Dropout(0.8),
            # Serializa(tranforma) un tensor(array)
            Flatten(),
            Dense(64, activation="relu", input_dim=HEIGHT * WIDTH),
            # Capa 6
            Dense(1, activation="sigmoid")  # , kernel_regularizer=l2(0.01))
        ])
    #############################################
    # Deep NN
    #############################################
    elif type == "2":
        model = Sequential([
            Input(shape=(500, 500, 1)),
            Flatten(),
            # Capa 1
            # Dense(32, activation="relu", input_dim=784),
            # Capa 2
            Dense(64, activation="relu", input_dim=784),
            # Capa 3
            # Dense(256, activation="relu", input_dim=784),
            # Capa 4
            # Dense(500, activation="relu", input_dim=784),
            # Capa 4
            # Dense(784, activation="relu", input_dim=784),
            # Capa 4
            # Dense(200, activation="relu", input_dim=784),
            # Capa 5
            Dense(128, activation="relu", input_dim=784),
            # Para evitar el sobreajuste se eliminan nodos aleatoriamente
            Dropout(0.8),
            Flatten(),
            # Capa 6
            Dense(10, activation="softmax")
        ])
    #############################################
    # Recurrent NN
    #############################################
    elif type == "3":
        model = Sequential([
            Input(shape=(500, 500)),
            SimpleRNN(64, activation="relu", input_shape=(500, 500)),
            Dropout(0.5),
            Dense(10, activation="softmax")
        ])
    #############################################
    #############################################
    # x_train = x_train.reshape(x_train.shape[0], 500, 500, 1)
    # y_train = to_categorical(y_train, 1)
    # x_test = numpy.reshape(x_test, (500, 500, 1))
    # y_test = to_categorical(y_test, 1)
    #############################################
    #############################################
    # Regularizacion
    regularization = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10)
    # sparse_categorical_crossentropy
    lr = 0.1
    model.compile(optimizer=gradient_descent_v2.SGD(learning_rate=lr, decay=lr / epochs), loss="binary_crossentropy",
                  metrics=["accuracy"])
    hist = model.fit(numpy.array(x_train, numpy.float32), numpy.array(list(map(int, y_train)), numpy.float32),
                     epochs=epochs, validation_split=0.1,
                     batch_size=32, callbacks=[regularization], validation_data=(x_val, y_val))
    accuracy = model.evaluate(numpy.array(x_test), numpy.array(y_test, numpy.float32))
    y_pred = (model.predict(numpy.array(x_test)) > 0.5).astype(int).ravel()
    plot_results(y_pred, numpy.array(y_test, numpy.float32), hist)
    #############################################
    #############################################
    print("Accuracy: " + str(round(accuracy[1], 3)))
    #############################################
    print("% error test: " + str(round(1.0 - accuracy[1], 3)))


#############################################
#############################################
#############################################
def load_dataset(dataset):
    imgs = []
    class_type = []

    max_elements = len(os.listdir(dataset + "/1"))

    for dir in os.listdir(dataset):
        elements = 0

        if os.path.isdir(os.path.join(dataset, dir)):
            for file in os.listdir(os.path.join(dataset, dir)):
                if elements < max_elements:
                    try:
                        path = os.path.join(dataset, dir, file)
                        img = numpy.array(Image.open(path))  # cv2.imread(path, cv2.COLOR_BGR2RGB)
                        img = (numpy.resize(img, (HEIGHT, WIDTH, 3))).astype(
                            "float32")  # numpy.array(cv2.resize(img, (HEIGHT, WIDTH))).astype("float32")
                        img /= 255
                        imgs.append(img)
                        class_type.append(int(dir))
                        elements += 1
                    except UnidentifiedImageError:
                        print("Image error")
    return imgs, class_type


if __name__ == "__main__":
    epochs = 5

    dataset = "medium10000_twoClasses"
    dataset_train = "data/{}/train".format(dataset)
    dataset_test = "data/{}/test".format(dataset)
    dataset_val = "data/{}/val".format(dataset)

    x_train, y_train = load_dataset(dataset_train)
    x_test, y_test = load_dataset(dataset_test)
    x_val, y_val = load_dataset(dataset_val)

    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("------------------------Practica 2: Fakkedit----------------------")
    print("------------------------------------------------------------------")
    print("------------------------------------------------------------------")
    print("Selecciona una opcion:")
    print("1.-Convolutive NN, 2.-Deep NN, 3.-Recurrent NN")
    # n = input()

    NeuralNetwork(x_train, y_train, x_test, y_test, x_val, y_val, "1", 10)
