import cv2
import os
import numpy as np
import pickle
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten, Dense
from helpers import resize_to_fit

dados = []
rotulos = []

past_base_imagens = "base_letras"

imagens = paths.list_images(past_base_imagens)

for arquivo in imagens:
    rotulo = arquivo.split(os.path.sep)[1]

    imagem = cv2.imread(arquivo)
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # padronizar o tamanho da imagem em 20X20
    imagem = resize_to_fit(imagem, 20, 20)
    # precisa adicionar uma dimensão na imagem, pq o keras precisa de 3 dimenssões no método
    # que será utilizado em seguida
    imagem = np.expand_dims(imagem, axis=2)

    rotulos.append(rotulo)
    dados.append(imagem)

dados = np.array(dados, dtype='float') / 255
rotulos = np.array(rotulos)

# separação em dados de treino (75%) e dados de teste (25%)
# Y (rótulos) são as respostas e X são os dados(imagens)
(X_train, X_test, Y_train, Y_test) = train_test_split(dados, rotulos, test_size=0.25, random_state=0)

# converter com one-hot-encoding
lb = LabelBinarizer().fit(Y_train)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# salvar o labelbinarizer e um arquivo com o pickle

with open('rotulos_modelo.dat', 'wb') as arquivo_pickle:
    pickle.dump(lb, arquivo_pickle)

# criar e treinar a inteligência artificial
# cria uma rede neural de várias camadas
modelo = Sequential()

# cria as camadas
modelo.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# criar a segunda camada
modelo.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# criar a terceira camada
modelo.add(Flatten())
modelo.add(Dense(500, activation="relu"))
# 500 é o número de neurônios

# última camada (saída)
modelo.add(Dense(26, activation="softmax"))

# compilar todasas camadas
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# TREINAMENTO DA REDE NEURAL
modelo.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=26, epochs=20, verbose=1)

# salvar modelo em arquivo
modelo.save("modelo_treinado.hdf5")
