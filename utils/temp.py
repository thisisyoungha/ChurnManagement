from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Add
from keras.initializers import glorot_uniform
from keras import regularizers, metrics
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import numpy as np
import os
import matplotlib.pyplot as plt

from utils.datapreprocess import scale_fit

class churn_model:
    def __init__(self, input_dim, final_output_dim, voc_index=186, reg_param=0.01, drop_rate=0.5):
        self.input_dim = input_dim
        self.final_output_dim = final_output_dim
        self.reg_param = reg_param
        self.drop_rate = drop_rate
        self.epoch = 5
        self.voc_index = voc_index
        self.model = self.build_model()
        self.checkpath = 'D:\churn\checkpoint'


    def build_model(self):
        nonvoc_input_dim = self.voc_index
        voc_input_dim = self.input_dim - self.voc_index
        nonvoc_hidden_dim = self.voc_index
        voc_hidden_dim = self.input_dim - self.voc_index
        nonvoc_output_dim = self.final_output_dim
        voc_output_dim = nonvoc_output_dim

        reg_param = self.reg_param
        drop_rate = self.drop_rate

        # output_dim1 = round(input_dim)  # 1st hidden node 개수 = input의 절반으로 설정
        # output_dim2 = round(output_dim1)  # 2nd hidden node 개수 = 또 절반으로 설정(Dimension reduction)
        # output_dim3 = round(output_dim2 / 2)

        # Non_VOC model
        input_layer = Input(shape=(nonvoc_input_dim,))
        hidden_layer = Dense(nonvoc_hidden_dim, activation='relu',
                             kernel_regularizer=regularizers.l2(reg_param),
                             bias_regularizer=regularizers.l2(reg_param))(input_layer)
        hidden_layer = Dropout(drop_rate)(hidden_layer)
        hidden_layer = Dense(nonvoc_hidden_dim, activation='relu',
                             kernel_regularizer=regularizers.l2(reg_param),
                             bias_regularizer=regularizers.l2(reg_param))(hidden_layer)
        hidden_layer = Dropout(drop_rate)(hidden_layer)
        output_layer = Dense(nonvoc_output_dim, activation='sigmoid')(hidden_layer)

        # VOC model
        voc_input_layer = Input(shape=(voc_input_dim,))
        voc_hidden_layer = Dense(voc_hidden_dim, activation='relu',
                                 kernel_regularizer=regularizers.l2(reg_param),
                                 bias_regularizer=regularizers.l2(reg_param))(voc_input_layer)
        voc_hidden_layer = Dropout(drop_rate)(voc_hidden_layer)
        voc_hidden_layer = Dense(voc_hidden_dim, activation='relu',
                                 kernel_regularizer=regularizers.l2(reg_param),
                                 bias_regularizer=regularizers.l2(reg_param))(voc_hidden_layer)
        voc_hidden_layer = Dropout(drop_rate)(voc_hidden_layer)
        voc_output_layer = Dense(voc_output_dim, activation='sigmoid')(voc_hidden_layer)

        # sum
        final_output_layer = Add()([output_layer, voc_output_layer])

        model = Model(inputs=[input_layer, voc_input_layer], outputs=final_output_layer)
        model.summary()
        model.input_shape()

        return model

    def train(self, X_train, X_test, Y_train, Y_test, x_offset, verbose=2):
        X_train = X_train[:, x_offset:]
        X_test = X_test[:, x_offset:]
        X_train, X_test, _ = scale_fit(X_train, X_test)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(filepath=os.path.join(self.checkpath, 'churn_check.hdf5'))
        history = self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                                 epochs=self.epoch, batch_size=100, verbose=0,
                                 callbacks=[checkpoint], shuffle=True)
        loss_and_accuracy = self.model.evaluate(X_test, Y_test, verbose=0)
        print('loss_and_Accuracy : ' + str(loss_and_accuracy))
        print("Baseline Error: %.2f%%" % (100 - loss_and_accuracy[1] * 100))
        if verbose == 2:
            self.plot_history(history)

    def train_predict(self, X_train, X_test, Y_train, Y_test, New_data, voc_index=185, x_offset=1, verbose=2):
        X_train = X_train[:, x_offset:]
        X_test = X_test[:, x_offset:]
        New_data = New_data[:, x_offset:]

        X_train, X_test, sc = scale_fit(X_train, X_test)
        New_data = sc.transform(New_data)
        print(X_train[:voc_index])
        print(X_train[voc_index:])
        # print('333')
        # print(np.concatenate(([X_train[:voc_index]], [X_train[voc_index:]]), axis=1))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        checkpoint = ModelCheckpoint(filepath=os.path.join(self.checkpath, 'churn_check.hdf5'))
        history = self.model.fit([X_train[:voc_index]], [X_train[voc_index:]], Y_train,
                                 validation_data=([X_test[:voc_index]], [X_test[voc_index:]], Y_test),
                                 epochs=self.epoch, batch_size=100, verbose=1,
                                 callbacks=[checkpoint], shuffle=True)
        loss_and_accuracy = self.model.evaluate([X_test[:voc_index], X_test[voc_index:]], Y_test, verbose=0)
        print('loss_and_Accuracy : ' + str(loss_and_accuracy))
        print("Baseline Error: %.2f%%" % (100 - loss_and_accuracy[1] * 100))
        if verbose == 2:
            self.plot_history(history)
        p_train = self.model.predict([X_train[:voc_index], X_train[voc_index:]])
        p_test = self.model.predict([X_test[:voc_index], X_test[voc_index:]])
        p_new = self.model.predict(New_data)
        return (p_train, p_test, p_new)

    def predict(self, X_train, X_test, New_data, x_offset):
        X_train = X_train[:, x_offset:]
        X_test = X_test[:, x_offset:]
        New_data = New_data[:, x_offset:]
        X_train, X_test, sc = scale_fit(X_train, X_test)
        New_data = sc.transform(New_data)
        p_train = self.model.predict(X_train)
        p_test = self.model.predict(X_test)
        p_new = self.model.predict(New_data)
        return (p_train, p_test, p_new)

    def plot_history(self, history):
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

