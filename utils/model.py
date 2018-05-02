"""
model.py

MLP 모델 bulid, parameter 관련 class 생성
모델은 keras로 설정되어 있음(build_model 부분)
"""
import os
from utils.datapreprocess import scale_fit


class Churn_model:
    def __init__(self, input_dim, final_output_dim, reg_param=0.01, drop_rate=0.5):
        self.input_dim = input_dim
        self.final_output_dim = final_output_dim
        self.epoch = 5
        self.batch_size = 100
        self.reg_param = reg_param
        self.drop_rate = drop_rate
        self.model = self.build_model()
        self.model_verbose = 1
        self.verbose = 0
        self.ES = False

        self.check_path = 'D:\churn\checkpoint'

    def build_model(self):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, BatchNormalization, Activation
        from keras.initializers import glorot_uniform
        from keras import regularizers
        input_dim = self.input_dim
        final_output_dim = self.final_output_dim
        reg_param = self.reg_param
        drop_rate = self.drop_rate

        output_dim1 = round(input_dim)  # 1st hidden node 개수 = input의 절반으로 설정
        output_dim2 = round(output_dim1)  # 2nd hidden node 개수 = 또 절반으로 설정(Dimension reduction)
        # output_dim3 = round(output_dim2)

        # Xavier uniform initialization + l2-regularization(0.01) model
        model = Sequential()
        model.add(Dense(units=output_dim1, input_shape=(input_dim,), use_bias=False,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='glorot_uniform',
                        kernel_regularizer=regularizers.l2(reg_param),
                        bias_regularizer=regularizers.l2(reg_param)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(units=output_dim2, use_bias=False,
                        kernel_initializer=glorot_uniform(),
                        bias_initializer=glorot_uniform(),
                        kernel_regularizer=regularizers.l2(reg_param),
                        bias_regularizer=regularizers.l2(reg_param)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(drop_rate))
        model.add(Dense(units=final_output_dim, kernel_initializer=glorot_uniform(),
                        bias_initializer=glorot_uniform(), activation='sigmoid'))
        model.summary()
        return model

    def train(self, X_train, X_test, Y_train, Y_test, x_offset=1):
        from time import time
        if self.verbose >= 1:
            print("학습 시작!")
        start_time = time()
        sc = scale_fit(X_train[:, x_offset:])
        self.sc = sc
        X_train = self.scale(X_train, x_offset)
        X_test = self.scale(X_test, x_offset)
        history = self.fit(X_train, X_test, Y_train, Y_test)
        if self.verbose >= 1:
            loss_and_accuracy = self.model.evaluate(X_test, Y_test, verbose=0)
            print('loss_and_Accuracy : ' + str(loss_and_accuracy))
            print("Baseline Error: %.2f%%" % (100 - loss_and_accuracy[1] * 100))
            end_time = time()
            print("학습 종료, 걸린 시간:", end_time - start_time, '\n')
        if self.model_verbose == 2:
            self.plot_history(history)

    def fit(self, X_train, X_test, Y_train, Y_test):
        # from keras.utils.training_utils import multi_gpu_model
        # self.model = multi_gpu_model(self.model, gpus=4)
        from keras.optimizers import Adam
        # self.model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error', metrics=['accuracy'])
        from keras.callbacks import ModelCheckpoint
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.check_path, 'churn_check.hdf5'))
        if self.ES:
            from keras.callbacks import EarlyStopping, ReduceLROnPlateau
            RL = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.001, cooldown=3,  verbose=1)
            ES = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
            history = self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                                     epochs=self.epoch, batch_size=self.batch_size, verbose=1,
                                     callbacks=[checkpoint, RL, ES], shuffle=True)
        else:
            history = self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                                     epochs=self.epoch, batch_size=self.batch_size, verbose=1,
                                     callbacks=[checkpoint], shuffle=True)
        return history

    def predict(self, X, x_offset):
        X = self.scale(X, x_offset)
        return self.model.predict(X)

    def scale(self, X, x_offset):
        X = X[:, x_offset:]
        return self.sc.transform(X)

    def plot_history(self, history):
        import matplotlib.pyplot as plt
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



