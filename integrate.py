from cProfile import label
import os, glob, random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time 

from scipy.fftpack import fft
import librosa

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam


class Music_genre():
    def __init__(self, folder_path = 'data/', ori=False, spec=False, FT=False, zc=False) -> None:
        self.label = os.listdir(folder_path)
        
        self.original = ori
        self.zc = zc
        self.FT = FT
        self.spec = spec
        self.music = {}

        for idx, folder_name in enumerate(self.label):
            self.music[folder_name] = []
            for i in glob.glob(folder_path + str(folder_name) + "/*.wav"):
                data, rate = librosa.load(i)

                if self.original:
                    self.music[folder_name].append(np.reshape(data[:67000], (67000, 1)))
                
                if self.zc:
                    zc = librosa.feature.zero_crossing_rate(data, frame_length = 2048, hop_length = 512)
                    self.music[folder_name].append(np.reshape(zc, (-1, 1))[:1280])

                if self.FT:
                    # data = fft(data[:67000])
                    data = np.abs(fft(data[:67000]))
                    self.music[folder_name].append(np.reshape(data, (67000, 1)))

                if self.spec:
                    spectogram = librosa.feature.melspectrogram(y=data, sr=rate)
                    spectogram = librosa.power_to_db(spectogram, ref=np.max)
                    spectogram = cv2.resize(spectogram, (128, 128))
                    self.music[folder_name].append(np.reshape(spectogram, (128, 128, 1)))


            print(folder_name, len(self.music[folder_name]))

    def data_plot(self):
        if self.original:
            plt.plot(self.music[self.label[0]][0])
            plt.ylabel('Amplitude')
            plt.xlabel('Time')

            plt.savefig("save/img/ori.png")

        if self.zc:
            plt.plot(self.music[self.label[0]][0])
            plt.savefig("save/img/ZC.png")
            plt.ylabel('Rate')
            plt.xlabel('Time')

        if self.FT:
            plt.plot(self.music[self.label[1]][0])
            plt.savefig("save/img/FT.png")

        if self.spec:
            plt.imshow(self.music[self.label[0]][0], origin="lower", cmap=plt.get_cmap('inferno'))
            plt.savefig("save/img/spec.png")
        plt.show()

    def transform(self, start=0, end=10):
        self.start, self.end = start, end
        self.train_x, self.train_y, self.test_x, self.test_y = [], [], [], []
        for idx, values in enumerate(self.music.values()):
            # print(key, len(values[:40]))
            random.shuffle(values)
            self.train_x = self.train_x + values[:start] + values[end:]
            self.train_y = self.train_y + [idx for num in range(len(values[:40]))]
            self.test_x  = self.test_x + values[start:end]
            self.test_y  = self.test_y + [idx for num in range(len(values[40:]))]

        self.train_x = np.array(self.train_x)
        self.train_y = to_categorical(np.array(self.train_y))
        self.test_x = np.array(self.test_x)
        self.test_y = to_categorical(np.array(self.test_y))
        print("Spilted data size: ")
        print("  ", len(self.train_x), len(self.train_y), len(self.test_x), len(self.test_y))
        self.input_shape = self.train_x.shape 
        print("Input size: ", self.input_shape)

    def spec_2d_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.input_shape[1:]))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        self.model.summary()

    def fft_1d_model(self):
        self.model = Sequential()
        self.model.add(Conv1D(32, 64, input_shape=self.input_shape[1:], activation='relu'))
        self.model.add(MaxPooling1D(pool_size=4)) 
        self.model.add(Conv1D(64, 32, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=4)) 
        self.model.add(Conv1D(128, 16, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=4)) 
        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu')) 
        self.model.add(Dense(10,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])
        self.model.summary()

    def model_train_evaluate(self, num_epochs=20, batch_size_=4):
        self.history = self.model.fit(self.train_x, self.train_y, epochs=num_epochs, batch_size = batch_size_, 
                                        validation_data=(self.test_x, self.test_y), shuffle=True)
        if self.spec:
            self.model.save("save/model/spectrogram_{}_{}.h5".format(self.start, self.end))

        loass, acc = self.model.evaluate(self.test_x, self.test_y)
        print("\nTrained model's accuracy: {:5.2f} %".format(100 * acc))
        return acc

    def plot_training(self):
        history = self.history
        plt.figure(figsize=(12, 4)) # 建立第一個畫板（figure）
        plt.subplot(121) 
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Accuracy of model')
        plt.ylabel('accuracy(%)')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')

        plt.subplot(122) 
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Loss of model')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper right')
        if self.zc:
            plt.savefig("save/img/model_zc_{}_{}.png".format(self.start, self.end))

        if self.original:
            plt.savefig("save/img/model_ori_{}_{}.png".format(self.start, self.end))
        if self.zc:
            plt.savefig("save/img/model_zc_{}_{}.png".format(self.start, self.end))
        if self.FT:
            plt.savefig("save/img/model_FT_{}_{}.png".format(self.start, self.end))
        if self.spec:
            plt.savefig("save/img/model_spec_{}_{}.png".format(self.start, self.end))
        plt.show()

if __name__=="__main__":
    EPOCHS, BATCH_SIZE = 20, 4

    ### 1d raw data & 1d conv ###
    # music_classifier = Music_genre(ori=True)
    # music_classifier.transform(10, 20)
    # music_classifier.fft_1d_model()

    ### 1d zero cross rate & 1d conv ###
    # music_classifier = Music_genre(zc=True)
    # music_classifier.transform(10, 20)
    # music_classifier.fft_1d_model()


    ### 1d fast Fourier transform & 1d conv ###
    # music_classifier = Music_genre(FT=True)
    # music_classifier.transform(30, 40)
    # music_classifier.fft_1d_model()

    ### 2d spectogram & 2d conv ###
    music_classifier = Music_genre(spec=True)
    music_classifier.transform(10, 20)
    music_classifier.spec_2d_model()

    # music_classifier.data_plot()


    ### Training and Evaluate ###
    time_start = time.time()
    evaluate_acc = music_classifier.model_train_evaluate(num_epochs=EPOCHS, batch_size_=BATCH_SIZE)
    time_end = time.time()
    print('Time cost: {:.3f}'.format(time_end - time_start), 's')

    music_classifier.plot_training()

    