from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import random
from skimage.measure import block_reduce

#To find the duration of wave file in seconds
import wave
import contextlib

#Keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json

import time
import datetime

os.environ['KMP_DUPLICATE_LIB_OK']  = 'True'
imwidth                             = 50
imheight                            = 34
total_examples                      = 2000
speakers                            = 4
examples_per_speaker                = 50
tt_split                            = 0.1
num_classes                         = 10
test_rec_folder                     = "./testrecs"
log_image_folder                     = "./logims"
recording_directory                 = "../SoundCNN/recordings/"
num_test_files                      = 1

THRESHOLD                           = 1000
CHUNK_SIZE                          = 512
FORMAT                              = pyaudio.paInt16
RATE                                = 8000#44100
WINDOW_SIZE                         = 50
CHECK_THRESH                        = 3
SLEEP_TIME                          = 0.5 #(seconds)
IS_PLOT                             = 1
LOG_MODE                            = 0 # 1 for time, 2 for frequency

def create_train_test(audio_dir):
    file_names = [f for f in os.listdir(audio_dir) if '.wav' in f]
    file_names.sort()
    test_list = []
    train_list = []
    
    for i in range(int(total_examples/examples_per_speaker)):
        test_list.extend(random.sample(file_names[(i*examples_per_speaker+1):(i+1)*examples_per_speaker], int(examples_per_speaker*tt_split)))

    train_list = [x for x in file_names if x not in test_list]

    y_test = np.zeros(len(test_list))
    y_train = np.zeros(len(train_list))
    x_train = np.zeros((len(train_list), imheight, imwidth))
    x_test = np.zeros((len(test_list), imheight, imwidth))

    tuni1   = np.zeros(len(test_list))
    tuni2   = np.zeros(len(test_list))

    for i, f in enumerate(test_list):
        y_test[i]     = int(f[0])
        spectrogram   = graph_spectrogram( audio_dir + f )
        graygram      = rgb2gray(spectrogram)
        normgram      = normalize_gray(graygram)
        norm_shape    = normgram.shape
        if(norm_shape[0]>150):
            continue
        redgram       = block_reduce(normgram, block_size = (3,3), func = np.mean)
        x_test[i,:,:] = redgram
        print("Progress Test Data: {:2.1%}".format(float(i) / len(test_list)), end="\r")

    for i, f in enumerate(train_list):
        y_train[i] = int(f[0])
        spectrogram   = graph_spectrogram( audio_dir + f )
        graygram      = rgb2gray(spectrogram)
        normgram      = normalize_gray(graygram)
        norm_shape    = normgram.shape
        if(norm_shape[0]>150):
            continue
        redgram       = block_reduce(normgram, block_size = (3,3), func = np.mean)
        x_train[i,:,:] = redgram
        print("Progress Training Data: {:2.1%}".format(float(i) / len(train_list)), end="\r")
        
    return x_train, y_train, x_test, y_test

def create_model(path):
    x_train, y_train, x_test, y_test = create_train_test(path)

    print("Size of Training Data:", np.shape(x_train))
    print("Size of Training Labels:", np.shape(y_train))
    print("Size of Test Data:", np.shape(x_test))
    print("Size of Test Labels:", np.shape(y_test))

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.reshape(x_train.shape[0], imheight, imwidth, 1)
    x_test = x_test.reshape(x_test.shape[0], imheight, imwidth, 1)
    input_shape = (imheight, imwidth, 1)
    batch_size = 4
    epochs = 1

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    print(model.summary())

    model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test))
    return model

def get_wav_data(path):
    input_wav           = path
    spectrogram         = graph_spectrogram( input_wav )
    graygram            = rgb2gray(spectrogram)
    normgram            = normalize_gray(graygram)
    norm_shape          = normgram.shape
    #print("Spec Shape->", norm_shape)
    if(norm_shape[0]>100):
        redgram             = block_reduce(normgram, block_size = (26,26), func = np.mean)
    else:
        redgram             = block_reduce(normgram, block_size = (3,3), func = np.mean)
    redgram             = redgram[0:imheight,0:imwidth]
    red_data            = redgram.reshape(imheight,imwidth, 1)
    empty_data          = np.empty((1,imheight,imwidth,1))
    empty_data[0,:,:,:] = red_data
    new_data            = empty_data
    return new_data

def save_model_to_disk(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def load_model_from_disk():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

def generate_log(in_dir, num_samps_per_cat):
    file_names = [f for f in os.listdir(in_dir) if '.wav' in f]
    checklist  = np.zeros(num_samps_per_cat * 10)
    final_list = []
    iternum = 0
    
    #Get a random sample for each category
    while(1):
        print("Iteration Number:", iternum)
        sample_names = random.sample(file_names,10)
        for name in sample_names:
            categ = int(name[0])
            if(checklist[categ]<num_samps_per_cat):
                checklist[categ]+=1
                final_list.append(name)
        if(int(checklist.sum())==(num_samps_per_cat * 10)):
            break 
        iternum+=1
    print(final_list)

    #Generate Images for each sample
    lif = os.path.join(log_image_folder,time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))
    if not os.path.exists(lif):
        os.makedirs(lif)
    for name in final_list:      
        #Time Domain Signal
        rate, data = wavfile.read(os.path.join(in_dir,name))
        if(LOG_MODE==1):   
            fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
            ax.set_title('Sound of ' +name[0] + ' - Sampled audio signal in time')
            ax.set_xlabel('Sample number')
            ax.set_ylabel('Amplitude')
            ax.plot(data)
            fig.savefig(os.path.join(lif, name[0:5]+'.png'))   # save the figure to file
            plt.close(fig)
    
        #Frequency Domain Signals
        if(LOG_MODE==2):
            fig,ax = plt.subplots(1)
            #fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
            #ax.axis('off')
            pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=511, NFFT=512)
            #ax.axis('off')
            #plt.rcParams['figure.figsize'] = [0.75,0.5]
            cbar = fig.colorbar(im)
            cbar.set_label('Intensity dB')
            #ax.axis("tight")

            # Prettify
            ax.set_title('Spectrogram of spoken ' +name[0] )
            ax.set_xlabel('time')
            ax.set_ylabel('frequency Hz')
            fig.savefig(os.path.join(lif, name[0]+'_spec.png'), dpi=300, frameon='false')
            plt.close(fig)




if __name__ == '__main__':
    if(not LOG_MODE):
        while(1):
            time.sleep(SLEEP_TIME)
            if(os.path.isfile('model.json')):
                print("please speak a word into the microphone")
                success = record_to_file(test_rec_folder)
                if(not success):
                    print(" Speak Again Clearly")
                    continue
            else:
                print("********************\n\nTraining The Model\n")
            if(os.path.isfile('model.json')):
                model = load_model_from_disk()
            else:
                model = create_model(recording_directory)
                save_model_to_disk(model)
            #fname = 'r4.wav'
            #new_data = get_wav_data(fname)
            for i in range(num_test_files):
            #for i in range(1):
                fname = str(i)+".wav"
                new_data    = get_wav_data(os.path.join(test_rec_folder,fname))    
                predictions = np.array(model.predict(new_data))
                maxpred = predictions.argmax()
                normpred = normalize_gray(predictions)*100
                predarr = np.array(predictions[0])
                sumx = predarr.sum()
                print("TestFile Name: ", fname, " The Model Predicts:", maxpred)
                for nc in range(num_classes):
                    confidence = np.round(100*(predarr[nc]/sumx))
                    print("Class ", nc, " Confidence: ", confidence)
                #print("TestFile Name: ",fname, " Values:", predictions)
                print("_____________________________\n")
    else:
        generate_log(recording_directory,6)












