
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import normalize

import pandas as pd


def load_data():
    all_fields = ['day', 'loc', 'season', 'win', 'teamid', 'opp_teamid', 'winpct', 'score_mean', 
              'fgm_mean', 'fga_mean', 'fgm3_mean', 'fga3_mean', 'ftm_mean', 'fta_mean', 'or_mean', 
              'dr_mean', 'ast_mean', 'to_mean', 'stl_mean', 'blk_mean', 'pf_mean', 
              'winpct_opp', 'score_mean_opp', 'fgm_mean_opp', 'fga_mean_opp', 'fgm3_mean_opp', 
              'fga3_mean_opp', 'ftm_mean_opp', 'fta_mean_opp', 'or_mean_opp', 'dr_mean_opp',
              'ast_mean_opp', 'to_mean_opp', 'stl_mean_opp', 'blk_mean_opp', 'pf_mean_opp']

    fields = ['score_mean', 
              'fgm_mean', 'fga_mean', 'fgm3_mean', 'fga3_mean', 'ftm_mean', 'fta_mean', 'or_mean', 
              'dr_mean', 'ast_mean', 'to_mean', 'stl_mean', 'blk_mean', 'pf_mean', 
              'winpct_opp', 'score_mean_opp', 'fgm_mean_opp', 'fga_mean_opp', 'fgm3_mean_opp', 
              'fga3_mean_opp', 'ftm_mean_opp', 'fta_mean_opp', 'or_mean_opp', 'dr_mean_opp',
              'ast_mean_opp', 'to_mean_opp', 'stl_mean_opp', 'blk_mean_opp', 'pf_mean_opp']

    
    data = pd.read_csv("../data/team_season_games.csv")
    
    X = data[fields].values
    X = normalize(X, axis=0)
    y = data['win'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42)

    return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':
    batch_size = 100
    num_classes = 2
    epochs = 50
    
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = load_data()
    
    
#     x_train = x_train.astype('float32')
#     x_test = x_test.astype('float32')
#     x_train /= 255
#     x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    
    # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = Sequential()
#     model.add(Flatten(input_shape=x_train.shape[1:]))
    model.add(Dense(512, input_dim=x_train.shape[1], activation="relu"))
#     model.add(Dropout(0.4))
    model.add(Dense(1024, activation="relu"))
#     model.add(Dropout(0.4))
    model.add(Dense(512, activation="relu"))
#     model.add(Dropout(0.4))
    model.add(Dense(256, activation="relu"))
#     model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu"))
#     model.add(Dropout(0.4))
    model.add(Dense(1, activation="softmax"))
    
    es = EarlyStopping(monitor="val_loss", min_delta=0.0005, patience=5, verbose=1)
    mc = ModelCheckpoint("./models/model.{epoch:02d}-{val_loss:.4f}.hdf5", save_best_only=True)
    
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_split=0.1,
              callbacks=[es, mc])
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
