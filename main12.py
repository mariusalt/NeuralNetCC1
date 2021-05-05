# mlp for multiclass classification
from numpy import argmax
import pandas as pd
import csv
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import kerastuner as kt
import tensorflow as tf
from sklearn import preprocessing
import skopt
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt import gp_minimize, forest_minimize
import tensorboard as tb
import tensorboard.program
import tensorboard.default
from keras.callbacks import TensorBoard
from keras import backend as K
from tensorflow.keras import regularizers
from datetime import datetime
log_dir_name = 'home/mat/Desktop/trials/'


start_date= datetime.now()
start_date=start_date.strftime("%d/%m/%Y %H:%M:%S")

# load the dataset
path = '/home/mat/Desktop/cond_coop/data/sesca.csv'
df = read_csv(path)
#print(df1.head(3))
# split into input and output columns
#df = df1.drop(df1.columns[0:19], axis=1)
print(df.head(5))
print(df['cooptyp'].unique())
df = df.drop('choice1', 1)
df = df.drop('cooptyp.1', 1)
print(list(df))
X = df.values[:, 1:]
y = df.values[:, 0]

# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define model

out=[]

for lay in range(2,16):
    dim_learning_rate = Real(low=1e-6, high=1e-1, prior='uniform',name='learning_rate')
    dim_activation = Categorical(categories=['relu', 'sigmoid'], name='activation')
  #  dim_regularization = Real(low=0.00001, high=0.01, prior='uniform', name='regularization')
    dim_regularization = Categorical(categories=[0.01, 0.001, 0.0001, 0.00001], name='regularization')
    dim_num_dense_nodes1 = Integer(low=5, high=512, name='num_dense_nodes1')
    dim_num_dense_nodes2 = Integer(low=5, high=512, name='num_dense_nodes2')
    dim_num_dense_nodes3 = Integer(low=5, high=512, name='num_dense_nodes3')
    dim_num_dense_nodes4 = Integer(low=5, high=512, name='num_dense_nodes4')
    dim_num_dense_nodes5 = Integer(low=5, high=512, name='num_dense_nodes5')
    dim_num_dense_nodes6 = Integer(low=5, high=512, name='num_dense_nodes6')
    dim_num_dense_nodes7 = Integer(low=5, high=512, name='num_dense_nodes7')
    dim_num_dense_nodes8 = Integer(low=5, high=512, name='num_dense_nodes8')
    dim_num_dense_nodes9 = Integer(low=5, high=512, name='num_dense_nodes9')
    dim_num_dense_nodes10 = Integer(low=5, high=512, name='num_dense_nodes10')
    dim_num_dense_nodes11 = Integer(low=5, high=512, name='num_dense_nodes11')
    dim_num_dense_nodes12 = Integer(low=5, high=512, name='num_dense_nodes12')
    dim_num_dense_nodes13 = Integer(low=5, high=512, name='num_dense_nodes13')
    dim_num_dense_nodes14 = Integer(low=5, high=512, name='num_dense_nodes14')
    dim_num_dense_nodes15 = Integer(low=5, high=512, name='num_dense_nodes15')
    dim_num_dropouts1 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts1')
    dim_num_dropouts2 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts2')
    dim_num_dropouts3 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts3')
    dim_num_dropouts4 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts4')
    dim_num_dropouts5 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts5')
    dim_num_dropouts6 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts6')
    dim_num_dropouts7 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts7')
    dim_num_dropouts8 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts8')
    dim_num_dropouts9 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts9')
    dim_num_dropouts10 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts10')
    dim_num_dropouts11 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts11')
    dim_num_dropouts12 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts12')
    dim_num_dropouts13 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts13')
    dim_num_dropouts14 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts14')
    dim_num_dropouts15 = Real(low=0, high=0.5, prior='uniform', name='num_dropouts15')
    dim_num_batch_size = Integer(low=5, high=100, name='num_batch_size')

    dimensions = [dim_learning_rate,
                  dim_activation,
                  dim_regularization,
                  dim_num_batch_size,
                  dim_num_dense_nodes1,
                  dim_num_dropouts1,
                  dim_num_dense_nodes2,
                  dim_num_dropouts2,
                  dim_num_dense_nodes3,
                  dim_num_dropouts3,
                  dim_num_dense_nodes4,
                  dim_num_dropouts4,
                  dim_num_dense_nodes5,
                  dim_num_dropouts5,
                  dim_num_dense_nodes6,
                  dim_num_dropouts6,
                  dim_num_dense_nodes7,
                  dim_num_dropouts7,
                  dim_num_dense_nodes8,
                  dim_num_dropouts8,
                  dim_num_dense_nodes9,
                  dim_num_dropouts9,
                  dim_num_dense_nodes10,
                  dim_num_dropouts10,
                  dim_num_dense_nodes11,
                  dim_num_dropouts11,
                  dim_num_dense_nodes12,
                  dim_num_dropouts12,
                  dim_num_dense_nodes13,
                  dim_num_dropouts13,
                  dim_num_dense_nodes14,
                  dim_num_dropouts14,
                  dim_num_dense_nodes15,
                    dim_num_dropouts15
                  ]
    dimensions=dimensions[:(lay+lay+4)]
    print(dimensions)

    default_parameters15 = [1e-5, 'relu', 0.001, 32, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70,0, 70,0, 70, 0, 70, 0,70, 0, 70, 0, 70, 0]
    default_parameters14 = [1e-5, 'relu', 0.001, 32, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0,70, 0,70, 0]
    default_parameters13 = [1e-5, 'relu', 0.001, 32, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0,70, 0]
    default_parameters12 =[1e-5,'relu', 0.001, 32,  70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0]
    default_parameters11 = [1e-5,'relu', 0.001, 32,  70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0]
    default_parameters10 = [1e-5,'relu', 0.001, 32,  70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0]
    default_parameters9 = [1e-5,'relu', 0.001, 32,  70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0 ]
    default_parameters8 = [1e-5,'relu', 0.001, 32,  70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0]
    default_parameters7 = [1e-5,'relu', 0.001, 32,  70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0]
    default_parameters6 = [1e-5,'relu', 0.001, 32,  70, 0, 70, 0, 70, 0, 70, 0, 70, 0, 70, 0]
    default_parameters5 = [1e-5,'relu', 0.001, 32,  70, 0, 70, 0, 70, 0, 70, 0, 70, 0]
    default_parameters4 = [1e-5,'relu', 0.001, 32,  70, 0, 70, 0, 70, 0, 70, 0]
    default_parameters3 = [1e-5,'relu', 0.001,  32, 70, 0, 70, 0, 70, 0 ]
    default_parameters2 = [1e-5,'relu', 0.001, 32,  70, 0, 70, 0]

    def_para=['none','none',default_parameters2,default_parameters3,default_parameters4,default_parameters5,default_parameters6,default_parameters7,default_parameters8,default_parameters9,default_parameters10
              ,default_parameters11,default_parameters12,default_parameters13,default_parameters14,default_parameters15]

    if lay==15:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10, num_dense_nodes11, num_dense_nodes12, num_dense_nodes13, num_dense_nodes14, num_dense_nodes15,
                         num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13, num_dropouts14, num_dropouts15):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))
            l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4,
                  num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9,
                  num_dense_nodes10, num_dense_nodes11, num_dense_nodes12, num_dense_nodes13, num_dense_nodes14,
                  num_dense_nodes15]
            l2=[  num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                  num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                  num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13, num_dropouts14, num_dropouts15
                  ]
            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)

                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])


            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate,activation, regularization,num_dense_nodes1,num_dense_nodes2,num_dense_nodes3,num_dense_nodes4,
                    num_dense_nodes5,num_dense_nodes6,num_dense_nodes7,num_dense_nodes8,num_dense_nodes9,
                    num_dense_nodes10,num_dense_nodes11,num_dense_nodes12, num_dense_nodes13, num_dense_nodes14, num_dense_nodes15,
                    num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                    num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                    num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13, num_dropouts14,
                    num_dropouts15,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dense_nodes6=num_dense_nodes6,
                                 num_dense_nodes7=num_dense_nodes7,
                                 num_dense_nodes8=num_dense_nodes8,
                                 num_dense_nodes9=num_dense_nodes9,
                                 num_dense_nodes10=num_dense_nodes10,
                                 num_dense_nodes11=num_dense_nodes11,
                                 num_dense_nodes12=num_dense_nodes12,
                                 num_dense_nodes13=num_dense_nodes13,
                                 num_dense_nodes14=num_dense_nodes14,
                                 num_dense_nodes15=num_dense_nodes15,
                                 num_dropouts1=num_dropouts1,
                                 num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5,
                                 num_dropouts6=num_dropouts6,
                                 num_dropouts7=num_dropouts7,
                                 num_dropouts8=num_dropouts8,
                                 num_dropouts9=num_dropouts9,
                                 num_dropouts10=num_dropouts10,
                                 num_dropouts11=num_dropouts11,
                                 num_dropouts12=num_dropouts12,
                                 num_dropouts13=num_dropouts13,
                                 num_dropouts14=num_dropouts14,
                                 num_dropouts15=num_dropouts15)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dense_nodes6,
                            num_dense_nodes7,
                            num_dense_nodes8,
                            num_dense_nodes9,
                            num_dense_nodes10,
                            num_dense_nodes11,
                            num_dense_nodes12,
                            num_dense_nodes13,
                            num_dense_nodes14,
                            num_dense_nodes15,
                            num_dropouts2,
                            num_dropouts3,
                            num_dropouts4,
                            num_dropouts5,
                            num_dropouts6,
                            num_dropouts7,
                            num_dropouts8,
                            num_dropouts9,
                            num_dropouts10,
                            num_dropouts11,
                            num_dropouts12,
                            num_dropouts13,
                            num_dropouts14,
                            num_dropouts15,
                            accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=12,
                                    random_state=1234)
        except ValueError:
            pass
    elif lay==14:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10, num_dense_nodes11, num_dense_nodes12, num_dense_nodes13, num_dense_nodes14,
                         num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13, num_dropouts14
                         ):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10, num_dense_nodes11, num_dense_nodes12, num_dense_nodes13, num_dense_nodes14]
                l2=[  num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                      num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                      num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13, num_dropouts14
                      ]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate,activation, regularization,num_dense_nodes1,num_dense_nodes2,num_dense_nodes3,num_dense_nodes4,
                    num_dense_nodes5,num_dense_nodes6,num_dense_nodes7,num_dense_nodes8,num_dense_nodes9,
                    num_dense_nodes10,num_dense_nodes11,num_dense_nodes12, num_dense_nodes13, num_dense_nodes14,
                    num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                    num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                    num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13, num_dropouts14
                    ,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dense_nodes6=num_dense_nodes6,
                                 num_dense_nodes7=num_dense_nodes7,
                                 num_dense_nodes8=num_dense_nodes8,
                                 num_dense_nodes9=num_dense_nodes9,
                                 num_dense_nodes10=num_dense_nodes10,
                                 num_dense_nodes11=num_dense_nodes11,
                                 num_dense_nodes12=num_dense_nodes12,
                                 num_dense_nodes13=num_dense_nodes13,
                                 num_dense_nodes14=num_dense_nodes14,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5,
                                 num_dropouts6=num_dropouts6,
                                 num_dropouts7=num_dropouts7,
                                 num_dropouts8=num_dropouts8,
                                 num_dropouts9=num_dropouts9,
                                 num_dropouts10=num_dropouts10,
                                 num_dropouts11=num_dropouts11,
                                 num_dropouts12=num_dropouts12,
                                 num_dropouts13=num_dropouts13,
                                 num_dropouts14=num_dropouts14)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dense_nodes6,
                            num_dense_nodes7,
                            num_dense_nodes8,
                            num_dense_nodes9,
                            num_dense_nodes10,
                            num_dense_nodes11,
                            num_dense_nodes12, num_dense_nodes13, num_dense_nodes14,
                            num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                     num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                     num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13, num_dropouts14, 'none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=12,
                                    random_state=1234)
        except ValueError:
            pass
    elif lay==13:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10, num_dense_nodes11, num_dense_nodes12, num_dense_nodes13,
                         num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13
                         ):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10, num_dense_nodes11, num_dense_nodes12, num_dense_nodes13]
                l2=[  num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                      num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                      num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13
                      ]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate,activation, regularization,num_dense_nodes1,num_dense_nodes2,num_dense_nodes3,num_dense_nodes4,
                    num_dense_nodes5,num_dense_nodes6,num_dense_nodes7,num_dense_nodes8,num_dense_nodes9,
                    num_dense_nodes10,num_dense_nodes11,num_dense_nodes12, num_dense_nodes13,
                    num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                    num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                    num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13
                    ,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dense_nodes6=num_dense_nodes6,
                                 num_dense_nodes7=num_dense_nodes7,
                                 num_dense_nodes8=num_dense_nodes8,
                                 num_dense_nodes9=num_dense_nodes9,
                                 num_dense_nodes10=num_dense_nodes10,
                                 num_dense_nodes11=num_dense_nodes11,
                                 num_dense_nodes12=num_dense_nodes12,
                                 num_dense_nodes13=num_dense_nodes13,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5,
                                 num_dropouts6=num_dropouts6,
                                 num_dropouts7=num_dropouts7,
                                 num_dropouts8=num_dropouts8,
                                 num_dropouts9=num_dropouts9,
                                 num_dropouts10=num_dropouts10,
                                 num_dropouts11=num_dropouts11,
                                 num_dropouts12=num_dropouts12,
                                 num_dropouts13=num_dropouts13)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dense_nodes6,
                            num_dense_nodes7,
                            num_dense_nodes8,
                            num_dense_nodes9,
                            num_dense_nodes10,
                            num_dense_nodes11,
                            num_dense_nodes12, num_dense_nodes13,
                            num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                            num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                            num_dropouts10, num_dropouts11, num_dropouts12, num_dropouts13
                               ,'none','none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI',  # Expected Improvement.
                                        n_calls=12,
                                        random_state=1234)
        except ValueError:
            pass
        
    elif lay==12:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10, num_dense_nodes11, num_dense_nodes12,
                         num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10, num_dropouts11, num_dropouts12
                         ):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10, num_dense_nodes11, num_dense_nodes12]
                l2=[  num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                      num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                      num_dropouts10, num_dropouts11, num_dropouts12
                      ]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate,activation, regularization,num_dense_nodes1,num_dense_nodes2,num_dense_nodes3,num_dense_nodes4,
                    num_dense_nodes5,num_dense_nodes6,num_dense_nodes7,num_dense_nodes8,num_dense_nodes9,
                    num_dense_nodes10,num_dense_nodes11,num_dense_nodes12,
                    num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                    num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                    num_dropouts10, num_dropouts11, num_dropouts12
                    ,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dense_nodes6=num_dense_nodes6,
                                 num_dense_nodes7=num_dense_nodes7,
                                 num_dense_nodes8=num_dense_nodes8,
                                 num_dense_nodes9=num_dense_nodes9,
                                 num_dense_nodes10=num_dense_nodes10,
                                 num_dense_nodes11=num_dense_nodes11,
                                 num_dense_nodes12=num_dense_nodes12,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5,
                                 num_dropouts6=num_dropouts6,
                                 num_dropouts7=num_dropouts7,
                                 num_dropouts8=num_dropouts8,
                                 num_dropouts9=num_dropouts9,
                                 num_dropouts10=num_dropouts10,
                                 num_dropouts11=num_dropouts11,
                                 num_dropouts12=num_dropouts12)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dense_nodes6,
                            num_dense_nodes7,
                            num_dense_nodes8,
                            num_dense_nodes9,
                            num_dense_nodes10,
                            num_dense_nodes11,
                            num_dense_nodes12,
                            num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10, num_dropouts11, num_dropouts12,'none','none','none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI',  # Expected Improvement.
                                        n_calls=12,
                                        random_state=1234)
        except ValueError:
            pass
    elif lay==11:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10, num_dense_nodes11,
                         num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10, num_dropouts11
                         ):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10, num_dense_nodes11]
                l2 = [num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10, num_dropouts11]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4,
                    num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9,
                    num_dense_nodes10, num_dense_nodes11,num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10, num_dropouts11,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dense_nodes6=num_dense_nodes6,
                                 num_dense_nodes7=num_dense_nodes7,
                                 num_dense_nodes8=num_dense_nodes8,
                                 num_dense_nodes9=num_dense_nodes9,
                                 num_dense_nodes10=num_dense_nodes10,
                                 num_dense_nodes11=num_dense_nodes11,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5,
                                 num_dropouts6=num_dropouts6,
                                 num_dropouts7=num_dropouts7,
                                 num_dropouts8=num_dropouts8,
                                 num_dropouts9=num_dropouts9,
                                 num_dropouts10=num_dropouts10,
                                 num_dropouts11=num_dropouts11)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dense_nodes6,
                            num_dense_nodes7,
                            num_dense_nodes8,
                            num_dense_nodes9,
                            num_dense_nodes10,
                            num_dense_nodes11,num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10, num_dropouts11,'none','none','none','none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI',  # Expected Improvement.
                                        n_calls=12,
                                        random_state=1234)
        except ValueError:
            pass

    elif lay==10:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10,num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9, 
                         num_dense_nodes10]
                l2 = [num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4,
                    num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9,
                    num_dense_nodes10,num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dense_nodes6=num_dense_nodes6,
                                 num_dense_nodes7=num_dense_nodes7,
                                 num_dense_nodes8=num_dense_nodes8,
                                 num_dense_nodes9=num_dense_nodes9,
                                 num_dense_nodes10=num_dense_nodes10,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5,
                                 num_dropouts6=num_dropouts6,
                                 num_dropouts7=num_dropouts7,
                                 num_dropouts8=num_dropouts8,
                                 num_dropouts9=num_dropouts9,
                                 num_dropouts10=num_dropouts10)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dense_nodes6,
                            num_dense_nodes7,
                            num_dense_nodes8,
                            num_dense_nodes9,
                            num_dense_nodes10,num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9,
                         num_dropouts10,'none','none',accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI',  # Expected Improvement.
                                        n_calls=12,
                                        random_state=1234)
        except ValueError:
            pass
    elif lay==9:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9,
                         num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9
                         ):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9]
                l2=[  num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                      num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9
                         ]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4,
                    num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8, num_dense_nodes9,
                    num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                    num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9
                    ,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dense_nodes6=num_dense_nodes6,
                                 num_dense_nodes7=num_dense_nodes7,
                                 num_dense_nodes8=num_dense_nodes8,
                                 num_dense_nodes9=num_dense_nodes9,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5,
                                 num_dropouts6=num_dropouts6,
                                 num_dropouts7=num_dropouts7,
                                 num_dropouts8=num_dropouts8,
                                 num_dropouts9=num_dropouts9)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dense_nodes6,
                            num_dense_nodes7,
                            num_dense_nodes8,
                            num_dense_nodes9,
                            num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                            num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8, num_dropouts9
                               ,'none','none','none','none','none','none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI',  # Expected Improvement.
                                        n_calls=12,
                                        random_state=1234)
        except ValueError:
            pass
    elif lay==8:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8,
                         num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8
                         ):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8]
                l2=[  num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                      num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8
                      ]

                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4,
                    num_dense_nodes5, num_dense_nodes6, num_dense_nodes7, num_dense_nodes8,
                    num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                    num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8
                    ,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dense_nodes6=num_dense_nodes6,
                                 num_dense_nodes7=num_dense_nodes7,
                                 num_dense_nodes8=num_dense_nodes8,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5,
                                 num_dropouts6=num_dropouts6,
                                 num_dropouts7=num_dropouts7,
                                 num_dropouts8=num_dropouts8)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dense_nodes6,
                            num_dense_nodes7,
                            num_dense_nodes8,
                            num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7, num_dropouts8,'none','none','none','none',accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI',  # Expected Improvement.
                                        n_calls=12,
                                        random_state=1234)
        except ValueError:
            pass
    elif lay==7:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7,num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6, num_dropouts7):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6, num_dense_nodes7]
                l2=[  num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                      num_dropouts5, num_dropouts6, num_dropouts7
                      ]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4,
                    num_dense_nodes5, num_dense_nodes6, num_dense_nodes7,
                    num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                    num_dropouts5, num_dropouts6, num_dropouts7
                    ,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dense_nodes6=num_dense_nodes6,
                                 num_dense_nodes7=num_dense_nodes7,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5,
                                 num_dropouts6=num_dropouts6,
                                 num_dropouts7=num_dropouts7)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dense_nodes6,
                            num_dense_nodes7,
                            num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                            num_dropouts5, num_dropouts6, num_dropouts7
                               ,'none','none','none','none','none','none','none','none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI',  # Expected Improvement.
                                        n_calls=12,
                                        random_state=1234)
        except ValueError:
            pass
    elif lay==6:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6,num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5, num_dense_nodes6]
                l2 = [num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4,
                    num_dense_nodes5, num_dense_nodes6,num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dense_nodes6=num_dense_nodes6,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5,
                                 num_dropouts6=num_dropouts6)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dense_nodes6,
                            num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5, num_dropouts6,'none','none','none','none','none','none','none','none','none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI',  # Expected Improvement.
                                        n_calls=12,
                                        random_state=1234)
        except ValueError:
            pass
    elif lay==5:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5,num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4, 
                         num_dense_nodes5]
                l2 = [num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4,
                    num_dense_nodes5,num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dense_nodes5=num_dense_nodes5,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4,
                                 num_dropouts5=num_dropouts5)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dense_nodes5,
                            num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                         num_dropouts5,
                            'none','none','none','none','none','none','none','none','none','none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI',  # Expected Improvement.
                                        n_calls=12,
                                        random_state=1234)
        except ValueError:
            pass
    elif lay==4:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4,
                         num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4
                         ):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4]
                l2 = [num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3, num_dense_nodes4,
                    num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4
                    ,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dense_nodes4=num_dense_nodes4,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3,
                                 num_dropouts4=num_dropouts4)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,
                            num_dense_nodes4,
                            num_dropouts1, num_dropouts2, num_dropouts3, num_dropouts4,
                           'none','none','none','none','none','none','none','none','none','none','none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=12,
                                    random_state=1234)
        except ValueError:
            pass
    elif lay==3:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3,
                         num_dropouts1, num_dropouts2, num_dropouts3):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2, num_dense_nodes3]
                l2=[  num_dropouts1, num_dropouts2, num_dropouts3]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, activation, regularization, num_dense_nodes1, num_dense_nodes2, num_dense_nodes3,
                    num_dropouts1, num_dropouts2, num_dropouts3,num_batch_size):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dense_nodes3=num_dense_nodes3,
                                 num_dropouts1=num_dropouts1,num_dropouts2=num_dropouts2,
                                 num_dropouts3=num_dropouts3)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,
                            num_dense_nodes3,num_dropouts1, num_dropouts2, num_dropouts3,
                            'none','none','none','none','none','none','none','none','none','none','none','none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy


        default_parameters = def_para[lay]
        print(default_parameters)
        fitness(x=default_parameters)

        try:
            search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=12,
                                    random_state=1234)
        except ValueError:
            pass
    elif lay==2:
        def create_model(learning_rate, activation, regularization, num_dense_nodes1,num_dropouts1, num_dense_nodes2, num_dropouts2):
            model = Sequential()

            model.add(InputLayer(input_shape=(n_features,)))

            for i in range(lay):
                name = 'layer_dense_{0}'.format(i + 1)
                l1 = [num_dense_nodes1, num_dense_nodes2]
                l2 = [num_dropouts1, num_dropouts2]
                print(l1,l2,learning_rate,activation,regularization)
                # add dense layer
                model.add(Dense(l1[i],
                                activation=activation,
                                name=name, kernel_regularizer=regularizers.l2(regularization)))
                model.add(tf.keras.layers.Dropout(l2[i]))

            # use softmax-activation for classification.
            model.add(Dense(6, activation='softmax'))

            # Use the Adam method for training the network.
            optimizer = keras.optimizers.Adam(lr=learning_rate)

            # compile the model so it can be trained.
            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model


        @use_named_args(dimensions=dimensions)
        def fitness(learning_rate, activation, regularization,num_batch_size, num_dense_nodes1, num_dropouts1, num_dense_nodes2,num_dropouts2):
            model = create_model(learning_rate=learning_rate,
                                 activation=activation,
                                 regularization=regularization,
                                 num_dense_nodes1=num_dense_nodes1,
                                 num_dense_nodes2=num_dense_nodes2,
                                 num_dropouts1=num_dropouts1,
                                 num_dropouts2=num_dropouts2)
            callback_log = TensorBoard(
                #      log_dir=log_dir,
                histogram_freq=0,
                write_graph=True,
                write_grads=False,
                write_images=False)
            es = EarlyStopping(monitor='val_accuracy', patience=10)
            # Use Keras to train the model.
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=100,
                                batch_size=num_batch_size,
                                validation_data=(X_test, y_test),
                                callbacks=[callback_log, es])
            accuracy = history.history['val_accuracy'][-1]
            print()
            print("Accuracy: {0:.2%}".format(accuracy))
            print()
            if num_dense_nodes1==70 & num_dense_nodes2==70:
                global best_accuracy
                best_accuracy = accuracy
                out.append([accuracy])
            if accuracy > best_accuracy:
                # Save the new model to harddisk.
                model.save('home/mat/Desktop/trials')
                out.append([learning_rate, num_dense_nodes1, lay, activation, regularization,
                            num_dense_nodes2,num_dropouts1, num_dropouts2,'none','none','none','none','none','none','none','none','none','none','none','none','none',num_batch_size, accuracy])
                best_accuracy = accuracy
            del model
            K.clear_session()
            return -accuracy

        default_parameters = def_para[lay]
        fitness(x=default_parameters)
        try:
            search_result = gp_minimize(func=fitness,
                                    dimensions=dimensions,
                                    acq_func='EI',  # Expected Improvement.
                                    n_calls=12,
                                    random_state=1234)
        except ValueError:
            pass




with open('/home/mat/Desktop/cond_coop/trials/main12/main1_18_sesca6.csv', "w", newline="") as f:
    writer = csv.writer(f)
    for i in out:
        writer.writerow(i)

print(out)

end_date= datetime.now()
end_date=end_date.strftime("%d/%m/%Y %H:%M:%S")
print(start_date)
print(end_date)
print("session started" ,start_date, "and ended", end_date)