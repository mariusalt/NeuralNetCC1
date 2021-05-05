# mlp for multiclass classification
from numpy import argmax
import pandas as pd
import csv
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
import kerastuner as kt
import tensorflow as tf

# load the dataset
path = '/home/mat/Desktop/sesca.csv'
df = read_csv(path)
#print(df1.head(3))
# split into input and output columns
#df = df1.drop(df1.columns[0:19], axis=1)
print(df.head(3))
df = df.drop('choice1', 1)
df = df.drop('cooptyp.1', 1)
print(list(df))
X = df.values[:, :-1]
y = df.values[:, -1]

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

def model_builder(hp):
    model = Sequential()
 #   hp_units1 = hp.Int('units', min_value=10, max_value=130, step=5)
    model.add(keras.layers.Flatten(input_shape=(n_features,)))
    hp_units1 = hp.Int('units1', min_value=10, max_value=100, step=15)
    model.add(Dense(units=hp_units1, activation='relu', kernel_initializer='he_normal'))
    hp_dropout1 = hp.Float('dropout1', 0, 0.5, step=0.5, default=0.5)
    model.add(tf.keras.layers.Dropout(hp_dropout1))
    hp_units2 = hp.Int('units2', min_value=10, max_value=100, step=15)
    model.add(Dense(units=hp_units2, activation='relu', kernel_initializer='he_normal'))
    hp_dropout2 = hp.Float('dropout2', 0, 0.5, step=0.5, default=0.5)
    model.add(tf.keras.layers.Dropout(hp_dropout2))
    hp_units3 = hp.Int('units3', min_value=10, max_value=100, step=15)
    model.add(Dense(units=hp_units3, activation='relu', kernel_initializer='he_normal'))
    hp_dropout3 = hp.Float('dropout3', 0, 0.5, step=0.5, default=0.5)
    model.add(tf.keras.layers.Dropout(hp_dropout3))
    hp_units4 = hp.Int('units4', min_value=0, max_value=100, step=20)
    model.add(Dense(units=hp_units4, activation='relu', kernel_initializer='he_normal'))
    hp_dropout4 = hp.Float('dropout4', 0, 0.5, step=0.5, default=0.5)
    model.add(tf.keras.layers.Dropout(hp_dropout4))
    hp_units5 = hp.Int('units5', min_value=0, max_value=100, step=20)
    model.add(Dense(units=hp_units5, activation='relu', kernel_initializer='he_normal'))
    hp_dropout5 = hp.Float('dropout5', 0, 0.5, step=0.5, default=0.5)
    model.add(tf.keras.layers.Dropout(hp_dropout5))
 #   hp_units4 = hp.Int('units4', min_value=10, max_value=100, step=15)
 #   model.add(Dense(units=hp_units4, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(6, activation='softmax'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3])


    model.compile(optimizer=keras.optimizers.Adam(hp_learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
out=[]
for i in range(1):
    name='tuner_round'+ str(i)
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=50,
                         factor=3,
                         hyperband_iterations=2,
                         directory='my_dir',
                         project_name=name)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)


    tuner.search(X_train, y_train, epochs=50, validation_split=0.3, callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    #print(best_hps.get(.))
    print(f"""
    learning rate:{best_hps.get('learning_rate')},layer1:{best_hps.get('units1')},layer2:{best_hps.get('units2')},layer2:{best_hps.get('units3')},layer2:{best_hps.get('units4')},layer2:{best_hps.get('units5')}, dropout1:{best_hps.get('dropout1')}, dropout2:{best_hps.get('dropout2')}, dropout3:{best_hps.get('dropout3')}, dropout4:{best_hps.get('dropout4')}.
    """)

    out.append([best_hps.get('learning_rate'),best_hps.get('units1'), best_hps.get('units2'), best_hps.get('units3'),best_hps.get('units4'),best_hps.get('units5'),best_hps.get('dropout1'),best_hps.get('dropout2'),best_hps.get('dropout3'),best_hps.get('dropout4'),best_hps.get('dropout5')])


print(out)


with open("/home/mat/Desktop/perf1.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([ 'lr','layer1','layer2','layer3','layer4','layer5','dropout1','dropout2','dropout3','dropout4','dropout5'])
    writer.writerows(out)
#,layer2:{best_hps.get('units2')}
#,layer4:{best_hps.get('units4')}
#,layer3:{best_hps.get('units3')}
#, dropout1:{best_hps.get('dropout1')}, dropout2:{best_hps.get('dropout2')}.


#es = EarlyStopping(monitor='val_acc', patience=5)

# fit the model
#history = model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=0, validation_split=0.3, callbacks=[es])
# evaluate the model
#loss, acc = model.evaluate(X_test, y_test, verbose=0)
#print('Test Accuracy: %.3f' % acc)
# make a prediction
#row = [5.1,3.5,1.4,0.2]
#yhat = model.predict([row])
#print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

#print(history.history.keys())
#pyplot.title('Learning Curves')
#pyplot.xlabel('Epoch')
#pyplot.ylabel('Cross Entropy')
#pyplot.plot(history.history['acc'], label='train')
#pyplot.plot(history.history['val_acc'], label='val')
#pyplot.legend()
#pyplot.show()



