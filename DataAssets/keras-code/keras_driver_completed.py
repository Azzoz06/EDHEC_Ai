"""
IBM Deep Learning (IDE) Generated Code.
Compatible Keras Version : 2.1
Tested on Python Version : 3.6.3
"""

# Import all dependencies
import os
import numpy as np
import keras
from keras.models import Model
import keras.backend as K
import keras.regularizers as R
import keras.constraints as C
from keras.layers import *
from keras.optimizers import *
import keras_helper as helper
from keras.callbacks import TensorBoard#

import json#
from os import environ#
from emetrics import EMetrics#


###############################################################################
# Set up working directories for data, model and logs.
###############################################################################
model_filename = "SignatureFraud.h5"

# writing the train model and getting input data
if environ.get('RESULT_DIR') is not None:
    output_model_folder = os.path.join(os.environ["RESULT_DIR"], "model")
    output_model_path = os.path.join(output_model_folder, model_filename)
else:
    output_model_folder = "model"
    output_model_path = os.path.join("model", model_filename)

os.makedirs(output_model_folder, exist_ok=True)


###############################################################################
# Set up HPO.
###############################################################################

config_file = "config.json"

if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        json_obj = json.load(f)
    if "initial_learning_rate" in json_obj:
        learning_rate = json_obj["initial_learning_rate"]
    else:
        learning_rate = 0.001000#
    if "batch_size" in json_obj:
        batch_size = json_obj["batch_size"]
    else:
        batch_size = 16#
    if "num_epochs" in json_obj:
        num_epochs = json_obj["num_epochs"]
    else:
        num_epochs = 10#
    if "decay" in json_obj:
        decay = json_obj["decay"]
    else:
        decay = 0.100000#
    if "beta_1" in json_obj:
        beta_1 = json_obj["beta_1"]
    else:
        beta_1 = 0.900000#
    if "beta_2" in json_obj:
        beta_2 = json_obj["beta_2"]
    else:
        beta_2 = 0.999000#
else:
    learning_rate = 0.001000#
    batch_size = 16
    num_epochs = 100
    decay = 0.100000#
    beta_1 = 0.900000#
    beta_2 = 0.999000#

def getCurrentSubID():
    if "SUBID" in os.environ:
        return os.environ["SUBID"]
    else:
        return None

class HPOMetrics(keras.callbacks.Callback):
    def __init__(self):
        self.emetrics = EMetrics.open(getCurrentSubID())

    def on_epoch_end(self, epoch, logs={}):
        train_results = {}
        test_results = {}

        for key, value in logs.items():
            if 'val_' in key:
                test_results.update({key: value})
            else:
                train_results.update({key: value})

        #print('EPOCH ' + str(epoch))
        self.emetrics.record("train", epoch, train_results)
        self.emetrics.record(EMetrics.TEST_GROUP, epoch, test_results)

    def close(self):
        self.emetrics.close()


# Perform data pre-processing
defined_metrics = []
defined_loss = []
# batch_size = 16
# num_epochs = 100


###############################################################################
# Load our data, split it, build then model and train
###############################################################################

ImageData_1_params = {
    "train_dataset": "keras-code/training_data.pickle",
    "val_dataset": "keras-code/validation_data.pickle",
    "test_dataset": "keras-code/test_data.pickle",
    "validation_split": 0.1,
    "test_split": 0.1,
    "rows": 32,
    "cols": 32,
    "dim_ordering": "channels_last",
    "dbformat": "Python Pickle",
    "num_classes": 2
}
ImageData_1_data = helper.image_data_handler(ImageData_1_params)
train_x = ImageData_1_data["train_x"]
train_y = ImageData_1_data["train_y"]
val_x = ImageData_1_data["val_x"]
val_y = ImageData_1_data["val_y"]
test_x = ImageData_1_data["test_x"]
test_y = ImageData_1_data["test_y"]
labels = ImageData_1_data["labels"]
ImageData_1_shape = train_x.shape[1:]

# Define network architecture
ImageData_1 = Input(shape=ImageData_1_shape)
Convolution2D_2 = Conv2D(
    32, (3, 3),
    data_format="channels_last",
    strides=(1, 1),
    padding="valid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="glorot_uniform",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None)(ImageData_1)
ReLU_10 = Activation("relu")(Convolution2D_2)
Convolution2D_3 = Conv2D(
    64, (3, 3),
    data_format="channels_last",
    strides=(1, 1),
    padding="valid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="glorot_uniform",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None)(ReLU_10)
ReLU_11 = Activation("relu")(Convolution2D_3)
Pooling2D_4 = MaxPooling2D(
    pool_size=(2, 3),
    strides=(1, 1),
    padding="valid",
    data_format="channels_last")(ReLU_11)
Dropout_5 = Dropout(0.25)(Pooling2D_4)
Flatten_6 = Flatten()(Dropout_5)
Dense_7 = Dense(
    128,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="glorot_uniform",
    kernel_regularizer=R.l2(0.01),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None)(Flatten_6)
Dropout_8 = Dropout(0.5)(Dense_7)
Dense_9 = Dense(
    2,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="glorot_uniform",
    kernel_regularizer=R.l2(0.01),
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None)(Dropout_8)
Softmax_12 = Activation("softmax")(Dense_9)
defined_loss = "categorical_crossentropy"
defined_metrics.append("accuracy")

model_inputs = [ImageData_1]
model_outputs = [Softmax_12]
model = Model(inputs=model_inputs, outputs=model_outputs)

# Starting Hyperparameter Optimization
hpo = HPOMetrics()

# Define optimizer
optim = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)

# Perform training and other misc. final steps
model.compile(loss=defined_loss, optimizer=optim, metrics=defined_metrics)
if len(model_outputs) > 1:
    train_y = [train_y] * len(model_outputs)
    if len(val_x) > 0: val_y = [val_y] * len(model_outputs)
    if len(test_x) > 0: test_y = [test_y] * len(model_outputs)

# Writing metrics
log_dir = os.environ.get("LOG_DIR")#
sub_id_dir = os.environ.get("SUBID")#
static_path_train = os.path.join("logs", "tb", "train")#
static_path_test = os.path.join("logs", "tb", "test")#
if log_dir is not None and sub_id_dir is not None:#
    tb_directory_train = os.path.join(log_dir, sub_id_dir, static_path_train)#
    tb_directory_test = os.path.join(log_dir, sub_id_dir, static_path_test)#

    tensorboard_train = TensorBoard(log_dir=tb_directory_train)#
    tensorboard_test = TensorBoard(log_dir=tb_directory_test)#
else:
    tb_directory_train = static_path_train#
    tb_directory_test = static_path_test#

    tensorboard_train = TensorBoard(log_dir=tb_directory_train)#
    tensorboard_test = TensorBoard(log_dir=tb_directory_test)#

if (len(val_x) > 0):
    history = model.fit(#
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(val_x, val_y),
        shuffle=True,
        callbacks=[tensorboard_train, tensorboard_test, hpo])#
else:
    history = model.fit(#
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        shuffle=True,
        callbacks=[tensorboard_train, tensorboard_test, hpo])#

hpo.close()        #

#print("Training history:" + str(history.history))#

if (len(test_x) > 0):
    test_scores = model.evaluate(test_x, test_y, verbose=1)
    #print(test_scores)
    print('Test loss:', test_scores[0])#
    print('Test accuracy:', test_scores[1])#

# if "model_result_path" not in locals() \
#  and "model_result_path" not in globals():
#     model_result_path = "./keras_model.hdf5"
model.save(output_model_path)#
print("Model saved in file: %s" % output_model_path)#
