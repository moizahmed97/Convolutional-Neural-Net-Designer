import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.python.keras.models import Model

from datasetOps.mnist import MNIST
data = MNIST(data_dir="datasetOps/data/MNIST/")

"""print("Size of:")
print("- Training-set:\t\t{}".format(data.num_train))
print("- Validation-set:\t{}".format(data.num_val))
print("- Test-set:\t\t{}".format(data.num_test))"""

# The number of pixels in each dimension of an image.
img_size = data.img_size

# The images are stored in one-dimensional arrays of this length.
img_size_flat = data.img_size_flat

# Tuple with height and width of images used to reshape arrays.
img_shape = data.img_shape

# Tuple with height, width and depth used to reshape arrays.
# This is used for reshaping in Keras.
img_shape_full = data.img_shape_full

# Number of classes, one class for each of 10 digits.
num_classes = data.num_classes

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = data.num_channels


#### HELPER FUNCTION FOR PLOTTING IMAGES
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# PLOTTING A FEW IMAGES TO Check

# Get the first images from the test-set.
images = data.x_test[0:9]

# Get the true classes for those images.
cls_true = data.y_test_cls[0:9]

# Plot the images and labels using our helper-function above.
#plot_images(images=images, cls_true=cls_true)

#### FUNCTION TO PLOT SOME ERRORS

def plot_example_errors(data,model2):
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.
    y_pred = model2.predict(x=data.x_test)

    cls_pred = np.argmax(y_pred, axis=1)
    # Boolean array whether the predicted class is incorrect.
    incorrect = (cls_pred != data.y_test_cls)

    # Get the images from the test-set that have been
    # incorrectly classified.
    images = data.x_test[incorrect]

    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = data.y_test_cls[incorrect]

    # Plot the first 9 images.
    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

#### FUNCTION TO BUILD THE MODEL

def AlexNetBuild(kernal=5,stride=1,filter=16):

    #### FUNCTION TO BUILD THE MODEL



    # Create an input layer which is similar to a feed_dict in TensorFlow.
    # Note that the input-shape must be a tuple containing the image-size.
    inputs = Input(shape=(img_size_flat,))

    # Variable used for building the Neural Network.
    net = inputs

    # The input is an image as a flattened array with 784 elements.
    # But the convolutional layers expect images with shape (28, 28, 1)
    net = Reshape(img_shape_full)(net)

    # First convolutional layer with ReLU-activation and max-pooling.
    net = Conv2D(kernel_size=kernal, strides=1, filters=16, padding='valid',
                 activation='relu', name='layer_conv1')(net)
    net = MaxPooling2D(pool_size=2, strides=2, padding='valid')(net)

    # Second convolutional layer with ReLU-activation and max-pooling.
    net = Conv2D(kernel_size=kernal, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv2')(net)
    net = MaxPooling2D(pool_size=2, strides=2)(net)

    # third convolutional layer with ReLU-activation and max-pooling.
    net = Conv2D(kernel_size=kernal, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv3')(net)

    # fourth convolutional layer with ReLU-activation and max-pooling.
    net = Conv2D(kernel_size=kernal, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv4')(net)

    # fifth convolutional layer with ReLU-activation and max-pooling.
    net = Conv2D(kernel_size=kernal, strides=1, filters=36, padding='same',
                 activation='relu', name='layer_conv5')(net)

    net = MaxPooling2D(pool_size=2, strides=2)(net)

    # Flatten the output of the conv-layer from 4-dim to 2-dim.
    net = Flatten()(net)

    # First fully-connected / dense layer with ReLU-activation.
    net = Dense(128, activation='relu')(net)

    # First fully-connected / dense layer with ReLU-activation.
    # from keras.layers import Dropout
    net = Dense(256, activation='relu')(net)
    net = Dropout(0.4)(net)

    # Last fully-connected / dense layer with softmax-activation
    # so it can be used for classification.
    net = Dense(num_classes, activation='softmax')(net)

    # Output of the Neural Network.
    outputs = net



    #####################################################

    model2 = Model(inputs=inputs, outputs=outputs)
    ###################################################

    # COMPILE THE MODEL BY PASSING THE DATA TO THE BUILT MODEL

    model2.compile(optimizer='rmsprop',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

    #TRAINING THE MODEL






    # EVALUATE THE MODEL

    return model2




def progress(model2):
    model2.fit(x=data.x_train,
               y=data.y_train,
               epochs=1, batch_size=128)
    result = model2.evaluate(x=data.x_test,
                             y=data.y_test)
    str = ""
    for name, value in zip(model2.metrics_names, result):
        str += ("%s: %f\n"%(name,value))

    str+= "{0}: {1:.2%}".format(model2.metrics_names[1], result[1])
    return str


#for name, value in zip(model2.metrics_names, result):
#    print(name, value)


#print("{0}: {1:.2%}".format(model2.metrics_names[1], result[1]))

#y_pred = model2.predict(x=data.x_test)

#cls_pred = np.argmax(y_pred, axis=1)

#plot_example_errors(cls_pred)


## FUNCTION TO PLOT THE CONVOLUTION LAYERS
def plot_conv_weights(model2, input_channel=0):
    layer_conv1 = model2.layers[2]
    weights = layer_conv1.get_weights()[0]
    # Get the lowest and highest values for the weights.
    # This is used to correct the colour intensity across
    # the images so they can be compared with each other.
    w_min = np.min(weights)
    w_max = np.max(weights)

    # Number of filters used in the conv. layer.
    num_filters = weights.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot all the filter-weights.
    for i, ax in enumerate(axes.flat):
        # Only plot the valid filter-weights.
        if i<num_filters:
            # Get the weights for the i'th filter of the input channel.
            # See new_conv_layer() for details on the format
            # of this 4-dim tensor.
            img = weights[:, :, input_channel, i]

            # Plot image.
            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

"""print(model2.summary())

layer_input = model2.layers[0]


layer_conv1 = model2.layers[2]
layer_conv2 = model2.layers[4]
weights_conv1 = layer_conv1.get_weights()[0]

#plot_conv_weights(weights=weights_conv1, input_channel=0)
weights_conv2 = layer_conv2.get_weights()[0]
"""
def plot_conv2_output(data,model2):
    from tensorflow.python.keras.models import Model
    image1 = data.x_test[0]
    layer_input = model2.layers[0]
    layer_conv2 = model2.layers[4]
    output_conv2 = Model(inputs=layer_input.input,
                         outputs=layer_conv2.output)

    values = output_conv2.predict(np.array([image1]))
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



def plot_image(data):
    image = data.x_test[0]
    plt.imshow(image.reshape(data.img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()

def plot_conv1_output(data,model2):
    from tensorflow.python.keras.models import Model
    image1 = data.x_test[0]
    layer_input = model2.layers[0]
    layer_conv1 = model2.layers[2]
    output_conv1 = Model(inputs=layer_input.input,
                         outputs=layer_conv1.output)

    values = output_conv1.predict(np.array([image1]))
    # Number of filters used in the conv. layer.
    num_filters = values.shape[3]

    # Number of grids to plot.
    # Rounded-up, square-root of the number of filters.
    num_grids = math.ceil(math.sqrt(num_filters))

    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(num_grids, num_grids)

    # Plot the output images of all the filters.
    for i, ax in enumerate(axes.flat):
        # Only plot the images for valid filters.
        if i<num_filters:
            # Get the output image of using the i'th filter.
            img = values[0, :, :, i]

            # Plot image.
            ax.imshow(img, interpolation='nearest', cmap='binary')

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()



def plot_image(data):
    image = data.x_test[0]
    plt.imshow(image.reshape(img_shape),
               interpolation='nearest',
               cmap='binary')

    plt.show()
"""image1 = data.x_test[0]
#plot_image(image1)

output_conv2 = Model(inputs=layer_input.input,
                     outputs=layer_conv2.output)

layer_output2 = output_conv2.predict(np.array([image1]))

#plot_conv_output(values=layer_output2)
"""


def plot_confusion_matrix(data,model2):
    from sklearn.metrics import confusion_matrix
    y_pred = model2.predict(x=data.x_test)

    cls_pred = np.argmax(y_pred, axis=1)
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.y_test_cls

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
