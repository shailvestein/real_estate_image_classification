
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import Conv2D, Input, Dense, GlobalAveragePooling2D, MaxPool2D, Dropout

from tensorflow.keras.models import Model

#### Constants
H, W = 224, 224
INPUT_SHAPE = (H,W,3)
TARGET_SHAPE = (H,W)
NUM_CLASSES = 6

def load_vectorizer(path='ohe_vectorizer.pkl'):
    with open(path, 'rb') as f:
        ohe_vectorizer = pkl.load(f)
    return ohe_vectorizer
    
def load_class_labels_list(path='class_list.pkl'):
    with open(path, 'rb') as f:
        class_list = pkl.load(f)
    return class_list

class_list = load_class_labels_list()
def to_numerical(x):
    """this function will return the index of class list after seeing the x in it"""
    return class_list.index(x)
 
#############################################################################
#############################################################################

# source: https://ogeek.cn/qa/?qa=903624/
def get_enhanced_image(img, clipLimit=2, tileGridSize=(8,8)):
    """tis function will return enhanced image"""
    # converting image color channel
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    # spliting image channel wise
    lab_planes = cv.split(lab)
    # Createing clahe object 
    clahe = cv.createCLAHE(clipLimit=clipLimit,tileGridSize=tileGridSize)
    # applying clahe to the splited image
    lab_planes[0] = clahe.apply(lab_planes[0])
    # Merging image channel wise
    lab = cv.merge(lab_planes)
    # again converting clahe images color channel to RGB
    bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    # return colored clahe image
    return bgr
    
#############################################################################
#############################################################################
    
## custom image datagenerator
class Loader:
    """this function will load image from memory and label from the given list and returns"""
    def __init__(self, X, y, target_shape, preprocess_func=None):
        # initialising attributes
        self.X = X
        self.y = y
        # the paths of images
        self.images   = self.X
        # the paths of segmentation images
        self.labels    = self.y
        # target shape
        self.target_shape = target_shape
        # preprocessing_func
        self.preprocess_func = preprocess_func

    
    def __getitem__(self, i):
        """this function will read and return image"""
        # reading image
        image = cv.imread(self.images[i], cv.IMREAD_UNCHANGED)
        # resizing image
        image = cv.resize(image, self.target_shape, cv.INTER_AREA)
        # enhancing image
        image = get_enhanced_image(image)
        
        # loading labels from the disk
        label  = self.labels[i]
        # setting condition for the preprocessing function
        if self.preprocess_func:
            # applying preprocessing function passed to the loader
            image = self.preprocess_func(image)

        # returning image and loader
        return image, label

    # returning length of the X
    def __len__(self):
        return len(self.X)
    
## Generator class
class Generator(tf.keras.utils.Sequence):   
    """this function will generate image with their respected labels in batches"""
    # defining class generator attributes
    def __init__(self, dataset, batch_size, shuffle=False):
        # defining dataset attribute
        self.dataset = dataset
        # defining batch size attribute
        self.batch_size = batch_size
        # defining shuffle attribute, to shuffle the data
        self.shuffle = shuffle
        # defining the array contains random number 
        self.indexes = np.arange(len(dataset))

    def __getitem__(self, i):
        # collect batch data
        # setting start and stop index
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        # empty list to store images and labels after loading
        images, labels = list(), list()
        for j in range(start, stop):
            # for each number between start and stop
            # append image and label into the list
            images.append(self.dataset[j][0])
            labels.append(self.dataset[j][1])
        # staking returned images to follow, (batch_size, height, width, channels) fashion for image data 
        image_batch = [np.stack(data[0], axis=0) for data in zip(images, labels)]
        # staking returned labels to follow, (batch_size, labels) fashion for image data
        label_batch = [np.stack(data[1], axis=0) for data in zip(images, labels)]
        # returning batch data after converting them into numpy array
        return (np.array(image_batch), np.array(label_batch))
    # this will return the steps per epoch
    def __len__(self):
        return len(self.indexes) // self.batch_size
    # on the epoch end shuffling data if it is set to True
    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
            
#############################################################################
#############################################################################
            


def initialise_custom_datagenerator(preprocess_func, train, y_train, val, y_val, BATCH_SIZE=1):
    # initialising data loader object
    train_loader = Loader(train['paths'].values, y_train, TARGET_SHAPE, preprocess_func=preprocess_func)
    val_loader = Loader(val['paths'].values, y_val, TARGET_SHAPE, preprocess_func=preprocess_func)
    # initialising custome data generator
    train_data_generator = Generator(train_loader, batch_size=BATCH_SIZE)
    val_data_generator = Generator(val_loader, batch_size=BATCH_SIZE)
    print(f"Custome data generator initialised successfully!")
    return train_data_generator, val_data_generator

#############################################################################
#############################################################################

def get_test_sample_predictions(data, model, preprocess_func,  target_shape, sample=5):
    """this function will plot the sample images from validation data
       with their actual and predicted label"""
    # generating random index of sample length
    rand_num = np.random.randint(0, len(data['paths']), sample)
    # for each index
    for i in rand_num:
        # spliting image path and image label
        path, label = data['paths'].values[i], data['labels'].values[i]
        # reading image from disk
        image = cv.imread(path, cv.IMREAD_UNCHANGED)
        # resizing image to the target shape
        image = cv.resize(image, target_shape, cv.INTER_AREA)
        # getting enhanced image
        image = get_enhanced_image(image)
        # applying preprocess function that was passed
        image_ = preprocess_func(image)
        # making image in standard (batch_size, height, width, channel) fashion 
        # by expanding its axis 
        image_ = tf.expand_dims(image_, axis=0)
        # prediction class label and taking argmax of it
        y_pred = np.argmax(model.predict(image_))
        # getting predicted image label from the class list
        pred_label = class_list[y_pred]
        # ploting image with its actual and predicted label
        plt.imshow(image)
        # setting actual label as plot title
        plt.title(f'Actual class: {class_list[label]}')
        # and predicted label as xlabel
        plt.xlabel(f"Predicted class: {pred_label}")
        plt.show()
        print("\n\n--------------------------------------------\n\n")
     
    
#############################################################################
#############################################################################

def plot_confusion_matrix(model, data_generator, name='model_name_here'):
    # empty list to store predicted and actual lables
    y_predicted, y_actual = list(), list()
    # for each batch in validation data
    for i in data_generator:
        # spliting image data and its label from generator
        image, label = i[0], i[1]
        # predicting y on validation set
        predicted = model.predict(image)
        # appending actual and predicted labels to the list
        for a_label, p_label in zip(label, predicted):
            y_predicted.append(p_label)
            y_actual.append(a_label)

    # taking argmax of each predicted and actual labels
    y_predicted = np.array([np.argmax(i) for i in y_predicted])
    y_actual = np.array([np.argmax(i) for i in y_actual])
    # calculating confusion matrix
    conf_mat = tf.math.confusion_matrix(y_actual, y_predicted, num_classes=6, )
    
    # ploting confusion matrix using seaborn heatmap
    # inititalising plot
    fig, ax = plt.subplots()
    # ploting heatmap
    ax = sns.heatmap(conf_mat, annot=True, fmt='d')
    # setting title to the plot
    ax.set_title(f"Confusion matrix of {name} on val data")
    # seting x labels and y label
    ax.set_ylabel('Actual labels')
    ax.set_xlabel('Predicted labels')
    # seting xticklabels
    ax.set_xticklabels(class_list)
    # seting yticklabels
    ax.set_yticklabels(class_list)
    # rotating xticks and yticks
    plt.xticks(rotation = 90)
    plt.yticks(rotation = 0)
    plt.show()

    
#############################################################################
#############################################################################

from sklearn.metrics import f1_score
# difining class to calculate f1 score on validation data
class F1_score(tf.keras.callbacks.Callback):
    """this will return f1-score for multi labels classes"""
    # declaring class attributes
    def __init__(self, data_generator):
        super().__init__()
        self.val_data = data_generator
    # when training begins append val_f1_score in logs to store f1_score
    def on_train_begin(self, logs={}):
        logs['val_f1_score'] = float('-inf')
    
    def on_epoch_end(self, epoch, logs={}):
        """f1 score calculation on validation set"""
        # empty list to store actual and predicted class labels
        y_predicted, y_actual = list(), list()
        # for each batch in data data
        for i in self.val_data:
            # spliting image data and actual labels
            image, label = i[0], i[1]
            # predicting labels on batch of image data
            predicted = self.model.predict(image)
            # for each label in batch predictions
            for a_label, p_label in zip(label, predicted):
                # appending predicted label to the list
                y_predicted.append(p_label)
                # apending actual label to the list
                y_actual.append(a_label)

        # reshaping into 1-D list after taking argmax for each class label
        # in order to feed in sklearn metrics
        y_preds = np.array([np.argmax(i) for i in y_predicted]).reshape(-1,1)
        y_actual = np.array([np.argmax(i) for i in y_actual]).reshape(-1,1)

        # calculating f1 score
        f1score = f1_score(y_actual, y_preds, average='weighted', )
        # adding val_f1_Score to the logs
        logs['val_f1_score'] = f1score

        print(f'      [==============================]  - val_f1_score: {round(f1score, 4)}')
        
#############################################################################
#############################################################################
        
        
def plot_evaluation_metrics(history, name='model'):
    """this will plot line plot between the metrics of model"""
    # defining loss, accuracy list for train and val data
    train_loss, val_loss = history['loss'], history['val_loss']
    train_accuracy, val_accuracy = history['accuracy'], history['val_accuracy']
    # val f1_score
    val_f1_score = history['val_f1_score']
    # list of epochs
    epochs = [i+1 for i in range(len(val_f1_score))]
    
    # initialising matplotlib.pyplot plot with 3 subplots
    print(f"""Evaluation metrics plot for "{name}" """)
    fig, [ax1,ax2,ax3] = plt.subplots(nrows=3, ncols=1, figsize=(8,16))
    # ploting line plot for train and val loss
    ax1.plot(epochs, train_loss, label='Train loss')
    ax1.plot(epochs, val_loss, label='Val loss')
    # adding legend
    ax1.legend()
    # setting min train and val loss as subplot-1 title
    ax1.set_title(f'Min train_loss: {round(min(train_loss), 4)}, Min val_loss: {round(min(val_loss), 4)}')
    # setting labels
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.grid()

    # ploting line plot for train and val accuracy
    ax2.plot(epochs, train_accuracy, label='Train accuracy')
    ax2.plot(epochs, val_accuracy, label='Val accuracy')
    # adding legend to the subplot
    ax2.legend()
    # setting Max train and val accuracy as title for the subplot 2
    ax2.set_title(f'Max train_acc: {round(max(train_accuracy), 4)}, Max val_acc: {round(max(val_accuracy), 4)}')
    # seting labels to the subplot
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    ax2.grid()
        
    ax3.plot(epochs, val_f1_score, label='Val f1_score')
    # adding legend to the subplot
    ax3.legend()
    # setting Max val f1_Score as title for the subplot 3
    ax3.set_title(f'Max val_f1_score: {round(max(val_f1_score), 4)}')
    # seting labels to the subplot
    ax3.set_xlabel('epochs')
    ax3.set_ylabel('f1_score')
    ax3.grid()
    
    # ploting the figure
    plt.show()
    
    
#############################################################################
#############################################################################


def initialise_callbacks(data_generator, checkpoint_model_name):
    # initialising custom f1_score object
    f1score = F1_score(data_generator)
    # early stop call back
    earlystop =  tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', 
                                              patience=5, 
                                              min_delta=0.02,
                                              mode='max',
                                              verbose=1, 
                                              restore_best_weights=True),
    # model checkpoint call back
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_model_name,
                                                monitor='val_f1_score', 
                                                mode='max',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=True)
    print(f"""Custom callbacks initialised successfully!\nModel checkpoint name will be: "{checkpoint_model_name}" """)
    return [f1score, earlystop, checkpoint]

#############################################################################
#############################################################################


def get_model_ready():
    efn = EfficientNetB0(include_top=False, 
                           weights='imagenet', 
                           input_shape=INPUT_SHAPE)

    for layers in efn.layers:
        layers.trainable=False

    # input layer
    inputs = Input(shape=INPUT_SHAPE, name='input_shape')
    # passing input layer to resnet 50
    x = efn(inputs)
    # Global pulling layer
    x = GlobalAveragePooling2D(name='global_pooling')(x)
    # Dense layer
    x = Dense(1024, activation='relu', name='dense_1')(x)
    # Dropout layer
    x = Dropout(0.5, name='dropout_1')(x)
    # Dense layer
    x = Dense(64, activation='relu', name='dense_2')(x)
    # Dropout layer
    x = Dropout(0.5, name='dropout_2')(x)
    # output layer
    outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    # intialising model
    efficientNet_model = Model(inputs=inputs, outputs=outputs, name='efficientNet_based_model')

    return efficientNet_model




#############################################################################
#############################################################################








#############################################################################
#############################################################################









#############################################################################
#############################################################################