# Classification of images with CNNs
import os
import shutil
import keras
from keras.models import Sequential
from keras.layers import Conv2D,\
                         MaxPooling2D,\
                         Flatten,\
                         Dense,\
                         Dropout
from keras.preprocessing.image import ImageDataGenerator


# Data preprocess
input_size = 128 # 128x128 pixel images
n_train = 8000
n_test = 2000
n_example = 10

# Apply image transformations and create flow for training set
def get_train(batch_size):

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    train = train_datagen.flow_from_directory(
                        'datasets/training_set',
                        target_size=(input_size,input_size),
                        color_mode='rgb',
                        class_mode='binary',
                        batch_size=batch_size)
    return train

# Transformations and flow for test set
def get_test(batch_size):

    test_datagen = ImageDataGenerator(rescale=1./255)

    test = test_datagen.flow_from_directory(
                        'datasets/test_set',
                        target_size=(input_size,input_size),
                        color_mode='rgb',
                        class_mode='binary',
                        batch_size=batch_size)
    return test

# Transformations and flow for example set
def get_example():

    example_datagen = ImageDataGenerator(rescale=1./255)

    example = example_datagen.flow_from_directory(
                        'datasets/example_set',
                        target_size=(input_size,input_size),
                        color_mode='rgb',
                        class_mode='binary',
                        batch_size=n_example,
                        shuffle=False)
    return example


# Network architecture
def create_model(drop_rate=0):

    model = Sequential()

    model.add(Conv2D(64,kernel_size=3,strides=1,activation='relu',
                     kernel_initializer='truncated_normal',
                     bias_initializer=keras.initializers.Constant(0.1),
                     input_shape=(input_size,input_size,3)))
    model.add(MaxPooling2D(2,strides=2))
    model.add(Conv2D(64,kernel_size=3,strides=1,activation='relu',
                     kernel_initializer='truncated_normal',
                     bias_initializer=keras.initializers.Constant(0.1)))
    model.add(MaxPooling2D(2,strides=2))
    model.add(Conv2D(64,kernel_size=3,strides=1,activation='relu',
                     kernel_initializer='truncated_normal',
                     bias_initializer=keras.initializers.Constant(0.1)))
    model.add(MaxPooling2D(2,strides=2))
    model.add(Conv2D(64,kernel_size=3,strides=1,activation='relu',
                     kernel_initializer='truncated_normal',
                     bias_initializer=keras.initializers.Constant(0.1)))
    model.add(MaxPooling2D(2,strides=2))
    model.add(Conv2D(64,kernel_size=3,strides=1,activation='relu',
                     kernel_initializer='truncated_normal',
                     bias_initializer=keras.initializers.Constant(0.1)))
    model.add(MaxPooling2D(2,strides=2))

    model.add(Flatten())
    model.add(Dropout(drop_rate))
    model.add(Dense(128,activation='relu',
                    kernel_initializer='truncated_normal',
                    bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dropout(drop_rate))
    model.add(Dense(128,activation='relu',
                    kernel_initializer='truncated_normal',
                    bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dropout(drop_rate))
    model.add(Dense(64,activation='relu',
                    kernel_initializer='truncated_normal',
                    bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dropout(drop_rate))
    model.add(Dense(16,activation='relu',
                    kernel_initializer='truncated_normal',
                    bias_initializer=keras.initializers.Constant(0.1)))
    model.add(Dense(1,activation='sigmoid',
                    kernel_initializer='truncated_normal',
                    bias_initializer=keras.initializers.Constant(0.1)))

    return model


# Training stages parameters
epochs = lr = drop_rate = batch_size = [] # Initialized later

# Load a model then train through the next stage
def train_next_stage(stage_num, start_num):

    i = stage_num - start_num

    next_stage = create_model(drop_rate[i])
    next_stage.compile(optimizer=keras.optimizers.Adam(lr[i]),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    next_stage.load_weights('training/stage_{}'.format(stage_num))

    train = get_train(batch_size[i])
    test = get_test(batch_size[i])

    next_stage.fit_generator(train,
                             n_train/batch_size[i],
                             epochs=epochs[i],
                             validation_data=test,
                             validation_steps=n_test/batch_size[i])

    next_stage.save_weights('training/stage_{}'.format(stage_num+1))


# Train a model through all the defined training stages
def train_from_stage(start_num):

    if start_num == 0:
        model = create_model()
        model.save_weights('training/stage_0')

    # Clear the other stages
    os.rename('training','training_temp')
    os.makedirs('training')
    shutil.copyfile('training_temp/stage_{}'.format(start_num),
                    'training/stage_{}'.format(start_num))
    shutil.rmtree('training_temp')

    end_num = start_num + len(epochs)

    for i in range(start_num,end_num):
        train_next_stage(i, start_num)


# Save and load runs if needed later
def save_run(name):
    shutil.copytree('training', 'saved/'+ name)
    shutil.copy('cnn.py', 'saved/'+ name)

def load_run(name):
    shutil.rmtree('training')
    shutil.copytree('saved/%s'% name, 'training')
    os.remove('training/cnn.py')


# Make predictions on the example set
def example_predict(stage_dir):
    model = create_model()
    model.load_weights(stage_dir)
    examples = get_example()
    y_pred = model.predict_generator(examples,1)
    print(y_pred)
    print(examples.filenames)


# Define training stages and parameters
epochs = [50,10,10,10]
lr = [0.001,0.0005,0.0005,0.0003] # Decrease learning rate over time
drop_rate = [0.2,0.25,0.3,0.35] # Increase dropout over time
batch_size = [32,32,32,32]

train_from_stage(0)

# Save model for later training
save_run('1-1')

# See example set predictions
example_predict('saved/1-1/stage_4')
