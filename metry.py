import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

'''
To enhance the performance of the model, we can:
1- Use data augmentation to artificially increase the size of the training set.
2- Implement dropout regularization to prevent overfitting.
3- Use early stopping to prevent overfitting.
'''

# Function to introduce noise
def introduce_noise(y, noise_level, noise_type='symmetric'):
    y_noisy = np.copy(y)
    total_classes = len(np.unique(y))
    
    if noise_type == 'symmetric':
        indices = np.random.choice(np.arange(y.shape[0]), size=int(noise_level * y.shape[0]))
        y_noisy[indices] = np.random.randint(0, total_classes, size=len(indices))
        
    elif noise_type == 'asymmetric':
        flip_dict = {0:1, 2:0, 4:7, 3:5, 1:3}  # truck to automobile, bird to airplane, etc.
        for source_class, target_class in flip_dict.items():
            indices = np.where(y == source_class)[0]
            flip_indices = np.random.choice(indices, size=int(noise_level * len(indices)))
            y_noisy[flip_indices] = target_class
            
    return y_noisy

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize input
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a simple CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define noise levels
noise_levels = [0.1, 0.3, 0.5, 0.8, 0.9]

for noise_level in noise_levels:
    print(f'Training with {noise_level * 100}% noise level')
    
    # Introduce noise
    y_train_noisy = introduce_noise(y_train, noise_level, 'symmetric')
    
    # Shuffle the data to ensure randomization
    x_train, y_train_noisy = shuffle(x_train, y_train_noisy, random_state=0)
    
    # Train the model
    model.fit(x_train, y_train_noisy, epochs=10, validation_data=(x_test, y_test))
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Accuracy: {accuracy * 100}%')

# *******************************************************************************************************************

# Function to introduce noise
def introduce_noise(y, noise_level, noise_type='symmetric'):
    y_noisy = np.copy(y)
    total_classes = len(np.unique(y))
    
    if noise_type == 'symmetric':
        indices = np.random.choice(np.arange(y.shape[0]), size=int(noise_level * y.shape[0]))
        y_noisy[indices] = np.random.randint(0, total_classes, size=len(indices))
        
    elif noise_type == 'asymmetric':
        flip_dict = {0:1, 2:0, 4:7, 3:5, 1:3}
        for source_class, target_class in flip_dict.items():
            indices = np.where(y == source_class)[0]
            flip_indices = np.random.choice(indices, size=int(noise_level * len(indices)))
            y_noisy[flip_indices] = target_class
            
    return y_noisy

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize input
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define a more complex CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Generate augmented data
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

datagen.fit(x_train)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Define noise levels
noise_levels = [0.1, 0.3, 0.5, 0.8, 0.9]

for noise_level in noise_levels:
    print(f'Training with {noise_level * 100}% noise level')
    
    # Introduce noise
    y_train_noisy = introduce_noise(y_train, noise_level, 'symmetric')
    
    # Shuffle the data to ensure randomization
    x_train, y_train_noisy = shuffle(x_train, y_train_noisy, random_state=0)
    
    # Fit the model on the batches generated by datagen.flow().
    model.fit(datagen.flow(x_train, y_train_noisy, batch_size=32),
              epochs=100,
              validation_data=(x_test, y_test),
              callbacks=[early_stopping])
    
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Accuracy: {accuracy * 100}%')
