#Importing the required libraries
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.image import resize
from tensorflow.keras.models import load_model
import warnings
from sklearn.metrics import classification_report
from sklearn import metrics
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from google.colab import drive
drive.mount('/content/drive')

!unzip -q '/content/drive/MyDrive/data/Danger/Dangerc-20250324T151121Z-001.zip' -d '/content'

!unzip -q '/content/drive/MyDrive/data/Normal/Normal-20250324T151113Z-001.zip' -d '/content/'

# Defining the folder structure
data_dir = '/content'
classes = ['Dangerc', 'Normal']

# Load and preprocess audio data
def load_and_preprocess_data(data_dir, classes, target_shape=(256, 256)):
    data = []
    labels = []
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.mp3'):
                file_path = os.path.join(class_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
                mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
                data.append(mel_spectrogram)
                labels.append(i)
    return np.array(data), np.array(labels)


# Split data into training and testing sets
data, labels = load_and_preprocess_data(data_dir, classes)
if len(data) == 0:
    print("❌ No data loaded. Check file paths and dataset.")
else:
    print(f"✅ Successfully loaded {len(data)} samples.")

labels = to_categorical(labels, num_classes=len(classes))
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


# Model Creation
input_shape = X_train[0].shape
input_layer = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(len(classes), activation='softmax')(x)
model = Model(input_layer, output_layer)


from keras.utils import plot_model
plot_model(model, show_shapes=True)


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


predicted_test = model.predict(X_test)
predicted_train = model.predict(X_train)
predicted_class = []
for i in range(len(predicted_train)):
  predicted_class.append(np.argmax(predicted_train[i]))
predicted_class_index = []
for i in range(len(predicted_test)):
  predicted_class_index.append(np.argmax(predicted_test[i]))
rounded_test = np.argmax(y_test,axis=1)
rounded_train = np.argmax(y_train,axis=1)


confusion_matrix = metrics.confusion_matrix(rounded_train, predicted_class)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

cm_display.plot()
plt.show()


confusion_matrix = metrics.confusion_matrix(rounded_test, predicted_class_index)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

cm_display.plot()
plt.show()


print(classification_report(rounded_test, predicted_class_index))


model.save('audio_classification_model.h5')


# Load the saved model
model = load_model('audio_classification_model.h5')
# Define the target shape for input spectrograms
target_shape = (256, 256)
# Define your class labels
classes = ['Danger', 'Normal']


# Function to preprocess and classify an audio file
def test_audio(file_path, model):
    # Load and preprocess the audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    mel_spectrogram = tf.reshape(mel_spectrogram, (1,) + target_shape + (1,))

    # Make predictions
    predictions = model.predict(mel_spectrogram)

    # Get the class probabilities
    class_probabilities = predictions[0]

    # Get the predicted class index
    predicted_class_index = np.argmax(class_probabilities)

    return class_probabilities, predicted_class_index


# Test an audio file
test_audio_file = '/content/Dangerc/dangerc129.mp3'
class_probabilities, predicted_class_index = test_audio(test_audio_file, model)

# Display results for all classes
for i, class_label in enumerate(classes):
    probability = class_probabilities[i]
    print(f'Class: {class_label}, Probability: {probability:.4f}')


# Calculate and display the predicted class and accuracy
predicted_class = classes[predicted_class_index]
accuracy = class_probabilities[predicted_class_index]
print(f'The audio is classified as: {predicted_class}')
print(f'Accuracy: {accuracy:.4f}')
