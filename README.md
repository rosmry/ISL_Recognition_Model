# Indian Sign Language (ISL) Recognition System

This project aims to develop an Indian Sign Language (ISL) recognition system that utilizes a webcam to detect and interpret ISL gestures. This system is specifically designed to recognize academic-related terms in sign language and provide directions to specific locations within the University. The system uses a deep learning model for sign language recognition and includes a user-friendly interface to deliver output as either text or audio messages, depending on the user's preference.

## Table of Contents
1. [Features](#features)
2. [Supported Signs](#supported-signs)
3. [Project Overview](#project-overview)
4. [How It Works](#how-it-works)
5. [Installation](#installation)
6. [Dataset](#dataset)
7. [Model](#model)
8. [Contact](#contact)

## Features
- **Real-Time Gesture Recognition**: Uses a webcam to capture and recognize ISL gestures in real-time.
- **Text and Speech Predictions**: Provides both text and audio outputs for recognized signs, enhancing accessibility to specific academic locations.
- **User-Friendly Interface**: Simple and intuitive UI developed with Tkinter, offering easy navigation and use.
- **Customizable Dataset**: Supports a dataset with 9 specific ISL signs, and can be extended to include more signs as needed.

## Supported Signs
The system is trained to recognize the following academic-related ISL signs:
- block1
- block2
- block3
- block4
- audi block
- r&d block
- registrar
- examination
- office
N.B: The words used for the model is specific to the University I have chosen. This model can be extended to any sign language related purpose.

## Project Overview
The project is structured to include all necessary components for ISL recognition:
1. **Data Collection**: A dataset of ISL signs, each represented by video samples.
2. **Data Preprocessing**: Video samples are preprocessed- standarized in terms of duartion, size and no.of frames.
3. **Model Training**: Deep learning models trained on the preprocessed dataset to recognize ISL gestures.
4. **User Interface**: A Tkinter-based UI that displays webcam visuals and provides real-time predictions and give output in text and speech formats.

## How It Works
1. Select the way in which the user wants the output – Text message or Audio.
2. The webcam captures the user's sign language gestures.
3. The captured video frames are processed and passed through a trained deep learning model.
4. The model predicts the corresponding ISL sign.
5. The prediction is displayed. If the prediction is wrong the user can refresh and show the gesture again.
6. Based on the prediction the output- direction to reach the location is given in the selected form – Text message and audio.
 
## Installation
Install the required dependencies using pip:
1. Tensorflow
2. OpenCV
3. Pillow
4. Joblib
5. pyttsx3

### Prerequisites
- Python 3.x
- Pip (Python package installer)

### Download the Pre trained Model
1. Download the pretrained model from this link: Click [here](models/pretrained_models/sign_language_model.h5) to view the pretrained model.
2. Place the downloaded model file in the models/pretrained_models directory.

### Create and Preprocess the Dataset
1. Create the Video Dataset: Ensure you have the necessary video samples for each sign.
2. Preprocess the Video Dataset: Standardize frames per second, duration, and resize the frames.
3. Extract Frames: Use a pretrained model (e.g., MobileNetV2) to extract frames from videos.

### Model Building and Training
1. Build and Train the Model: Train your deep learning model on the preprocessed dataset.
2. Save the Model: Save the trained model with a .h5 extension.

### Running the Application
To start the user interface, use the following command:
python src/main.py

### Using the UI
1. Open the application.
   
   ![image](https://github.com/rosmry/ISL_Recognition_Model/assets/131836824/3b3be516-0b8b-4294-b1af-1a59692cc1db)


3. Click on 'Text' for text-based results or 'Audio' for audio results.
   
4. Perform the sign in front of your webcam.
5. View the predicted output below the webcam feed.
6. The final direction output is given based on user's prefernce.

## Dataset
The dataset used for this project consists of 9 words, each represented by a folder containing video samples. The dataset is organized in the data/sign_language_dataset directory.

data/
└── sign_language_dataset/
    ├── block1/
    ├── block2/
    ├── block3/
    ├── block4/
    ├── audi_block/
    ├── r&d_block/
    ├── registrar/
    ├── examination/
    └── office/

### Adding Your Own Data
1. Create a Google Drive Directory: Open Google Drive and create a new directory. Name it something like sign_language_dataset.
Inside this directory, create subfolders for each word you want to recognize. For example, create folders named block1, block2, block3, block4, audi_block, r&d_block, registrar, examination, and office.
2. Upload Videos: Upload the corresponding videos for each sign language word into their respective folders. Ensure you have a sufficient number of videos for each word to train the model effectively.
3. Preprocess the Video Dataset: In your preprocessing code, specify the path to the Google Drive directory where your videos are stored. You might need to mount your Google Drive if you are working in a Colab notebook. Ensure your preprocessing script can read videos from this path, standardize frames per second, duration, and resize the frames as needed.
4. Store Preprocessed Videos: Create a new directory to store the preprocessed video dataset. This can be within your local project directory or another location in your Google Drive. Save the preprocessed video frames or processed data into this directory.
5. Update Preprocessing Code: Ensure your preprocessing code correctly reads videos from the input path and saves preprocessed data to the specified output path. Adjust the code to handle any specific requirements, such as converting videos to frames etc.

## Model
1. The model used for Indian Sign Language recognition is a neural network implemented using TensorFlow and Keras.
2. The labels (sign language words) are encoded into numerical values using LabelEncoder, and then converted into one-hot encoded vectors using to_categorical.
3. The dataset is split into training, testing and validation sets using train_test_split from sklearn.
4. The model is a Sequential neural network with an input layer, two dense layers as hidden layers each followed by a dropout layer to prevent overfitting, output layer with softmax activation function for multi-class classification.
5. The model is trained using the training data for 50 epochs with a batch size of 32. The validation data is used to monitor the model's performance during training.
6. The model's performance is evaluated on the test set to determine the accuracy.
7. After training, the model is saved in the models/pretrained_models directory for later use.

### Pretrained Models
If you prefer to use a pretrained model instead of training from scratch, you can download the pretrained model from [this link](models/pretrained_models/sign_language_model.h5) and place it in the `models/pretrained_models` directory. The model should be named `sign_language_model.h5`.


## Contact
Rose Mary Jose 
rosemaryjose152@gmail.com

Project Link: https://github.com/rosmry/ISL_Recognition_Model


