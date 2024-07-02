import cv2
import numpy as np
import threading
import time
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib
import pyttsx3


# Initialize TTS engine
engine = pyttsx3.init()

# Load the MobileNetV2 base model for feature extraction
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Load the trained classification model
classification_model = load_model("C:/Users/HP/OneDrive/Desktop/ISL/sign_language_model.h5")

# Load the label encoder
label_encoder = joblib.load("C:/Users/HP/OneDrive/Desktop/ISL/label_encoder.pkl")

# Directions dictionary
directions_dict = {
    "Audi Block": "Go straight to the main gate, first building from the main gate.",
    "Block 1": "Just opposite to the central block you are standing.",
    "Block 2": "Go straight to the back gate direction, second building to your left.", 
    "Block 3": "Go straigh to the back gate direction, take the second left and first right.",
    "Block 4": "Go straight to the back gate direction, second building to your right.",
    "Examination": "Go to first block, opposite to central block, and the examination office is second room to the right.",
    "Office": "Go staright to the central block, inside to your front is the admission office.",
    "R&D Block": " Go straight to the back gate direction, last building to the back gate.",
    "Registrar": "Go straight from the entrance, take the first left, and the registrar office is the third room on the right."
}

# Function to extract features from a single frame
def extract_features(frame):
    img = frame.resize((224, 224))  # Resize frame to the input size expected by MobileNetV2
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = base_model.predict(img)
    features = features.squeeze()  # Remove batch dimension
    features = np.mean(features, axis=(0, 1))  # Global average pooling
    return features

# Function to predict the class of a frame
def predict_frame_class(frame):
    features = extract_features(frame)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    prediction = classification_model.predict(features)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]

# Adaptive thresholds for each gesture
adaptive_thresholds = {
    "Audi Block": 8,
    "Block 1": 8,
    "Block 2": 8, 
    "Block 3": 8,
    "Block 4": 8,
    "Examination": 2,
    "Office": 2,
    "R&D Block": 8,
    "Registrar": 8
}

# Function to show the webcam feed in a new window
def show_webcam(window, label, prediction_label, stop_event, is_speech=False):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    current_prediction = [None]
    last_prediction = [None]
    confidence_score = [0]
    prediction_start_time = [None]

    def update_frame():
        if not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_img = Image.fromarray(frame)
                frame_img = frame_img.resize((400, 300), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(image=frame_img)
                label.img_tk = img_tk
                label.config(image=img_tk)

                # Predict the class of the frame
                predicted_class = predict_frame_class(frame_img)

                if predicted_class != current_prediction[0]:
                    current_prediction[0] = predicted_class
                    prediction_start_time[0] = time.time()  # Reset the timer for new prediction
                    confidence_score[0] = 0  # Reset confidence score
                else:
                    confidence_score[0] += 1  # Increase confidence score for consistent prediction
                    threshold = adaptive_thresholds.get(predicted_class, 5)  # Default threshold is 5
                    if confidence_score[0] >= threshold:
                        last_prediction[0] = predicted_class
                        confidence_score[0] = 0  # Reset confidence score after successful prediction
                        prediction_label.config(text=f"Prediction: {predicted_class}")

                        # Get directions for the predicted class
                        directions = directions_dict.get(predicted_class, "No directions available.")
                        if is_speech:
                            engine.stop()  # Stop any ongoing speech
                            engine.say(directions)
                            engine.runAndWait()
                        else:
                            prediction_label.config(text=f"Prediction: {predicted_class}\nDirections: {directions}")

            label.after(30, update_frame)
        else:
            cap.release()
            label.destroy()

    update_frame()

# Function to close the window and stop the webcam feed
def close_window(window, stop_event):
    stop_event.set()
    window.destroy()

# Function to refresh prediction
def refresh_prediction(stop_event):
    stop_event.set()

# Main application window
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ISL Recognition")
        self.geometry("800x600")
        self.config(bg="white")

        # Add university logo at the top middle portion of the interface
        logo_image = Image.open("C:/Users/HP/OneDrive/Desktop/ISL/christ_logo.jpg")
        logo_photo = ImageTk.PhotoImage(logo_image)

        self.logo_label = Label(self, image=logo_photo, bg="white")
        self.logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
        self.logo_label.pack(side="top", pady=20)

        # Add title under the logo
        self.title_label = Label(self, text="Sign Language Detection", font=("Helvetica", 24, "bold"), bg="white")
        self.title_label.pack(side="top", pady=10)

        # Add message for output selection
        self.message_label = Label(self, text="Click any button below based on the output type you need:", font=("Helvetica", 18), bg="white")
        self.message_label.pack(side="top", pady=20)

        # Add buttons for output selection
        self.text_button = Button(self, text="Text", font=("Helvetica", 16), command=self.open_text_window, bg="lightgray")
        self.text_button.pack(side="top", pady=10)

        self.speech_button = Button(self, text="Speech", font=("Helvetica", 16), command=self.open_speech_window, bg="lightgray")
        self.speech_button.pack(side="top", pady=10)

    # Function to open the text output window
    def open_text_window(self):
        self.new_window("Text Output", "lightblue", is_speech=False)

    # Function to open the speech output window
    def open_speech_window(self):
        self.new_window("Speech Output", "lightgreen", is_speech=True)

    # Function to create a new window with webcam and prediction
    def new_window(self, title, bg_color, is_speech):
        window = tk.Toplevel(self)
        window.title(title)
        window.geometry("800x600")
        window.config(bg=bg_color)

        stop_event = threading.Event()

        # Frame for webcam visuals at the top center
        webcam_frame = Label(window, bg="black", width=400, height=300)
        webcam_frame.place(relx=0.5, rely=0.2, anchor="center")

        # Space for prediction below the webcam frame
        prediction_label = Label(window, text="Prediction: ", font=("Helvetica", 16), bg=bg_color)
        prediction_label.place(relx=0.5, rely=0.7, anchor="center")

        # Add a Refresh button
        refresh_button = Button(window, text="Refresh", command=lambda: refresh_prediction(stop_event), bg="lightgray", font=("Helvetica", 16))
        refresh_button.place(relx=0.5, rely=0.8, anchor="center")

        # Add a Close button
        close_button = Button(window, text="Close", command=lambda: close_window(window, stop_event), bg="lightgray", font=("Helvetica", 16))
        close_button.place(relx=0.5, rely=0.9, anchor="center")

        threading.Thread(target=show_webcam, args=(window, webcam_frame, prediction_label, stop_event, is_speech)).start()

if __name__ == "__main__":
    app = Application()
    app.mainloop()
