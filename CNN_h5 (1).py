import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np


# Define the path to the folder containing the categories
data_dir = r"C:\Amrita\Deep Learning\DATA_LABELS"

# Retrieve the category names
categories = os.listdir(data_dir)

# # Load the saved model
# loaded_model = load_model(r"C:\Amrita\Deep Learning\Codes\CNN\trained_model.h5")
# # Load and preprocess the image
# image_path = r"C:\Amrita\DL\ma\cropped_img68_bottom_character_62.jpg"  # Replace with the path to your image
# img = image.load_img(image_path, target_size=(100, 100), grayscale=True)
# img_array = image.img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array /= 255.0

# # Make predictions on the image
# predictions = loaded_model.predict(img_array)

# # Process the predictions
# predicted_class_index = np.argmax(predictions, axis=1)[0]
# predicted_category = categories[predicted_class_index]

# print("Predicted category:", predicted_category)






















# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing import image
# from PIL import Image
# import numpy as np
# import json

# # Load the model architecture from the JSON file
# with open(r"C:\Amrita\Deep Learning\Codes\CNN\model.json", "r") as json_file:
#     loaded_model_json = json_file.read()

# loaded_model = model_from_json(loaded_model_json)

# # Load the model weights from the h5 file
# loaded_model.load_weights(r"C:\Amrita\Deep Learning\Codes\CNN\model.h5")

# # Load and preprocess the image
# image_path = r"C:\Amrita\DL\ma\cropped_img68_bottom_character_62.jpg"  # Replace with the path to your image
# img = Image.open(image_path).convert("L")  # Convert image to grayscale
# img = img.resize((100, 100))
# img_array = np.asarray(img) / 255.0
# img_array = np.expand_dims(img_array, axis=0)
# img_array = np.expand_dims(img_array, axis=3)

# # Make predictions on the image
# predictions = loaded_model.predict(img_array)

# # Process the predictions
# predicted_class_index = np.argmax(predictions, axis=1)[0]
# predicted_category = categories[predicted_class_index]

# print("Predicted category:", predicted_category)






























from tkinter import *
from tkinter import filedialog
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model architecture from the JSON file
with open(r"C:\Amrita\Deep Learning\Codes\CNN\model.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)

# Load the model weights from the h5 file
loaded_model.load_weights(r"C:\Amrita\Deep Learning\Codes\CNN\model.h5")

# Create a Tkinter window
window = Tk()

# Define a function to handle image selection and prediction
def predict_image():
    # Open a file dialog for image selection
    image_path = filedialog.askopenfilename()
    
    # Load and preprocess the image
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(100, 100), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    
    # Make predictions on the image
    predictions = loaded_model.predict(img_array)
    
    # Process the predictions
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_category = categories[predicted_class_index]
    
    # Display the predicted category in the GUI
    result_label.config(text="Predicted category: " + predicted_category)

# Create a button for image selection
select_button = Button(window, text="Select Image", command=predict_image)
select_button.pack()

# Create a label for displaying the prediction result
result_label = Label(window, text="Predicted category: ")
result_label.pack()

# Run the Tkinter main loop
window.mainloop()
