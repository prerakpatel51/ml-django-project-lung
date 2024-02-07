# from django.shortcuts import render
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet50 import preprocess_input

# # Load the saved model
# model = load_model('model/small_pneumonia_model3.h5')

# # Preprocess input images
# def preprocess_image(image):
#     image = cv2.resize(image, (224, 224))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format if needed
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#     return image

# def predict_pneumonia(request):
#     if request.method == 'POST' and request.FILES['image']:
#         # Get the uploaded image from the request
#         uploaded_image = request.FILES['image']
        
#         # Read the image using OpenCV
#         image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), -1)
        
#         # Preprocess the image
#         preprocessed_image = preprocess_image(image)
        
#         # Make predictions
#         predictions = model.predict(preprocessed_image)
        
#         # Decode predictions
#         predicted_class = np.argmax(predictions)
#         class_names = ['Normal', 'Pneumonia']
#         predicted_label = class_names[predicted_class]

#         return render(request, 'lungs_app/result.html', {'predicted_label': predicted_label})
#     return render(request, 'lungs_app/upload.html')












from django.conf import settings

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the saved model
model = load_model('model/small_pneumonia_model3.h5')

# Preprocess input images
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format if needed
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def predict_pneumonia(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            # Get the uploaded image from the request
            uploaded_image = request.FILES['image']
            
            # Save the uploaded image to a temporary location
            fs = FileSystemStorage()
            filename = fs.save(uploaded_image.name, uploaded_image)
            uploaded_image_url = fs.url(filename)
            
            # Read the image using OpenCV
            image = cv2.imread(os.path.join(settings.MEDIA_ROOT, uploaded_image_url[1:]))  # Remove the leading slash from the URL
            
            # Preprocess the image
            preprocessed_image = preprocess_image(image)
            
            # Make predictions
            predictions = model.predict(preprocessed_image)
            
            # Decode predictions
            predicted_class = np.argmax(predictions)
            class_names = ['Normal', 'Pneumonia']
            predicted_label = class_names[predicted_class]

            # Delete the temporary uploaded image
            os.remove(os.path.join(settings.MEDIA_ROOT, filename))

            return render(request, 'lungs_app/result.html', {'uploaded_image_url': uploaded_image_url, 'predicted_label': predicted_label})
    return render(request, 'lungs_app/upload.html')
