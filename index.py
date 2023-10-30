import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn

# Streamlit app title and description
st.set_page_config(page_title="Cancer Predicition" ,
                    page_icon="https://cdn.the-scientist.com/assets/articleNo/70781/aImg/48671/cancer-cells-article-o.jpg")
st.title('Lung and Colon Cancer Prediction')
# Image uploader widget

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

class CNN(nn.Module):

    def __init__(self,num_classes):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(15376, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 5),
        )
        
    def forward(self, x):

        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x
def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize as per your model's requirements
    ])
    try:
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return input_tensor

    except Exception as e:
        st.write("Upload relevant image")
        return None

def predict(test):
    model= torch.load("model.pt")
    model.eval()
    output = model(test)
    pred_class = np.argmax(output)
    labels = ["Colon Adenocarcinoma","Colon Benign Tissue","Lung Adenocarcinoma","Lung Benign Tissue","Lung Squamous Cell Carcinoma"]
    # st.write("OUTPUT FROM MODEL : " , output)
    # st.write("Class: " ,pred_class)
    # st.write("Class name: " , labels[pred_class])
    return labels[pred_class]

# Display the uploaded image
if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("")

    with col2:
        st.image(image.resize((200,200)), caption='Uploaded Image', use_column_width=False)

    with col3:
        st.write("")

    tensor = preprocess(image)
    if tensor != None:
        with torch.no_grad():
            predicted_class = predict(tensor)
    
    # Display the prediction
        st.write(f'The image you uploaded seems to be of : {predicted_class}')
    # Optionally, you can display the uploaded image as well
else:
    st.write('No image uploaded yet.')