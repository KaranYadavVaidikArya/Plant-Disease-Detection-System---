import streamlit as st
import tensorflow as tf
import numpy as np

# Function to predict disease from the model
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page",["HOME","ABOUT","DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page",["Home"," ","Disease Recognition"])

# import Image from pillow to open images
from PIL import Image
img = Image.open("Diseases.png")

# display image using streamlit
# width is used to set the width of an image
st.image(img)

#Main Page
if(app_mode=="HOME"):
    st.markdown(
        "<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture", 
        unsafe_allow_html=True
    )
    st.markdown("""
    Welcome to the Plant Disease Detection System for Sustainable Agriculture! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** State-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless experience.
    - **Fast and Efficient:** Get results quickly, enabling informed decisions.

     ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project and our goals on the **About** page.
""")

#About Page
elif app_mode == "ABOUT":
    st.header("About")
    st.markdown("""
    ### Project Background and Motivation
    Plant diseases are a significant challenge for farmers worldwide, leading to reduced crop yields and economic losses. Early and accurate disease detection can save crops and ensure food security. This system leverages artificial intelligence to empower farmers and researchers with an easy-to-use and reliable tool for identifying plant diseases, promoting sustainable agricultural practices.

    ### Technology Stack
    - **Machine Learning Framework:** TensorFlow/Keras
    - **Web Framework:** Streamlit

    ### Model Details
    - **Architecture:** Convolutional Neural Network (CNN)
    - **Performance Metrics:** 
        - Accuracy: 95%+
        - Precision, Recall, and F1-Score available in detailed evaluation reports
    - **Training Data:** 80/20 split for training and validation, with an additional test set for predictions.

    ### Dataset Information
    - The dataset contains 87.9K RGB images of healthy and diseased crop leaves.
    - Categorized into **38 different classes**.
    - Dataset Structure:
        - **Train:** 70,295 images
        - **Validation:** 17,572 images
        - **Test:** 33 images
    """)
 
#Prediction Page
elif(app_mode=="DISEASE RECOGNITION"):
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image, width=4, use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))