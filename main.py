import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction


def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model('trained_model.keras')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return -1

    image = tf.keras.preprocessing.image.load_img(
        test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


# Sidebar
st.sidebar.title('Dashboard')
app_mode = st.sidebar.selectbox(
    "Select Page", ["Home", "About", "Disease Recognition"]
)

# Home Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "https://images.pexels.com/photos/1072824/pexels-photo-1072824.jpeg?cs=srgb&dl=pexels-akilmazumder-1072824.jpg&fm=jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
Welcome to the Plant Disease Recognition System!

Our mission is to help identify plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

### How It Works
1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant showing signs of disease.
2. **Analysis:** Our system uses advanced deep learning models to identify potential diseases.
3. **Result:** View the results along with recommendations.

### Why Choose Us?
- **Accuracy:** State-of-the-art machine learning ensures reliable results.
- **Speed:** Fast predictions for timely agricultural decisions.

### Get Started
Navigate to the **Disease Recognition** page to upload your image and begin.

### About Us
Learn more about this project and the team behind it on the **About** page.
""")

# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo. This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.

#### Content
1. Train (70,295 images)  
2. Valid (17,572 images)  
3. Test (33 images)
""")

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("DISEASE RECOGNITION")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        if st.button("Predict Disease"):
            st.image(test_image, use_container_width=True)
            with st.spinner("Please Wait..."):
                result_index = model_prediction(test_image)

                if result_index == -1:
                    st.error("Prediction failed due to model error.", icon="❌")
                else:
                    class_name = [
                        'Apple___Apple_scab',
                        'Apple___Black_rot',
                        'Apple___Cedar_apple_rust',
                        'Apple___healthy',
                        'Blueberry___healthy',
                        'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy',
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_',
                        'Corn_(maize)___Northern_Leaf_Blight',
                        'Corn_(maize)___healthy',
                        'Grape___Black_rot',
                        'Grape___Esca_(Black_Measles)',
                        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy',
                        'Orange___Haunglongbing_(Citrus_greening)',
                        'Peach___Bacterial_spot',
                        'Peach___healthy',
                        'Pepper,_bell___Bacterial_spot',
                        'Pepper,_bell___healthy',
                        'Potato___Early_blight',
                        'Potato___Late_blight',
                        'Potato___healthy',
                        'Raspberry___healthy',
                        'Soybean___healthy',
                        'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch',
                        'Strawberry___healthy',
                        'Tomato___Bacterial_spot',
                        'Tomato___Early_blight',
                        'Tomato___Late_blight',
                        'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot',
                        'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot',
                        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                        'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]
                    st.success(
                        "Model is predicting it's a **{}**.".format(class_name[result_index]), icon="✅")
