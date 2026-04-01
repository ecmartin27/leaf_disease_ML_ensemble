import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# define custom functions for the u-net model
#define dice loss
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

#define total loss function
def total_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

# load both models
@st.cache_resource
def load_models():
    # load classifier
    classifier = tf.keras.models.load_model('disease_classification_model.keras')
    
    # load U-Net w/ custom functions
    segmentor = tf.keras.models.load_model(
        'unet_segmentation.keras',
        custom_objects={
            'dice_coef': dice_coef,
            'dice_loss': dice_loss,
            'total_loss': total_loss
        }
    )
    return classifier, segmentor

clf_model, unet_model = load_models()
class_names = ['Healthy', 'Powdery', 'Rust']

# UI
st.title("Plant Disease Analyzer & Segmenter")
st.write("Upload a leaf image to classify the disease and see affected areas.")
st.write("**Currently the Segmentation has a lot of background noise and is unable to locate disease spots. Please only use the classification tool until this is fixed.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_input = np.expand_dims(img_array, axis=0)

        # UI layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption='Original Image', use_container_width=True)

    with col2:
        # run classification
        predictions = clf_model.predict(img_input)[0]  # grab the first result in the batch
        result_idx = np.argmax(predictions)
        result = class_names[result_idx]
        confidence = 100 * predictions[result_idx]

        st.subheader(f"Prediction: {result}")
        
        # min/max to keep progress bar between 0 and 100
        progress_val = max(0, min(int(confidence), 100))
        st.progress(progress_val)
        
        st.write(f"Confidence: {confidence:.2f}%")


    # run U-Net segmentation
    st.divider()
    
    # run prediction - will be ready if user clicks box to see the overlay
    mask_pred = unet_model.predict(img_input)
    single_mask = np.squeeze(mask_pred[0])

    # optional disease overlay
    if st.checkbox("Show Disease Overlay"):
        with st.spinner("Generating overlay..."):
            # get the single image from the batch (224, 224, 3)
            overlay = (img_input[0] * 255).astype(np.uint8).copy()
            
            # apply red color where mask probability is > 0.9
            overlay[single_mask > 0.9] = [255, 0, 0]
            
            # display the overlay in the center or a new column
            st.image(overlay, caption="Red Areas = Detected Disease", use_container_width=True)



