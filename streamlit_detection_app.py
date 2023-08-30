# Python In-built packages
from pathlib import Path
import PIL
import cv2

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import pandas as pd

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 20, 100, 35)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

# Create a list to store uploaded images



source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'), accept_multiple_files=True)
    
    try:
            col1, col2 = st.columns(2)
            if not source_img:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(default_detected_image_path)
                with col1: 
                    st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
                with col2:
                    st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
            else:
    
                # Create a button to change images
                change_button = st.sidebar.button("Next Image")
                

                #col1, col2 = st.columns(2)

                num_images = len(source_img)

                # Initialize an index to keep track of the current image
                image_index = 0
                uploaded_images=[]
                detected_images = []
                cls=[]
                conf=[]
                x=[]
                y=[]
                w=[]
                h=[]
                file_name=[]
                for uploaded_image in source_img:
                    uploaded_image = PIL.Image.open(uploaded_image)
                    uploaded_images.append(uploaded_image)
                # Function to run YOLOv5 inference and display the results
                for source_image in source_img:
                    file_name.append(source_image.name)
                    uploaded_image = PIL.Image.open(source_image)
                    res = model.predict(uploaded_image,conf=confidence)
                    res_plotted = res[0].plot()[:, :, ::-1]
                    detected_images.append(res_plotted)
                    boxes = res[0].boxes
                    for box in boxes:
                        cls.append("Plastic")
                        conf.append(box.data.tolist()[0][-2])
                        box_co=box.data.tolist()[0][0:4]
                        x.append(box_co[0])
                        y.append(box_co[1])
                        w.append(box_co[2])
                        h.append(box_co[3])
                df=pd.DataFrame({'File_name':file_name,"X": x,"Y": y,"Width":w,"Height":h,"class":cls,"confidence":conf})
                # Display the original and detected images
                def display_images(original_image_path, detected_image_path):
                    with col1:
                        st.image(original_image_path, caption="Original Image", use_column_width=True)
                    
                    with col2:
                        st.image(detected_image_path, caption="Detected Image", use_column_width=True)
                        with st.expander("Detection Results"):
                            st.dataframe(df)   
                            csv_file = df.to_csv(index=False)
                            st.download_button(
                                label="Download",
                                data=csv_file,
                                file_name="data.csv",
                                key="download-csv"
                            )
                # Display the initial images
                # original_image_path = uploaded_images[image_index]
                # detected_image_path = detected_images[image_index]
                # display_images(original_image_path, detected_image_path)
                # When the button is clicked, increment the index to change images
                if change_button:
                    image_index += 1
                    if image_index >= num_images:
                        image_index = 0  # Start over if we've reached the end of the list
                        # Get paths for the new images and update the display
                original_image_path = uploaded_images[image_index]
                detected_image_path = detected_images[image_index]
                display_images(original_image_path, detected_image_path)
    except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)
    
    

    
    

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
