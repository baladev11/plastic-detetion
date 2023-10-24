from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube
import pandas as pd
import time

import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.sidebar.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.sidebar.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    #global speed
    speed=res[0].speed["inference"]
    #global boxes_len
    boxes_len=len(res[0].boxes)
    # cls=[]
    # confi=[]
    # x=[]
    # y=[]
    # w=[]
    # h=[]
    # file_name=[]
    # obj_id=[]
    # for box in res[0].boxes:
    #     file_name.append(f"frame_{count}")
    #     cls.append("Plastic")
    #     confi.append(round(box.conf.tolist()[0],2))
    #     box_co=box.xyxyn.tolist()[0]
    #     x.append(box_co[0])
    #     y.append(box_co[1])
    #     w.append(box_co[2])
    #     h.append(box_co[3])
    #     obj_id.append(box.id.tolist()[0])
    # df=pd.DataFrame({'File_name':file_name,"object_id":obj_id,"X": x,"Y": y,"Width":w,"Height":h,"class":cls,"confidence":confi})

    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    # col1, col2, col3 = st.columns(3)

    # with col1:
    #     st.header(" Inference time")
    # #     st.write(f"{speed}s")

    # with col2:
    #     st.header("Object count")
    # #     st.write(f"{boxes_len}")

    # with col3:
    #     #count=1
    #     st.header("Frame number")
    #     st.write(f"{count}")
    #     count+=1

    # st.dataframe(df)
    return [speed,boxes_len]

def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url")
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    print(speed)
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()


    col1, col2 = st.columns(2)

    with col1: 
        with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
            video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
         with col2:            
            try:
                vid_cap = cv2.VideoCapture(
                    str(settings.VIDEOS_DICT.get(source_vid)))
                st_frame = st.empty()
                count=0
                #start_time=0
                while (vid_cap.isOpened()):
                    count+=1
                    #current_time=time.time()
                    #fps=1/(current_time-start_time)
                    #start_time=current_time
                    success, image = vid_cap.read()
                    if success:
                        obj_s=round(_display_detected_frames(conf,
                                                model,
                                                st_frame,
                                                image,
                                                is_display_tracker,
                                                tracker
                                                )[0],1)
                        obj=_display_detected_frames(conf,
                                                model,
                                                st_frame,
                                                image,
                                                is_display_tracker,
                                                tracker
                                                )[1]
                        # Add custom CSS to remove column spacing
                        st.empty()
                        col1, col2, col3 = st.columns(3)
                        col1.write(f"Inference time: {obj_s}ms")
                        col2.write(f"object count: {obj}")
                        col3.write(f"Frame number: {count}")
                         # Create three columns and apply custom headings
                    else:
                        vid_cap.release()
                        break
                for i in obj_s:
                    print(i)
            except Exception as e:
                #st.sidebar.error("Error loading video: " + str(e))
                st.sidebar.write("video processed successfully")
                
            col1, col2, col3 = st.columns(3)

            with col1:
                st.header(" ")
                #st.write(f"{speed}s")

            with col2:
                st.header(" ")
                #st.write(f"{boxes_len}")

            with col3:
                #count=1
                st.header(" ")
                #st.write(f"{count}")


    
   
