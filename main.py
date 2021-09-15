import streamlit as st
from fitness_analysis import *
import cv2

def start(images):
    bootstrap_images_in_folder = './files/test/frames'

    # Output folders for bootstrapped images and CSVs.
    bootstrap_images_out_folder = './files/test/poses'

    # Initialize helper.
    bootstrap_helper = BootstrapHelper(
        images_in_folder=bootstrap_images_in_folder,
        images_out_folder=bootstrap_images_out_folder,
        images = images
        )

    # Bootstrap all images.
    # Set limit to some small number for debug.
    bootstrap_helper.bootstrap(per_pose_class_limit=None)

def vtoimg(file):
    with st.spinner('Επεξεργασία αρχείου...'):
        with open("temp.mp4", "wb") as outfile:
            # Copy the BytesIO stream to the output file
            outfile.write(file.read())
        images = []
        vidcap = cv2.VideoCapture("temp.mp4")
        success,image = vidcap.read()
        print(success)
        count = 0
        while success:
          images.append(image)
          success, image = vidcap.read()
          #print('Read a new frame: ', success)
          count += 1
    return images

def streamlit_init():
    st.title('Ανάλυση Αθλητικής Δραστηριότητας')
    rule_option = st.selectbox(
    'Επιλογή άσκησης για έλεγχο:',
     ['Πλάγιες εκτάσεις'])
    uploaded_file = st.file_uploader("Επιλογή αρχείου", help="Αρχείο βίντεο της άσκησης.")

    if st.button('Υπολογισμός'):
        if uploaded_file is not None:
            images = vtoimg(uploaded_file)
        start(images)

streamlit_init()
