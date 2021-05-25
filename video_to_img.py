import cv2
import pathlib


def vtoimg(exercise):
    vidcap = cv2.VideoCapture("Data/" + exercise + "/video/trainer.mp4")
    success,image = vidcap.read()
    print(success)
    count = 0
    while success:
      cv2.imwrite("Data/" + exercise + "/frames/frame%d.jpg" % count, image)     # save frame as JPEG file
      success, image = vidcap.read()
      #print('Read a new frame: ', success)
      count += 1

exercise = "Side_Lateral_Raise"
vtoimg(exercise)
