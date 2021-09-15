# -*- coding: utf-8 -*-
from FullBodyPoseEmbedder import *
import streamlit as st
import numpy as np
import cv2, os, sys, tqdm
#from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


class PoseSample(object):

  def __init__(self, index, landmarks, embedding):
    self.index = index
    self.landmarks = landmarks
    self.embedding = embedding

class BootstrapHelper(object):
  """Helps to bootstrap images and filter pose samples for classification."""

  def __init__(self,
               images_in_folder,
               images_out_folder,
               images):
    self._images_in_folder = images_in_folder
    self._images_out_folder = images_out_folder
    self._images = images

    # Get list of pose classes and print image statistics.
    self._pose_class_names = ["Side_Lateral_Raise"]

  def bootstrap(self, per_pose_class_limit=None):
    """Bootstraps images in a given folder.

    Required image in folder (same use for image out folder):
      pushups_up/
        image_001.jpg
        image_002.jpg
        ...
      pushups_down/
        image_001.jpg
        image_002.jpg
        ...
      ...
    """

    for pose_class_name in self._pose_class_names:
      print('Bootstrapping ', pose_class_name, file=sys.stderr)

      # Paths for the pose class.
      images_in_folder = self._images_in_folder
      images_out_folder = self._images_out_folder
      images = self._images
      #if not os.path.exists(images_out_folder):
        #os.makedirs(images_out_folder)


      # Get list of images.
      #image_names = sorted([n for n in os.listdir(images_in_folder) if not n.startswith('.')])
      if per_pose_class_limit is not None:
        #image_names = image_names[:per_pose_class_limit]
        #image_names = image_names[::5]
        images = images[:per_pose_class_limit]
        images = images[::5]

      # Bootstrap every image.
      mistakes = []
      all_frames = []
      first_angles = []
      second_angles = []
      mistakes_count = 0

      # Initialize progress bar.
      progress_bar = st.progress(0)
      count = 0
      i = 0
      total_frames = len(images)
      step = 1/total_frames

      with mp_pose.Pose(model_complexity=1) as pose_tracker:
        for image in tqdm.tqdm(images):
          # Load image.
          #input_frame = cv2.imread(os.path.join(images_in_folder, image_name))
          input_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

          # Initialize fresh pose tracker and run i.
          result = pose_tracker.process(image=input_frame)
          pose_landmarks = result.pose_landmarks

          # Save image with pose prediction (if pose was detected).
          output_frame = input_frame.copy()
          if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
              image=output_frame,
              landmark_list=pose_landmarks,
              connections=mp_pose.POSE_CONNECTIONS)
          output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
          #cv2.imwrite(os.path.join(images_out_folder, image_name), output_frame)

          if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array(
                [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                 for lmk in pose_landmarks.landmark],
                dtype=np.float32)
            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

          # Draw XZ projection and concatenate with the image.
          projection_xz = self._draw_xz_projection(
            output_frame=output_frame, pose_landmarks=pose_landmarks)
          #output_frame = np.concatenate((output_frame, projection_xz), axis=1)


          frame_checked = self._check_rule(pose_landmarks, i)
          all_frames.append(frame_checked[1])
          first_angles.append(frame_checked[2])
          second_angles.append(frame_checked[3])

          # Check diff rule
          if frame_checked[0] is not None:
            mistakes.append(frame_checked[0])
            mistakes_count += 1
          # Check max angle rule
          if (frame_checked[2] > 100) or (frame_checked[3] > 100):
            mistakes_count += 1

          # Update progress bar.
          count = min(count + step, 1)
          progress_bar.progress(count)
          i += 1
      progress_bar.empty()

      #print('\n', mistakes, '\n')
      fig, (ax1, ax2) = plt.subplots(1, 2)
      ax1.set_xlabel("Αριθμός εικόνας")
      ax1.set_ylabel("Μοίρες")
      ax1.set_title("Γωνία αριστερού άνω άκρου")
      ax1.plot(first_angles)
      ax2.set_title("Γωνία δεξιού άνω άκρου")
      ax2.set_xlabel("Αριθμός εικόνας")
      ax2.plot(second_angles)
      plt.tight_layout()
      st.pyplot(fig)
      fig2, ax = plt.subplots()
      plt.title('Διαφορά γωνιών άνω άκρων:')
      plt.xlabel("Αριθμός εικόνας")
      plt.ylabel("Μοίρες")
      ax.plot(all_frames)
      st.pyplot(fig2)

      if mistakes_count!=0:
        self._show_results(mistakes, images, len(all_frames), mistakes_count)
      else:
        st.metric("Τελική βαθμολογία", "100%")


  def _draw_xz_projection(self, output_frame, pose_landmarks, r=0.5, color='red'):
    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
    img = Image.new('RGB', (frame_width, frame_height), color='white')

    if pose_landmarks is None:
      return np.asarray(img)

    # Scale radius according to the image width.
    r *= frame_width * 0.01

    draw = ImageDraw.Draw(img)
    for idx_1, idx_2 in mp_pose.POSE_CONNECTIONS:
      # Flip Z and move hips center to the center of the image.
      x1, y1, z1 = pose_landmarks[idx_1] * [1, 1, -1] + [0, 0, frame_height * 0.5]
      x2, y2, z2 = pose_landmarks[idx_2] * [1, 1, -1] + [0, 0, frame_height * 0.5]

      draw.ellipse([x1 - r, z1 - r, x1 + r, z1 + r], fill=color)
      draw.ellipse([x2 - r, z2 - r, x2 + r, z2 + r], fill=color)
      draw.line([x1, z1, x2, z2], width=int(r), fill=color)

    return np.asarray(img)

  def _check_rule(self, landmarks, image_index):
    wrong_frame = None

    # Transforms pose landmarks into embedding.
    pose_embedder = FullBodyPoseEmbedder()

    frame = PoseSample(
              index=image_index,
              landmarks=landmarks,
              embedding=pose_embedder(landmarks)
          )
    first_angle = frame.embedding[0] - 20
    second_angle = frame.embedding[1] - 20
    diff = abs(first_angle - second_angle)
    frame_info = diff
    if (diff > 15):
      #print('\n', first_angle, second_angle, frame.name)
      wrong_frame = (frame.index, diff)
    return (wrong_frame, frame_info, first_angle, second_angle)

  def _show_results(self, diff_mistakes, images, count_all, mistakes_count):
    if len(diff_mistakes)!=0:
        diff_list = [m[1] for m in diff_mistakes]
        index = diff_list.index(max(diff_list))
        img_pose = images[diff_mistakes[index][0]]
        st.subheader("Μέγιστη διαφορά γωνιών: " + str(max(diff_list)) )
        st.image(img_pose, channels="BGR")

    score = ((count_all - mistakes_count) / count_all) * 100
    st.metric("Τελική βαθμολογία", str(int(score)) + "%")
