import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

import tqdm
from numba import jit

def combine_images_to_wulsd_wrapper(dir):
  
  # frames using cv2
  frames = [cv2.imread(os.path.join(dir, frame)) for frame in os.listdir(dir)]
  print(f"Loaded {len(frames)} frames from {dir}")

  # frames of shape (num_frames) to (num_frames, height, width, channels)

  height, width, channels = frames[0].shape
  
  # reorder the array to be in num_frames, height, width, channels
  #frames = np.array(frames).transpose(1, 0, 2, 3)

  new_dir = os.path.join(dir, "..", Path(dir).name + "_WULSD")
  os.makedirs(new_dir, exist_ok=True)
  #num_frames = len(frames)
  
  out_array = np.zeros((height, len(frames), width, channels), dtype=np.uint8)
  print(out_array.shape)
  for i in range(len(frames)):
    for y in range(height):
      for x in range(width):
        if frames[i] is not None:
          out_array[y, i, x, :] = frames[i][y, x, :]
  
  print(f"Saving {out_array.shape[0]} images to {new_dir}")

  for i in range(height):
    img = Image.fromarray(out_array[i])
    #img.save(os.path.join(new_dir, str(i) + ".png"))
    img = img.convert('RGB')
    img.save(os.path.join(new_dir, str(i) + ".jpg"), 'JPEG')


@jit(nopython=True)
def combine_images_to_wulsd(frames, width, height, channels, num_frames):
  if not frames:
    return np.zeros((height, width, num_frames, channels), dtype=np.uint8)
  
  out_array = np.zeros((height, width, num_frames, channels), dtype=np.uint8)
  print(out_array.shape)
  for y in range(height):
    for x in range(width):
      for i in range(num_frames):
        out_array[y, x, i, :] = frames[i, y, x, :]
        #out_array[y, x, i] = 1#temp
        
  return out_array
  
def png_to_jpg(path, newpath):
  pngs = [f for f in os.listdir(path) if f.endswith('.png')]
  os.makedirs(newpath, exist_ok=True)
  for png in tqdm.tqdm(pngs):
    img = Image.open(os.path.join(path, png))
    img = img.convert('RGB')
    img.save(os.path.join(newpath, png.replace('.png', '.jpg')), 'JPEG')


def main():
  dir = os.path.join("..", "renders", "marble_90d_ortho_cycles")
  combine_images_to_wulsd_wrapper(dir)



if __name__ == '__main__':
  main()