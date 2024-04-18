import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from numba import jit

# combine frames, which consist of a single pixel slice, into a single image 
def combine_slices(dir):
  frames = [Image.open(dir + '/' + frame) for frame in os.listdir(dir)]
  width, height = frames[0].size
  new_img = Image.new('RGBA', (width, height * len(frames)), "black")
  pixels = new_img.load()
  for i, frame in enumerate(frames):
    for x in range(width):
      for y in range(height):
        pixel = frame.getpixel((x, y))
        pixels[x, i * height + y] = pixel
  return new_img

import imageio
from tqdm.auto import tqdm

def combine_imgs2(dir):
    files = [os.path.join(dir,f) for f in os.listdir(dir) if f.endswith(".png")]
    files = sorted(files)
    #print(files)

    img_count = len(files)
    img = imageio.imread(files[0])
    row_count = img.shape[0]
    result = np.zeros([row_count, img_count, img.shape[1], img.shape[2]], dtype=img.dtype)
    for i, img_path in tqdm(enumerate(files)):
        img = imageio.imread(img_path)
        for row in range(row_count):
                result[row, i] = img[row]

    for i, row_img in tqdm(enumerate(result)):
        imageio.imwrite(os.path.join(dir,"rows",f"{i}.png"), row_img)






def combine_images_to_wulsd_wrapper(dir):
  
  # frames using cv2
  frames = [cv2.imread(os.path.join(dir, frame)) for frame in os.listdir(dir)]

  # frames of shape (num_frames) to (num_frames, height, width, channels)

  height, width, channels = frames[0].shape
  
  # reorder the array to be in num_frames, height, width, channels
  #frames = np.array(frames).transpose(1, 0, 2, 3)

  new_dir = os.path.join(dir, Path(dir).name + "_WULSD")
  os.makedirs(new_dir, exist_ok=True)
  #num_frames = len(frames)
  
  out_array = np.zeros((height, len(frames), width, channels), dtype=np.uint8)
  print(out_array.shape)
  for i in range(len(frames)):
    for y in range(height):
      for x in range(width):
        if frames[i] is not None:
          out_array[y, i, x, :] = frames[i][y, x, :]
  
  for i in range(height):
    img = Image.fromarray(out_array[i])
    img.save(os.path.join(new_dir, str(i) + ".png"))


#@jit(nopython=True)
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
  


def horiz_derivative(img):
    width, height = img.size
    pixels = img.convert("RGBA").load()
    new_img = Image.new('RGBA', (width, height), "black")
    new_pixels = new_img.load()
    max_val = 0
    for x in range(width):
        for y in range(height):
            if x == 0:
                new_pixels[x, y] = (0, 0, 0, 0)
            elif pixels[x - 1, y][3] == 0:
                new_pixels[x, y] = (0, 0, 0, 0)
            else:
                new_pixels[x, y] = (pixels[x, y][0] - pixels[x - 1, y][0], pixels[x, y][1] - pixels[x - 1, y][1], pixels[x, y][2] - pixels[x - 1, y][2], 255)
                max_val = max(max_val, abs(new_pixels[x, y][0]))
    # normalize
    #for x in range(width):
    #    for y in range(height):
    #        new_pixels[x, y] = (int(new_pixels[x, y][0] / max_val * 255), int(new_pixels[x, y][1] / max_val * 255), int(new_pixels[x, y][2] / max_val * 255), new_pixels[x, y][3])
    return new_img

def vert_derivative(img):
  width, height = img.size
  pixels = img.convert("RGBA").load()
  new_img = Image.new('RGBA', (width, height), "black")
  new_pixels = new_img.load()
  max_val = 0
  for x in range(width):
    for y in range(height):
      if y == 0:
        new_pixels[x, y] = (0, 0, 0, 0)
      elif pixels[x, y - 1][3] == 0:
        new_pixels[x, y] = (0, 0, 0, 0)
      else:
        new_pixels[x, y] = (pixels[x, y][0] - pixels[x, y - 1][0], pixels[x, y][1] - pixels[x, y - 1][1], pixels[x, y][2] - pixels[x, y - 1][2])
        max_val = max(max_val, abs(new_pixels[x, y][0]))
  # normalize
  #for x in range(width):
  #  for y in range(height):
  #    new_pixels[x, y] = (int(new_pixels[x, y][0] / max_val * 255), int(new_pixels[x, y][1] / max_val * 255), int(new_pixels[x, y][2] / max_val * 255), new_pixels[x, y][3])
  return new_img

def get_pixel_slice(img, x, horizontal=False):
  width, height = img.size
  pixels = img.load()
  if horizontal:
    slice = Image.new('RGBA', (width, 1), "black")
    slice_pixels = slice.load()
    for i in range(width):
      slice_pixels[i, 0] = pixels[i, x]
  else:
    slice = Image.new('RGBA', (1, height), "black")
    slice_pixels = slice.load()
    for i in range(height):
      slice_pixels[0, i] = pixels[x, i]
  return slice

def get_slice_derivative(slice, horizontal=False):
  width, height = slice.size
  pixels = slice.load()
  if horizontal:
    new_slice = Image.new('RGBA', (width, 1), "black")
    new_pixels = new_slice.load()
    for i in range(width):
      if i == 0:
        new_pixels[i, 0] = (0, 0, 0, 0)
      elif pixels[i - 1, 0][3] == 0:
        new_pixels[i, 0] = (0, 0, 0, 0)
      else:
        new_pixels[i, 0] = (pixels[i, 0][0] - pixels[i - 1, 0][0], pixels[i, 0][1] - pixels[i - 1, 0][1], pixels[i, 0][2] - pixels[i - 1, 0][2], 255)
  else:
    new_slice = Image.new('RGBA', (1, height), "black")
    new_pixels = new_slice.load()
    for i in range(height):
      if i == 0:
        new_pixels[0, i] = (0, 0, 0, 0)
      elif pixels[0, i - 1][3] == 0:
        new_pixels[0, i] = (0, 0, 0, 0)
      else:
        new_pixels[0, i] = (pixels[0, i][0] - pixels[0, i - 1][0], pixels[0, i][1] - pixels[0, i - 1][1], pixels[0, i][2] - pixels[0, i - 1][2], 255)
  
  # normalize
  max_val = max(abs(new_pixels[i, 0][0]) for i in range(width)) if horizontal else max(abs(new_pixels[0, i][0]) for i in range(height))
  for i in range(width):
    if horizontal:
      new_pixels[i, 0] = (int(new_pixels[i, 0][0] / max_val * 255), int(new_pixels[i, 0][1] / max_val * 255), int(new_pixels[i, 0][2] / max_val * 255), 255)
    else:
      new_pixels[0, i] = (int(new_pixels[0, i][0] / max_val * 255), int(new_pixels[0, i][1] / max_val * 255), int(new_pixels[0, i][2] / max_val * 255), 255)
  
  return new_slice

def widen_slice(slice, factor, horizontal=False):
  width, height = slice.size
  pixels = slice.load()
  if horizontal:
    new_slice = Image.new('RGBA', (width, height * factor), "black")
    new_pixels = new_slice.load()
    for y in range(height):
      for i in range(factor):
        for x in range(width):
          new_pixels[x, y * factor + i] = pixels[x, y]
  else:
    new_slice = Image.new('RGBA', (width * factor, height), "black")
    new_pixels = new_slice.load()
    for x in range(width):
      for i in range(factor):
        for y in range(height):
          new_pixels[x * factor + i, y] = pixels[x, y]
  return new_slice

def diff_and_integrate_slice(slice, horizontal=False):
  width, height = slice.size
  pixels = slice.load()
  array = []
  if horizontal:
    for i in range(1, width):
      if pixels[i, 0][3] == 0:
        array.append(0)
      else:
        array.append(pixels[i - 1, 0][0] + pixels[i, 0][0] - pixels[i - 1, 0][0])
  else:
    for i in range(1, height):
      if pixels[0, i][3] == 0:
        array.append(0)
      else:
        array.append(pixels[0, i - 1][0] + pixels[0, i][0] - pixels[0, i - 1][0])
  
  #plt.plot(array)
  #plt.show()

  return array

     
def stitch_slice_diffs(img, horizontal=False):
  width, height = img.size
  # always take the middle value of each differance array and stitch them together
  if horizontal:
    new_slice_diff = []
    for y in range(height):
      slice = get_pixel_slice(img, y, True)
      slice_diff = diff_and_integrate_slice(slice, True)
      new_slice_diff.append(slice_diff[width // 2])
  else:
    new_slice_diff = []
    for x in range(width):
      slice = get_pixel_slice(img, x, False)
      slice_diff = diff_and_integrate_slice(slice, False)
      new_slice_diff.append(slice_diff[height // 2])
  return new_slice_diff


def main():
  dir = os.path.join("..", "scratch", "marble_100mm_10mm")
  #dir = os.path.join("..", "rendered_scratch", "1440_persp_100mm_10mm")
  combine_imgs2(dir)

  #img = combine_slices(dir)
  #img.show()
  #img.save(os.path.join(dir, "combined.png"))


  


if __name__ == '__main__':
  main()