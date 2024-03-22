import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
from numba import jit
import os
from image_augmentation import combine_slices
from tqdm.auto import tqdm
import skimage as sk
import open3d as o3d

def calculate_hough_space(image):
  edges = cv2.Canny(image, 2, 5)

  gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
  gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

  return calculate_hough_spaces_helper(image, edges, gradient_x, gradient_y)


@jit(nopython=True)
def calculate_hough_spaces_helper(image, edges, gradient_x, gradient_y):
  num_rotations, width = image.shape
  x_center = width // 2

  # initialize hough spaces
  hough_space = np.zeros((x_center, num_rotations))


  # iterate over all pixels in the edge image
  for x in range(1, width - 1):
    for theta in range(1, num_rotations):
      # if the pixel is an edge pixel
      if edges[theta, x] > 0:
        # only iterate over the amplitudes further away from the center than the current pixel
        for amplitude in range(np.abs(x - x_center), x_center):
          if amplitude == 0:
            continue

          # check if (delta_x / delta_theta) > 0 using the gradients
          if gradient_x[theta, x] * gradient_y[theta, x] < 0:
            # calculate the corresponding hough space phi
            phi = np.arcsin((x - x_center) / amplitude) - ((theta / num_rotations) * 2 * np.pi)
            phi = phi % (2 * np.pi)
            phi = int(phi / (2 * np.pi) * (num_rotations - 1))
            # add the amplitude to the corresponding hough space
            hough_space[amplitude, phi] += 1
          else:
            # calculate the corresponding hough space phi
            phi = np.arccos((x - x_center) / amplitude) - ((theta / num_rotations) * 2 * np.pi) + (np.pi / 2)
            phi = phi % (2 * np.pi)
            phi = int(phi / (2 * np.pi) * (num_rotations - 1))
            # add the amplitude to the corresponding hough space
            hough_space[amplitude, phi] += 1

  return hough_space

def post_process_hough_space(hough_space):
    sum = np.sum(hough_space)
    size = hough_space.shape[0] * hough_space.shape[1]
    temp = size / sum if sum != 0 else 0.5
    r = int(temp * min(hough_space.shape[0], hough_space.shape[1]))

    # remove low frequencies by subtracting the lowpass filtered image
    fspace = np.fft.fft2(hough_space)
    fspace = np.fft.fftshift(fspace)
    rows, cols = hough_space.shape
    crow, ccol = rows // 2, cols // 2
    # create a mask first, center circle is 1, remaining all zeros
    mask = np.zeros((rows, cols), np.uint8)
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r**2
    mask[mask_area] = 1
    fshift = fspace * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    hough_space = hough_space - img_back

    # clip all values below 0 to 0
    hough_space[hough_space < 0] = 0
    

    # Weigh the matrix with weights e^(-0.001)*[1,2,...,A_max]
    for amplitude in range(1, hough_space.shape[0]):
        hough_space[amplitude, :] = hough_space[amplitude, :] * np.exp(-0.001 * amplitude)

    # normalize the matrix
    hough_space = (hough_space - np.min(hough_space)) / (np.max(hough_space) - np.min(hough_space))

    hough_space = hough_space * 255

    # gradient in y direction
    gradient_hough = np.abs(cv2.Sobel(hough_space, cv2.CV_64F, 0, 1, ksize=5))

    # normalize the gradient
    if gradient_hough.max() > 0: 
      gradient_hough = (gradient_hough - np.min(gradient_hough)) / (np.max(gradient_hough) - np.min(gradient_hough))

    gradient_hough = gradient_hough * 255

    t, _ = cv2.threshold(gradient_hough.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    
    hough_space[gradient_hough < t] = 0

    coordinates_hough = sk.feature.peak_local_max(hough_space, threshold_rel = 0.2, min_distance=5)
    hough_temp = np.zeros(hough_space.shape)

    '''
    for i in range(len(coordinates_h1)):
      h1_temp[coordinates_h1[i, 0], coordinates_h1[i, 1]] = 1
      # find nearest coordinate and draw a line
      nearest = (0, 0)
      min_dist = 100000
      for j in range(len(coordinates_h1)):
        if i == j:
          continue
        dist = np.sqrt((coordinates_h1[i, 0] - coordinates_h1[j, 0]) ** 2 + (coordinates_h1[i, 1] - coordinates_h1[j, 1]) ** 2)
        if dist < min_dist and dist != 0:
          min_dist = dist
          nearest = (coordinates_h1[j, 0], coordinates_h1[j, 1])
      if nearest != (0, 0):
        # draw a line between the two points
        cv2.line(h1_temp, (coordinates_h1[i, 1], coordinates_h1[i, 0]), (nearest[1], nearest[0]), 1, 1)

    for i in range(len(coordinates_h2)):
      h2_temp[coordinates_h2[i, 0], coordinates_h2[i, 1]] = 1
      # find nearest coordinate and draw a line
      nearest = (0, 0)
      min_dist = 100000
      for j in range(len(coordinates_h2)):
        if i == j:
          continue
        dist = np.sqrt((coordinates_h2[i, 0] - coordinates_h2[j, 0]) ** 2 + (coordinates_h2[i, 1] - coordinates_h2[j, 1]) ** 2)
        if dist < min_dist and dist != 0:
          min_dist = dist
          nearest = (coordinates_h2[j, 0], coordinates_h2[j, 1])
      # draw a line between the two points
      cv2.line(h2_temp, (coordinates_h2[i, 1], coordinates_h2[i, 0]), (nearest[1], nearest[0]), 1, 1)

      '''


    hough_temp[coordinates_hough[:, 0], coordinates_hough[:, 1]] = 1
    ## set those coordinates which dont! have a local maximum to 0 by multiplying with the temp array
    hough_space = hough_space * hough_temp
    return hough_space

def compute_3d_coordinates(hough_space, indices, image, z):
  init_width, num_rotations = image.shape  # Anzahl der Rotationen
  width = init_width * 2
  
  points = []
  for i in range(len(indices[0])):
    amplitude_h1, phase_h1 = indices[0][i], indices[1][i]
    x = amplitude_h1 * np.cos((phase_h1 / init_width) * 2 * np.pi)
    y = amplitude_h1 * np.sin((phase_h1 / init_width) * 2 * np.pi)
    points.append((x, y, z, hough_space[amplitude_h1, phase_h1]))

  return np.array(points)

def full_hough_to_ply(dir):
    # iterate over all images in the directory
    frames = [cv2.imread(os.path.join(dir, frame), 0) for frame in os.listdir(dir)]
    z_max = len(frames)
    #frames = frames[::100]
    frames = frames[::-1]

    width = frames[0].shape[1]
    # floating points for ply point cloud
    points_coordinates = []

    with tqdm(total=len(frames), desc="Processing frames") as pbar:
      for i, frame in enumerate(frames):
        hough_space = calculate_hough_space(frame)
        hough_space = post_process_hough_space(hough_space)

        indices = np.transpose(np.argwhere(hough_space > 0))

        z = i / len(frames) * z_max
        points = compute_3d_coordinates(hough_space, indices, frame, z)
        points_coordinates.extend(points)
        
        pbar.update(1)

    # write the points to a ply file
    with open("point_cloud_test.ply", "w") as file:
      file.write("ply\n")
      file.write("format ascii 1.0\n")
      file.write("element vertex " + str(len(points_coordinates)) + "\n")
      file.write("property float32 x\n")
      file.write("property float32 y\n")
      file.write("property float32 z\n")
      file.write("property uint8 red\n")
      file.write("property uint8 green\n")
      file.write("property uint8 blue\n")
      file.write("end_header\n")
      for point in points_coordinates:
        file.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + " " + str(int(point[3])) + " 0 " + str(int(255 - point[3])) + "\n")

def main():
  #dir = os.path.join("..", "scratch", "vonroi_wulsd")
  #dir = os.path.join("..", "scratch", "full_persp_50mm_2.5m", "rows")
  #full_hough_to_ply(dir)

  cloud = o3d.io.read_point_cloud("point_cloud_1080_bam.ply")
  #cloud = o3d.io.read_point_cloud("point_cloud_perspective_bam.ply")
  # normalize the colors
  colors = np.asarray(cloud.colors)
  max = np.max(colors[:, 2])
  min = np.min(colors[:, 2])
  colors[:, 2] = (colors[:, 2] - min) / (max - min)

  points = o3d.utility.Vector3dVector(cloud.points)

  # use colormap to color the point cloud
  cmap = plt.get_cmap('inferno')
  colors = cmap(colors[:, 2])[:, :3]
  cloud.colors = o3d.utility.Vector3dVector(colors)
  pcd = o3d.visualization.draw_geometries([cloud])

if __name__ == "__main__":
  main()