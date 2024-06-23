import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
from numba import jit
import os
from tqdm.auto import tqdm
import skimage as sk
import open3d as o3d
import random
import time

#1. Two different hough spaces h1 and h2
#2. Fusing both of them
#3. using ellipse ratio 0 (the same but mine basically)
#4. using a different angle
#5. fusing both angles

def hough_roadmap1(dir):
  #1. get the wulsds of dir
  #2. compute h1 and h2
  #3. post process independantly
  #4. scale the results to world coordinates
  #5. plot the results in different colors? or just together idk
  pass

def hough_roadmap2(dir):
  #1. get the wulsds of dir
  #2. compute h1 and h2
  #3. add them together
  #4. post process together
  #5. scale the results to world coordinates
  #6. plot results
  pass

def hough_roadmap3(dir):
  #1. calculate hough space with ellipse ratio of 0
  #2. should result in very simlar results
  #3. post process
  #4. scale the results to world coordinates
  #5. plot results
  pass

def hough_roadmap4(dir):
  #1. calculate hough space with a different angle
  #2. post process
  #3. scale the results to world coordinates
  #4. plot results
  pass

def hough_roadmap5(dir):
  #1. calculate hough space with both angles
  #2. scale the results to world coordinates
  #3. fuse the hough spaces
  #4. post process together
  #5. plot results
  pass

def hough_transform_wulsd(dir):
  frame_names = os.listdir(dir)
  frame_names = [frame for frame in frame_names if frame.split('.')[0].isdigit()]
  frame_names = sorted(frame_names, key=lambda x: int(x.split('.')[0]))
  frame_names = frame_names[::40]
  frames = [cv2.imread(os.path.join(dir, frame), 0) for frame in frame_names if frame.endswith('.png')]
  z_max = len(frames)
  theta_max, rad_max = frames[0].shape

  hough_space1 = np.zeros((z_max, rad_max // 2, theta_max))
  hough_space2 = np.zeros((z_max, rad_max // 2, theta_max))

  with tqdm(total=z_max, desc="Processing frames") as pbar:
    for z, frame in enumerate(frames):
      edges = cv2.Canny(frame, 2, 5)
      gradient_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
      gradient_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

      h1, h2 = hough_transform_wulsd_helper(frame, edges, gradient_x, gradient_y)
      hough_space1[z] = h1
      hough_space2[z] = h2
      pbar.update(1)

  return hough_space1, hough_space2


@jit(nopython=True)
def hough_transform_wulsd_helper(image, edges, gradient_x, gradient_y):
  num_rotations, width = image.shape
  x_center = width // 2

  # initialize hough spaces
  hough_space1 = np.zeros((x_center, num_rotations))
  hough_space2 = np.zeros((x_center, num_rotations))


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
            hough_space1[amplitude, phi] += 1
          else:
            # calculate the corresponding hough space phi
            phi = np.arccos((x - x_center) / amplitude) - ((theta / num_rotations) * 2 * np.pi) + (np.pi / 2)
            phi = phi % (2 * np.pi)
            phi = int(phi / (2 * np.pi) * (num_rotations - 1))
            # add the amplitude to the corresponding hough space
            hough_space2[amplitude, phi] += 1

  return hough_space1, hough_space2

def hough_transform(dir, ellipse_ratio = 0, threshold = 20):
  frame_names = os.listdir(dir)
  # sort the frames named 0.png, 1.png, ..., 100.png, ...
  # filter out every frame thats not an int
  frame_names = [frame for frame in frame_names if frame.split('.')[0].isdigit()]
  frame_names = sorted(frame_names, key=lambda x: int(x.split('.')[0]))
  frame_names = frame_names[::4]
  frames = [cv2.imread(os.path.join(dir, frame), 0) for frame in frame_names if frame.endswith('.png')]
  theta_max = len(frames)

  # get the first frame
  frame = frames[0]
  # get the frame size
  height, width = frame.shape
  z_max = height
  rad_max = frame.shape[1] // 2
  

  # hough spaces are per z slice the radius on one axis and the angle on the other axis
  hough_spaces = np.zeros((theta_max, rad_max, z_max), dtype=np.uint32)

  with tqdm(total=len(frames), desc="Processing frames") as pbar:
    for i, frame in enumerate(frames):
      if i == 0:
        continue

      # check if cv2 can read the frame
      if frame is None:
        continue

      
      # get gradients in x and y direction
      #grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
      #grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
#
      ## get the gradient magnitude
      #grad_mag = np.sqrt(grad_x**2 + grad_y**2)

      ## normalize the gradient magnitude
      #grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


      # blur frame
      #grad_mag = cv2.GaussianBlur(frame, (5, 5), 0)
#
      ## sort points by gradient magnitude
      #points = np.argwhere(grad_mag > threshold)
      #points = points[points[:, 1].argsort()]
#
      #grad_mag = np.zeros_like(grad_mag)

      #grad_mag[points[0, 0], points[0, 1]] = 255

      #grad_mag = cv2.erode(grad_mag, np.ones((3, 3), np.uint8), iterations=1)


      # threshold the gradient magnitude
      #grad_mag = cv2.threshold(grad_mag, threshold, 255, cv2.THRESH_BINARY)[1]
      #plt.imshow(grad_mag, cmap='gray')
      #plt.show()
      edges = cv2.Canny(frame, 2, 5)
      grad_mag = edges
      threshold = 0

      theta_base = (i / theta_max) * 2 * np.pi if i != 0 else 0

      hough_spaces = hough_transform_helper(grad_mag, hough_spaces, theta_base, ellipse_ratio, threshold, rad_max, z_max, theta_max, width, height)

      pbar.update(1)

  return hough_spaces

@jit(nopython=True)
def hough_transform_helper(grad_mag, hough_spaces, theta_base, ellipse_ratio, threshold, rad_max, z_max, theta_max, width, height):
  # get points where the gradient magnitude is above a threshold
  points = np.argwhere(grad_mag > threshold)
  
  for y, x in points:
    # loop over all possible radii
    for a in range(np.abs(rad_max - x), rad_max):
      if a == 0:
        continue

      b = a * ellipse_ratio

      x_rel = x - rad_max
      #y_rel = np.sqrt((1 - (x_rel**2 / a**2)) * b**2) # mine
      y_rel = (b * np.sqrt(a**2 - x_rel**2)) / a # wolfram alpha
      if a == x_rel:
        y_rel = 0

      z0 = int(y - y_rel)
      z1 = int(y + y_rel)

      y_im = np.sqrt(a**2 - x_rel**2) if a != x_rel else 0

      #print (f"x: {x}, y: {y}, x_rel: {x_rel}, y_rel: {y_rel}, z0: {z0}, z1: {z1}") if a == 295 else None

      # get the theta values for the ellipses
      theta0 = int(((np.pi + math.atan2(x_rel, y_im) - theta_base) % (2 * np.pi))  / (2 * np.pi) * (theta_max)) % theta_max
      theta1 = int(((np.pi + math.atan2(x_rel, -y_im) - theta_base) % (2 * np.pi)) / (2 * np.pi) * (theta_max)) % theta_max

      #theta0 = int((math.acos(x_rel / a) - theta_base) % (2 * np.pi) * (theta_max / (2 * np.pi)))
      #theta1 = int((math.acos(-x_rel / a) - theta_base) % (2 * np.pi) * (theta_max / (2 * np.pi)))

      if z0 >= 0 and z0 < height:
        hough_spaces[theta0, a, z0] += 1
      if z1 >= 0 and z1 < height:
        hough_spaces[theta1, a, z1] += 1
  return hough_spaces


def post_process_3d(hough_spaces):
  
  print("Removing low frequencies...")

  start = time.time()

  #hough_spaces_lowpass = sk.ndimage.gaussian_filter(hough_spaces, 5)
  #hough_spaces = hough_spaces - hough_spaces_lowpass
  #return hough_spaces

  hough_spaces_fourier = np.fft.fftn(hough_spaces)
  time_elapsed = time.time() - start
  print(f"Time elapsed: {time_elapsed}")

  print("Shifting the fourier transform...")
  hough_spaces_fourier = np.fft.fftshift(hough_spaces_fourier)
  print("Creating mask...")

  radius = 10
  rows, cols, depth = hough_spaces.shape
  crow, ccol, cdep = rows // 2, cols // 2, depth // 2
  mask = np.zeros((rows, cols, depth), np.uint8)
  center = [crow, ccol, cdep]
  x, y, z = np.ogrid[:rows, :cols, :depth]
  mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2 <= radius**2
  mask[mask_area] = 1
  fshift = hough_spaces_fourier * mask
  del mask, mask_area, hough_spaces_fourier

  f_ishift = np.fft.ifftshift(fshift)
  del fshift
  hough_spaces = hough_spaces - np.abs(np.fft.ifftn(f_ishift))
  del f_ishift


  # Weigh the matrix with weights e^(-0.001)*[1,2,...,A_max] along the amplitude axis
  print("Weighing the matrix...")
  for amplitude in range(1, hough_spaces.shape[1]):
    hough_spaces[:, amplitude, :] = hough_spaces[:, amplitude, :] * np.exp(-0.001 * amplitude)

  #return hough_spaces

  hough_mask = hough_spaces.flatten()
  # normalize the mask
  hough_mask = cv2.normalize(hough_mask.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

  t, _ = cv2.threshold(hough_mask.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
  print(f"Threshold: {t}")
  hough_mask = hough_mask.reshape(hough_spaces.shape)

  hough_spaces[hough_mask <= t] = 0

  #return hough_spaces

  # find local maximas
  print("Finding local maximas...")
  coordinates_hough = sk.feature.peak_local_max(hough_spaces, threshold_rel = 0.2, min_distance=3)

  hough_mask = np.zeros(hough_spaces.shape)
  hough_mask[coordinates_hough[:, 0], coordinates_hough[:, 1], coordinates_hough[:, 2]] = 1

  hough_spaces = hough_spaces * hough_mask

  return hough_spaces

def post_processed_to_point_cloud_wulsd(hough_space, transformation = (1, 1, 0, 0), z_max = None):
  if z_max is None:
    z_max = hough_space.shape[0]

  z_step = z_max / hough_space.shape[0]

  scale_radius, scale_height, translation_angle, translation_height = transformation

  extrema = np.argwhere(hough_space > 0)
  points = []
  colors = []

  for z, a, theta_base in extrema:
    colors.append([hough_space[z, a, theta_base], 0, 0])
    theta_base = (theta_base + translation_angle) % hough_space.shape[2]
    a = a * scale_radius
    z = z * z_step * scale_height + translation_height

    theta = (theta_base / (hough_space.shape[2] - 1)) * 2 * np.pi + np.pi
    # get the x and y values
    x = a * np.cos(theta)
    y = a * np.sin(theta)
    points.append([x, y, z])

  colors = np.array(colors)
  colors = colors / np.max(colors)

  return np.array(points), colors

def post_processed_to_point_cloud(hough_space, transformation = (1, 1, 0, 0)):
  scale_radius, scale_height, translation_angle, translation_height = transformation

  extrema = np.argwhere(hough_space > 0)
  points = []
  colors = []

  for theta_base, a, z in extrema:
    colors.append([hough_space[theta_base, a, z], 0, 0])
    theta_base = (theta_base + translation_angle) % hough_space.shape[0]
    a = a * scale_radius
    z = z * scale_height + translation_height

    theta = (theta_base / (hough_space.shape[0] - 1)) * 2 * np.pi
    # get the x and y values
    x = a * np.cos(theta)
    y = a * np.sin(theta)
    points.append([x, y, z])

  colors = np.array(colors)
  colors = colors / np.max(colors)

  return np.array(points), colors

def plot_point_cloud(points, colors):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  pcd.colors = o3d.utility.Vector3dVector(colors)

  o3d.visualization.draw_geometries([pcd])




def main():

  if False:
    #dir_wulsd = "../scratch/dual_angle/voronoi_90d/rows"
    dir_wulsd = "../scratch/marble_100mm_10mm/rows"

    z_max = len(os.listdir(dir_wulsd))

    hough_space1, hough_space2 = hough_transform_wulsd(dir_wulsd)
    hough_space1 = post_process_3d(hough_space1)
    hough_space2 = post_process_3d(hough_space2)

    points1, colors1 = post_processed_to_point_cloud_wulsd(hough_space1, transformation=(1, 1, 0, 0), z_max=z_max)
    points2, colors2 = post_processed_to_point_cloud_wulsd(hough_space2, transformation=(1, 1, 0, 0), z_max=z_max)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points1)
    # paint the point cloud red
    pcd.paint_uniform_color([1, 0, 0])

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    # paint the point cloud blue
    pcd2.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pcd, pcd2])

  if True:
    #dir_wulsd = "../scratch/dual_angle/voronoi_90d/rows"
    dir_wulsd = "../scratch/marble_100mm_10mm/rows"

    z_max = len(os.listdir(dir_wulsd))

    hough_space1, hough_space2 = hough_transform_wulsd(dir_wulsd)
    hough_space = hough_space1 + hough_space2
    hough_space = post_process_3d(hough_space)

    points, colors = post_processed_to_point_cloud_wulsd(hough_space, transformation=(1, 1, 0, 0), z_max=z_max) 

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points)
    pcd1.paint_uniform_color([1, 0, 0])

    #o3d.visualization.draw_geometries([pcd])

#  if True:
    dir = "../scratch/marble_100mm_10mm"
    hough_space = hough_transform(dir, ellipse_ratio=0, threshold=20)
    hough_space = post_process_3d(hough_space)
    points, colors = post_processed_to_point_cloud(hough_space, transformation=(1, 1, 0, 0))

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points)
    pcd2.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([pcd1, pcd2])


if __name__ == "__main__":
  main()
    