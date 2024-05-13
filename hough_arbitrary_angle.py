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
from sklearn.neighbors import NearestNeighbors

def full_hough_to_ply(dir):
  ellipse_ratio = 0#.258
  threshold = 20

  frame_names = os.listdir(dir)
  # sort the frames named 0.png, 1.png, ..., 100.png, ...
  # filter out every frame thats not an int
  frame_names = [frame for frame in frame_names if frame.split('.')[0].isdigit()]
  frame_names = sorted(frame_names, key=lambda x: int(x.split('.')[0]))
  frame_names = frame_names[::8]
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
      grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
      grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

      # get the gradient magnitude
      grad_mag = np.sqrt(grad_x**2 + grad_y**2)

      # normalize the gradient magnitude
      grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


      # blur frame
      #grad_mag = cv2.GaussianBlur(frame, (5, 5), 0)
#
      ## sort points by gradient magnitude
      #points = np.argwhere(grad_mag > threshold)
      #points = points[points[:, 1].argsort()]
#
      #grad_mag = np.zeros_like(grad_mag)
#
      #grad_mag[points[0, 0], points[0, 1]] = 255

      #grad_mag = cv2.erode(grad_mag, np.ones((3, 3), np.uint8), iterations=1)


      # threshold the gradient magnitude
      #grad_mag = cv2.threshold(grad_mag, threshold, 255, cv2.THRESH_BINARY)[1]
      #plt.imshow(grad_mag, cmap='gray')
      #plt.show()

      theta_base = (i / theta_max) * 2 * np.pi if i != 0 else 0

      hough_spaces = hough_helper(grad_mag, hough_spaces, theta_base, ellipse_ratio, threshold, rad_max, z_max, theta_max, width, height)

      pbar.update(1)


    # save the hough spaces to a file
    #np.save("hough_spaces.npy", hough_spaces)
    #return  

    #hough_spaces = post_process_hough(hough_spaces)
    with tqdm(total=hough_spaces.shape[0], desc="Post processing hough spaces") as pbar:
      for i, hough_space in enumerate(hough_spaces):  
        #plt.imshow(hough_spaces[i], cmap='jet')
        #plt.show()
        hough_spaces[i] = post_process_hough(hough_space)
        
        
        
        #hough_spaces[i, 295, 168] = 1000
        pbar.update(1)

    # convert extrema to 3D points
    points = []
    colors = []

    # get the extrema
    extrema = np.argwhere(hough_spaces > 0)
    with tqdm(total=len(extrema), desc="Converting extrema to 3D points") as pbar:
      for theta_base, a, z  in extrema:
        theta = (theta_base / hough_spaces.shape[0]) * 2 * np.pi
        # get the x and y values
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        points.append([x, y, z])
        colors.append([hough_spaces[theta_base, a, z], 0, 0])
        pbar.update(1)

    # normalize the colors
    colors = np.array(colors)
    colors = colors / np.max(colors)

    #colors[:, 2] = 1 - colors[:, 0]

    color_vals = colors[:, 0]

    cmap = plt.get_cmap('inferno')
    colors = cmap(color_vals)[:, :3]

    # create a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    colors *= 255
    colors = colors.astype(np.uint8)

    """
    filename ="test_arbitrary_angle.ply"

    # write the points to a ply file
    with open(filename, "w") as file:
      file.write("ply\n")
      file.write("format ascii 1.0\n")
      file.write("element vertex " + str(len(points)) + "\n")
      file.write("property float32 x\n")
      file.write("property float32 y\n")
      file.write("property float32 z\n")
      file.write("property uint8 red\n")
      file.write("property uint8 green\n")
      file.write("property uint8 blue\n")
      file.write("end_header\n")
      for i, point in enumerate(points):
        file.write(str(float(point[0])) + " " + str(float(point[1])) + " " + str(float(point[2])) + " " + str(colors[i][0]) + " " + str(colors[i][1]) + " " + str(colors[i][2]) + "\n")
    
    """
    o3d.visualization.draw_geometries([pcd])

    


def full_hough_to_ply_cols(dir):
  ellipse_ratio = 0.5
  threshold = 50

  frame_names = os.listdir(dir)
  # sort the frames named 0.png, 1.png, ..., 100.png, ...
  frame_names = sorted(frame_names)
  width = len(frame_names)
  frame_names = frame_names[::5]
  frames = [cv2.imread(os.path.join(dir, frame), 0) for frame in frame_names if frame.endswith('.png')]

  x_frac = width / len(frames)

  # get the first frame
  frame = frames[0]
  # get the frame size
  height, theta_max = frame.shape

  hough_spaces = np.zeros((height, width, theta_max), dtype=np.uint32)

  with tqdm(total=len(frames), desc="Processing frames") as pbar:
    for x, frame in enumerate(frames):
      x = int(x * x_frac)

      # get gradients in x and y direction
      grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
      grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

      # get the gradient magnitude
      grad_mag = np.sqrt(grad_x**2 + grad_y**2)

      # normalize the gradient magnitude
      grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

      points = np.argwhere(grad_mag > threshold)

      for theta, y in points:
        a = np.abs(width // 2 - x)
        if a == 0:
          continue
        b = a * ellipse_ratio
        x_rel = width // 2 - x
        y_rel = np.sqrt((1 - (x_rel**2 / a**2)) * b**2)

        z0 = int(theta - y_rel)
        z1 = int(theta + y_rel)

        theta0 = int(math.atan2(x_rel, y_rel) + theta) % theta_max
        theta1 = int(math.atan2(x_rel, -y_rel) + theta) % theta_max

        if z0 >= 0 and z0 < height:
          hough_spaces[z0, x, theta0] += 1
        if z1 >= 0 and z1 < height:
          hough_spaces[z1, x, theta1] += 1

      pbar.update(1)

    # convert extrema to 3D points
    points = []
    colors = []
    # get the extrema
    extrema = np.argwhere(hough_spaces > 10)
    with tqdm(total=len(extrema), desc="Converting extrema to 3D points") as pbar:
      for z, a, theta in extrema:
        # get the x and y values
        x = a * np.cos(theta)
        y = a * np.sin(theta)

        points.append([x, y, z])
        colors.append([hough_spaces[z, a, theta], 0, 0])
        pbar.update(1)

    # normalize the colors
    colors = np.array(colors)
    colors = colors / np.max(colors)

    # create a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])





      



    

@jit(nopython=True)
def hough_helper(grad_mag, hough_spaces, theta_base, ellipse_ratio, threshold, rad_max, z_max, theta_max, width, height):
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

def post_process_hough(hough_space):

  #plt.imshow(hough_space, cmap='gray')
  #plt.show()
  
  sum = np.sum(hough_space)
  size = hough_space.shape[0] * hough_space.shape[1]
  temp = size / sum if sum != 0 else 0.5
  r = int(temp * min(hough_space.shape[0], hough_space.shape[1]))
  #r = 10


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
  #hough_space = (hough_space - np.min(hough_space)) / (np.max(hough_space) - np.min(hough_space))
  #hough_space = hough_space * 255

  #return hough_space
  
  # gradient in y direction
  gradient_hough = np.abs(cv2.Sobel(hough_space, cv2.CV_64F, 0, 1, ksize=5))

  # normalize the gradient
  #if gradient_hough.max() > 0: 
  #  gradient_hough = (gradient_hough - np.min(gradient_hough)) / (np.max(gradient_hough) - np.min(gradient_hough))
#
  #gradient_hough = gradient_hough * 255

  t, _ = cv2.threshold(gradient_hough.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
  
  hough_space[gradient_hough <= t] = 0

  #plt.imshow(hough_space, cmap='inferno')
  #plt.show()


  hough_space = cv2.GaussianBlur(hough_space, (5, 5), 0)

  coordinates_hough = sk.feature.peak_local_max(hough_space, threshold_rel = 0.2, min_distance=10)
  #coordinates_hough = sk.feature.peak_local_max(hough_space, threshold_rel = 0.2, min_distance=10)

  # sort by x coordinate
  coordinates_hough = coordinates_hough[coordinates_hough[:, 1].argsort()]
  
  # Triangular moving average
  #for i in range(6, len(coordinates_hough) - 6):
  #    distances = []
  #    for j in range(-6, 6):
  #        dist = np.sqrt((coordinates_hough[i, 0] - coordinates_hough[i + j, 0]) ** 2 + (coordinates_hough[i, 1] - coordinates_hough[i + j, 1]) ** 2)
  #        distances.append([dist, j])
  #    distances = np.array(distances)
  #    distances = distances[distances[:, 0].argsort()]
  #    distances = distances[:4]

      #coordinates_hough[i, 0] = (coordinates_hough[i, 0] + coordinates_hough[i + int(distances[0, 1]), 0] + coordinates_hough[i + int(distances[1, 1]), 0] + coordinates_hough[i + int(distances[2, 1]), 0] + coordinates_hough[i + int(distances[3, 1]), 0]) // 5
      #coordinates_hough[i, 1] = (coordinates_hough[i, 1] + coordinates_hough[i + int(distances[0, 1]), 1] + coordinates_hough[i + int(distances[1, 1]), 1] + coordinates_hough[i + int(distances[2, 1]), 1] + coordinates_hough[i + int(distances[3, 1]), 1]) // 5

  



  # smooth out the coordinates by considering the 4 nearest neighbors
  #for i in range(len(coordinates_hough)):
  #    nearest_neighbors = []
  #    for j in range(len(coordinates_hough)):
  #        if i == j:
  #            continue
  #        dist = np.sqrt((coordinates_hough[i, 0] - coordinates_hough[j, 0]) ** 2 + (coordinates_hough[i, 1] - coordinates_hough[j, 1]) ** 2)
  #        nearest_neighbors.append(dist)
  #    nearest_neighbors = np.array(nearest_neighbors)
  #    nearest_neighbors = nearest_neighbors.argsort()
  #    nearest_neighbors = nearest_neighbors[:4]
  #    # get the average y of the nearest neighbors
  #    y = 0
  #    for neighbor in nearest_neighbors:
  #        y += coordinates_hough[neighbor, 0]
  #    y = y / 4
  #    #coordinates_hough[i, 0] = (coordinates_hough[i, 0] + y) // 2





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
  #hough_temp = hough_space * hough_temp

  #plt.imshow(hough_temp, cmap='jet')
  #plt.show()
  

  #hough_temp = smooth_hough_space(hough_temp, coordinates_hough)
  hough_space = hough_space * hough_temp

  #plt.imshow(hough_space, cmap='inferno')
  #plt.show()

  return hough_space
  

#@jit(nopython=True, debug=True)
def post_process_hough_3d(hough_spaces):

  # normalize
  #hough_spaces = hough_spaces / np.max(hough_spaces)

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

  # find local maximas
  print("Finding local maximas...")
  coordinates_hough = sk.feature.peak_local_max(hough_spaces, threshold_rel = 0.2, min_distance=3)

  hough_mask = np.zeros(hough_spaces.shape)
  hough_mask[coordinates_hough[:, 0], coordinates_hough[:, 1], coordinates_hough[:, 2]] = 1

  hough_spaces = hough_spaces * hough_mask

  #hough_spaces[hough_spaces <= t] = 0
  return hough_spaces

  # create 3d gradient magnitude
  print("Calculating gradient magnitude...")
  hough_gradient = np.gradient(hough_spaces)
  hough_gradient = np.abs(hough_gradient)

  # threshold the gradient magnitude
  print("Thresholding the gradient magnitude...")
  gradient_flat = hough_gradient.flatten()
  # use otsu thresholding
  t, _ = cv2.threshold(gradient_flat.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
  hough_spaces[hough_gradient <= t] = 0

  return hough_spaces


def full_hough_to_ply_sausage(dir):
  ellipse_ratio = 0.5
  threshold = 40

  frame_names = os.listdir(dir)
  # sort the frames named 0.png, 1.png, ..., 100.png, ...
    # filter out every frame thats not an int
  frame_names = [frame for frame in frame_names if frame.split('.')[0].isdigit()]
  frame_names = sorted(frame_names, key=lambda x: int(x.split('.')[0]))
  z_total = len(frame_names)

  #frame_names = frame_names[::4]
  frames = [cv2.imread(os.path.join(dir, frame), 0) for frame in frame_names if frame.endswith('.png')]
  frame = frames[0]
  # get the frame size
  theta_max, width = frame.shape
  z_max = len(frames)
  rad_max = width // 2

  # hough spaces are per z slice the radius on one axis and the angle on the other axis
  hough_spaces = np.zeros((z_total, rad_max, theta_max), dtype=np.uint32)

  with tqdm(total=len(frames), desc="Processing frames") as pbar:
    for i, frame in enumerate(frames):
      if i == 0:
        continue

      # check if cv2 can read the frame
      if frame is None:
        continue

      # get gradients in x and y direction
      grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
      grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

      # get the gradient magnitude
      grad_mag = np.sqrt(grad_x**2 + grad_y**2)

      # normalize the gradient magnitude
      grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
      #plt.imshow(grad_mag, cmap='gray')
      #plt.show()

      y = i / z_max * z_total

      hough_spaces = hough_helper_sausage(grad_mag, hough_spaces, y, ellipse_ratio, threshold, rad_max, z_total, theta_max, width)

      pbar.update(1)

    # save the hough spaces to a file
    np.save("hough_spaces_sausage.npy", hough_spaces)
    #return

    with tqdm(total=hough_spaces.shape[0], desc="Post processing hough spaces") as pbar:
      for i, hough_space in enumerate(hough_spaces):  
        #plt.imshow(hough_spaces[i], cmap='jet')
        #plt.show()
        hough_spaces[i] = post_process_hough(hough_space)
        pbar.update(1)

    # convert extrema to 3D points
    points = []
    colors = []

    # get the extrema
    extrema = np.argwhere(hough_spaces > 0)
    with tqdm(total=len(extrema), desc="Converting extrema to 3D points") as pbar:
      for z, a, theta_base  in extrema:
        theta = (theta_base / hough_spaces.shape[2]) * 2 * np.pi
        # get the x and y values
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        points.append([x, y, z])
        colors.append([hough_spaces[z, a, theta_base], 0, 0])
        pbar.update(1)

    # normalize the colors
    colors = np.array(colors)
    colors = colors / np.max(colors)

    #colors[:, 2] = 1 - colors[:, 0]

    color_vals = colors[:, 0]

    cmap = plt.get_cmap('inferno')
    colors = cmap(color_vals)[:, :3]

    # create a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd])

    colors *= 255
    colors = colors.astype(np.uint8)

    # write the points to a ply file
    with open("test_arbitrary_angle_sausage.ply", "w") as file:
      file.write("ply\n")
      file.write("format ascii 1.0\n")
      file.write("element vertex " + str(len(points)) + "\n")
      file.write("property float32 x\n")
      file.write("property float32 y\n")
      file.write("property float32 z\n")
      file.write("property uint8 red\n")
      file.write("property uint8 green\n")
      file.write("property uint8 blue\n")
      file.write("end_header\n")
      for i, point in enumerate(points):
        file.write(str(float(point[1])) + " " + str(float(point[2])) + " " + str(float(point[0])) + " " + str(colors[i][0]) + " " + str(colors[i][1]) + " " + str(colors[i][2]) + "\n")
  

@jit(nopython=True)
def hough_helper_sausage(grad_mag, hough_spaces, y, ellipse_ratio, threshold, rad_max, z_total, theta_max, width):
  # get points where the gradient magnitude is above a threshold
  points = np.argwhere(grad_mag > threshold)
  
  for theta_base, x in points:
    theta = (theta_base / theta_max) * 2 * np.pi if theta_base != 0 else 0

    # loop over all possible radii
    for a in range(np.abs(rad_max - x), rad_max):
      if a == 0:
        continue

      b = a * ellipse_ratio

      x_rel = x - rad_max
      y_rel = (b * np.sqrt(a**2 - x_rel**2)) / a

      z0 = int(y - y_rel)
      z1 = int(y + y_rel)

      y_im = np.sqrt(a**2 - x_rel**2) if a != x_rel else 0

      # get the theta values for the ellipses
      theta0 = int(((np.pi + math.atan2(x_rel, y_im) - theta) % (2 * np.pi))  / (2 * np.pi) * (theta_max)) % theta_max
      theta1 = int(((np.pi + math.atan2(x_rel, -y_im) - theta) % (2 * np.pi)) / (2 * np.pi) * (theta_max)) % theta_max

      if z0 >= 0 and z0 < z_total:
        hough_spaces[z0, a, theta0] += 1
      if z1 >= 0 and z1 < z_total and z1 != z0:
        hough_spaces[z0, a, theta1] += 1

  return hough_spaces
  
import copy
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.206, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    print("Transformation: " + str(transformation))
    source_temp.transform(transformation)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([source_temp, target_temp, coord_frame],
                                  zoom=0.5,
                                  front=[0.9288, -0.2951, -0.2242],
                                  lookat=[0, 1, 1],
                                  up=[0, 0, 1])


def estimate_transformation(source, target, iterations=10):
  # the source and target are hough spaces of shape (theta, a, z)

  # normalize the source and target
  source = source / np.max(source)
  target = target / np.max(target)

  print("Source shape: " + str(source.shape))
  print("Target shape: " + str(target.shape))

  source_points = np.argwhere(source > 0.2)
  target_points = np.argwhere(target > 0.2)

  theta_max = source.shape[0]

  # plot the source and target points in 3d
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='r', label='Source')
  ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='b', label='Target')
  ax.legend()
  plt.show()

  source_points = estimate_transformation_helper(source_points, target_points, theta_max, iterations=iterations)


  
    

  # plot the source and target points and the source points after transformation
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='r', label='Source')
  ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='b', label='Target')
  ax.legend()
  plt.show()

@jit(nopython=True)
def find_corresponding_points(source_points, target_points, theta_max):
  distances = []
  indices = []
  for point in source_points:
        nearest = None
        min_dist = np.inf
        for i, target_point in enumerate(target_points):
          theta_diff = min((point[0] - target_point[0])**2, (point[0] - target_point[0] + theta_max)**2, (point[0] - target_point[0] - theta_max)**2)
          a_diff = (point[1] - target_point[1])**2
          z_diff = (point[2] - target_point[2])**2
          dist = np.sqrt(theta_diff + a_diff + z_diff)
          if dist < min_dist:
            min_dist = dist
            nearest = i
        distances.append(min_dist)
        indices.append(nearest)
  return distances, indices
  

def estimate_transformation_helper(source_points, target_points, theta_max,  iterations=10):

  for _ in range(iterations):
    distances = []
    indices = []

    # find corresponding points
    # for each point in source find the nearest point in target
    # use the nearest neighbor algorithm
    distances, indices = find_corresponding_points(source_points, target_points, theta_max)
    
    #with tqdm(total=len(source_points), desc="Finding corresponding points") as pbar:
    #  for point in source_points:
    #    nearest = None
    #    min_dist = np.inf
    #    for i, target_point in enumerate(target_points):
    #      theta_diff = min((point[0] - target_point[0])**2, (point[0] - target_point[0] + theta_max)**2, (point[0] - target_point[0] - theta_max)**2)
    #      a_diff = (point[1] - target_point[1])**2
    #      z_diff = (point[2] - target_point[2])**2
    #      dist = np.sqrt(theta_diff + a_diff + z_diff)
    #      if dist < min_dist:
    #        min_dist = dist
    #        nearest = i
    #    distances.append(min_dist)
    #    indices.append(nearest)
    #    #corresponding_points.append([point, nearest])
    #    pbar.update(1)

    #nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(target_points)
    #distances, indices = nbrs.kneighbors(source_points)
    #weights = np.array([source[source_points[i][0], source_points[i][1], source_points[i][2]] + target[target_points[indices[i][0]][0], target_points[indices[i][0]][1], target_points[indices[i][0]][2]] for i in range(len(source_points))])
    #weights = weights / np.max(weights)
    #scale = np.average(np.array([source_points[i] / target_points[indices[i]] for i in range(len(source_points))]), axis=0)#, weights=weights)
    scale = np.mean(np.array([target_points[indices[i]] / source_points[i] for i in range(len(source_points))]), axis=0)#, weights=weights)
    print(f"Scale: {scale}")
    source_points[:, 1] = source_points[:, 1] * scale[1]  # scale in y direction
    source_points[:, 2] = source_points[:, 2] * scale[2]  # scale in z direction

    # translation (not in y direction)
    #translation = np.average(np.array([source_points[i] - target_points[indices[i]] for i in range(len(source_points))]), axis=0)#, weights=weights)
    translation = np.mean(np.array([target_points[indices[i]] - source_points[i] for i in range(len(source_points))]), axis=0)#, weights=weights)

    translation_theta = np.mean(np.array([target_points[indices[i]][0] - source_points[i][0] for i in range(len(source_points))]), axis=0)
    translation_theta = theta_max - translation_theta if np.abs(theta_max - translation_theta) < translation_theta else translation_theta
    translation_theta = theta_max + translation_theta if np.abs(theta_max + translation_theta) < translation_theta else translation_theta 
    print(f"Translation: {translation}")
    source_points[:, 0] = (source_points[:, 0] + translation[0]) % theta_max
    source_points[:, 1] = source_points[:, 1] + translation[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='r', label='Source')
    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='b', label='Target')
    #for i in range(len(source_points)):
    #  ax.plot([source_points[i, 0], target_points[indices[i], 0]], [source_points[i, 1], target_points[indices[i], 1]], [source_points[i, 2], target_points[indices[i], 2]], c='g')
    ax.legend()
    plt.show()

  return source_points
    






def estimate_transformation_test(source, target, iterations=10):

  corresponding_points = []

  # find corresponding points
  # for each point in source find the nearest point in target
  # use the nearest neighbor algorithm
  for point in source:
    nearest = (0, 0)
    min_dist = 100000
    for target_point in target:
      #print(f"Point: {point}, Target point: {target_point}")
      dist = np.sqrt((point[0] - target_point[0]) ** 2 + (point[1] - target_point[1]) ** 2)
      if dist < min_dist:
        min_dist = dist
        nearest = target_point
    corresponding_points.append([point, nearest])

  # use the corresponding points to estimate the transformation
  # scale
  scale = np.mean(np.array([point[1] / point[0] for point in corresponding_points]), axis=0)
  # multiply the source points with the scale
  corresponding_points = np.array([point[0] * scale for point in corresponding_points])

  # translation
  translation = np.mean(np.array([(point[1] - point[0]) / 2 for point in corresponding_points]), axis=0)
  #print(f"Translation: {translation}")

  
  #print(f"Scale: {scale}")

  # use the translation and scale on the source points
  source_transformed = np.array([point[0] * scale + translation for point in corresponding_points])

  # plot the source and target points and the source points after transformation
  if iterations == 0:
    #plt.scatter(source_transformed[:, 0], source_transformed[:, 1], c='r', label='Source transformed')
    #plt.scatter([point[1][0] for point in corresponding_points], [point[1][1] for point in corresponding_points], c='b', label='Target')
    #plt.scatter([point[0][0] for point in corresponding_points], [point[0][1] for point in corresponding_points], c='g', label='Source')
    #plt.legend()
    #plt.show()
    return source_transformed
  

  # recursivley iterate
  estimate_transformation(source_transformed, target, iterations - 1)



def test_hough():
  
  # load the hough spaces from a file
  hough_spaces1 = np.load("hough_spaces.npy")
  #hough_spaces1 = hough_spaces1[::2]
  hough_spaces2 = np.load("hough_spaces_60d_orth_voronoi_1440.npy")
  hough_spaces2 = hough_spaces2[::16]
  print(hough_spaces1.shape)
  print(hough_spaces2.shape)

  hough_spaces1 = post_process_hough_3d(hough_spaces1)
  hough_spaces2 = post_process_hough_3d(hough_spaces2)

  estimate_transformation(hough_spaces1, hough_spaces2, iterations=10)

  return

  hough_spaces1 = post_process_hough_3d(hough_spaces1)
  hough_spaces2 = post_process_hough_3d(hough_spaces2)

  for i in range(34, hough_spaces1.shape[0], 10):
    extremas1 = np.argwhere(hough_spaces1[i] > 0)
    extremas2 = np.argwhere(hough_spaces2[i] > 0)
    if len(extremas1) > 0 and len(extremas2) > 0:
      source_transformed = estimate_transformation(extremas1, extremas2, iterations=10)
      if source_transformed is not None:
        # plot the source and target points and the source points after transformation
        plt.scatter(source_transformed[:, 0], source_transformed[:, 1], c='r', label='Source transformed')
        plt.scatter(extremas2[:, 0], extremas2[:, 1], c='b', label='Target')
        plt.scatter(extremas1[:, 0], extremas1[:, 1], c='g', label='Source')
        plt.legend()
  return
  

  #extrema1 = np.load(os.path.join("..", "post_processed_mirror_60d.npy"))
  #extrema2 = np.load(os.path.join("..", "post_processed_mirror_90d.npy"))

  #hough_spaces1 = np.zeros((1440, 960, 1080))
  #hough_spaces2 = np.zeros((1440, 960, 1080))

  #hough_spaces1[extrema1[:, 0], extrema1[:, 1], extrema1[:, 2]] = 1
  #hough_spaces2[extrema2[:, 0], extrema2[:, 1], extrema2[:, 2]] = 1

  # source and target are images with possibly different sizes
  extrema_source = np.argwhere(hough_spaces1[100] > 0)
  #extrema_target = np.argwhere(target > 0)
  extrema_target = np.argwhere(hough_spaces2[100] > 0)
  # translate the target points by 100
  #extrema_target = extrema_target + 20

  estimate_transformation(extrema_source, extrema_target, iterations=10)
  return

  points1 = []
  points2 = []

  for theta_base, a, z  in extrema1:
      theta = (theta_base / 1440) * 2 * np.pi
      # get the x and y values
      x = a * np.cos(theta)
      y = a * np.sin(theta)
      points1.append([x, y, z])

  for theta_base, a, z  in extrema2:
      theta = (theta_base / 1440) * 2 * np.pi
      # get the x and y values
      x = a * np.cos(theta)
      y = a * np.sin(theta)
      points2.append([x, y, z])

  # use the hough spaces as source and target point clouds
  source = o3d.geometry.PointCloud()
  source.points = o3d.utility.Vector3dVector(points1)

  #target = o3d.geometry.PointCloud()
  #target.points = o3d.utility.Vector3dVector(points2)
  target = o3d.io.read_point_cloud("point_cloud_1080_bam.ply")
  # turn by 180 degrees
  target.rotate([[1, 0, 0], [0, -1, 0], [0, 0, -1]])



  # load two point clouds
  #source = o3d.io.read_point_cloud("75d_honey_thresholded.ply")
  #source.scale(0.82, center=source.get_center())
  #target = o3d.io.read_point_cloud("60d_orthographic_voronoi_1440.ply")

  # iterative closest point algorithm
  
  trans_init = np.asarray([[1.4, 0, 0, 0],
                            [0, 1.4, 0, 0],
                            [0, 0, 1.4, 0],
                            [0, 0, 0, 1]])
  print("Initial alignment")
  draw_registration_result(source, target, trans_init)
  print("Apply point-to-point ICP")

  estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
  estimation_method.with_scaling = True

  convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)

  reg_p2p = o3d.pipelines.registration.registration_icp(
      source, target, 50, estimation_method=estimation_method
  )
  print(reg_p2p)
  draw_registration_result(source, target, reg_p2p.transformation)




  """
  # load the hough spaces from a file
  hough_spaces1 = np.load("hough_spaces_75d_honey_360.npy")
  hough_spaces1 = hough_spaces1[::4]
  hough_spaces2 = np.load("hough_spaces_60d_orth_voronoi_1440.npy")
  hough_spaces2 = hough_spaces2[::16]

  print(hough_spaces1.shape)
  print(hough_spaces2.shape)

  hough_spaces1 = post_process_hough_3d(hough_spaces1)
  hough_spaces2 = post_process_hough_3d(hough_spaces2)

  # choose 3 random hough spaces
  rand1 = random.randint(0, hough_spaces1.shape[0] - 1)
  rand2 = random.randint(0, hough_spaces1.shape[0] - 1)
  rand3 = random.randint(0, hough_spaces1.shape[0] - 1)

  hough_space1_1 = hough_spaces1[rand1]
  hough_space1_2 = hough_spaces1[rand2]
  hough_space1_3 = hough_spaces1[rand3]

  hough_space2_1 = hough_spaces2[rand1]
  hough_space2_2 = hough_spaces2[rand2]
  hough_space2_3 = hough_spaces2[rand3]

  shape1 = np.max([hough_space1_1.shape[0], hough_space2_1.shape[0]])
  shape2 = np.max([hough_space1_1.shape[1], hough_space2_1.shape[1]])

  shape = (shape1, shape2)
  print(shape)

  shape1 = hough_space1_1.shape
  shape2 = hough_space2_1.shape

  hough_diff1 = np.zeros((shape[0], shape[1], 3))
  hough_diff2 = np.zeros((shape[0], shape[1], 3))
  hough_diff3 = np.zeros((shape[0], shape[1], 3))

  hough_diff1[:shape1[0], :shape1[1], :] = np.repeat(hough_space1_1[:, :, np.newaxis], 3, axis=2)
  hough_diff2[:shape1[0], :shape1[1], :] = np.repeat(hough_space1_2[:, :, np.newaxis], 3, axis=2)
  hough_diff3[:shape1[0], :shape1[1], :] = np.repeat(hough_space1_3[:, :, np.newaxis], 3, axis=2)

  hough_diff1[:hough_space2_1.shape[0], :hough_space2_1.shape[1], 0] -= hough_space2_1
  hough_diff2[:hough_space2_2.shape[0], :hough_space2_2.shape[1], 0] -= hough_space2_2
  hough_diff3[:hough_space2_3.shape[0], :hough_space2_3.shape[1], 0] -= hough_space2_3

  hough_diff1[:hough_space2_1.shape[0], :hough_space2_1.shape[1], 2] = hough_space2_1
  hough_diff2[:hough_space2_2.shape[0], :hough_space2_2.shape[1], 2] = hough_space2_2
  hough_diff3[:hough_space2_3.shape[0], :hough_space2_3.shape[1], 2] = hough_space2_3

  hough_diff1[:,:, 0] = 0#np.abs(hough_diff1[:,:, 0])
  hough_diff2[:,:, 0] = 0#np.abs(hough_diff2[:,:, 0])
  hough_diff3[:,:, 0] = 0#np.abs(hough_diff3[:,:, 0])




  # overlay them
  fig, axs = plt.subplots(2, 3)
  axs[0, 0].imshow(hough_space1_1, cmap='inferno')
  axs[0, 1].imshow(hough_space1_2, cmap='inferno')
  axs[0, 2].imshow(hough_space1_3, cmap='inferno')

  axs[1, 0].imshow(hough_space2_1, cmap='inferno')
  axs[1, 1].imshow(hough_space2_2, cmap='inferno')
  axs[1, 2].imshow(hough_space2_3, cmap='inferno')

  # title is the frame number
  axs[0, 0].set_title(f"Frame {rand1}")
  axs[0, 1].set_title(f"Frame {rand2}")
  axs[0, 2].set_title(f"Frame {rand3}")

  plt.show()

  plt.imshow(hough_diff1, interpolation='none')
  plt.colorbar()
  plt.show()

  plt.imshow(hough_diff2, interpolation='none')
  plt.colorbar()
  plt.show()

  plt.imshow(hough_diff3, interpolation='none')
  plt.colorbar()
  plt.show()
  """




def test_hough2():
  # load the hough spaces from a file
  hough_spaces = np.load("hough_spaces_60d_orth_voronoi_1440.npy")
  #hough_spaces = np.load("hough_spaces_sausage.npy")

  # only take every 5th frame
  hough_spaces = hough_spaces[::4]

  # reorder from z, a, theta to theta, a, z
  #hough_spaces = np.moveaxis(hough_spaces, 0, 2) 
  #hough_spaces = np.moveaxis(hough_spaces, 0, 1)

  print(hough_spaces.shape)


  hough_spaces = post_process_hough_3d(hough_spaces)

  #for i, hough_space in enumerate(hough_spaces):
  #  if i % 5 == 0:
  #    plt.imshow(hough_space, cmap='inferno')
  #    plt.colorbar()
  #    plt.show()



  # post process the hough spaces
  #with tqdm(total=hough_spaces.shape[0], desc="Post processing hough spaces") as pbar:
  #  for i, hough_space in enumerate(hough_spaces):  
  #    hough_spaces[i] = post_process_hough(hough_space)
  #    pbar.update(1)

  # convert extrema to 3D points
  points = []
  colors = []

  # get the extrema
  extrema = np.argwhere(hough_spaces > 0)
  with tqdm(total=len(extrema), desc="Converting extrema to 3D points") as pbar:
    for theta_base, a, z  in extrema:
      theta = (theta_base / hough_spaces.shape[0]) * 2 * np.pi
      # get the x and y values
      x = a * np.cos(theta)
      y = a * np.sin(theta)
      points.append([x, y, z])
      colors.append([hough_spaces[theta_base, a, z], 0, 0])
      pbar.update(1)

  # get the extrema
  #extrema = np.argwhere(hough_spaces > 0)
  #with tqdm(total=len(extrema), desc="Converting extrema to 3D points") as pbar:
  #  for theta_base, a, z  in extrema:
  #    theta = (theta_base / hough_spaces.shape[0]) * 2 * np.pi
#
  #    # get the x and y values
  #    x = a * np.cos(theta)
  #    y = a * np.sin(theta)
  #    points.append([x, y, z])
  #    colors.append([hough_spaces[theta_base, a, z], 0, 0])
  #    pbar.update(1)

  # normalize the colors
  colors = np.array(colors)
  colors = colors / np.max(colors)

  #colors[:, 2] = 1 - colors[:, 0]

  color_vals = colors[:, 0]

  cmap = plt.get_cmap('inferno')
  colors = cmap(color_vals)[:, :3]

  # create a point cloud
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  pcd.colors = o3d.utility.Vector3dVector(colors)

  o3d.visualization.draw_geometries([pcd])

  
  colors *= 255
  colors = colors.astype(np.uint8)
  """
  # write the points to a ply file
  with open("test_arbitrary_angle_sausage.ply", "w") as file:
    file.write("ply\n")
    file.write("format ascii 1.0\n")
    file.write("element vertex " + str(len(points)) + "\n")
    file.write("property float32 x\n")
    file.write("property float32 y\n")
    file.write("property float32 z\n")
    file.write("property uint8 red\n")
    file.write("property uint8 green\n")
    file.write("property uint8 blue\n")
    file.write("end_header\n")
    for i, point in enumerate(points):
      file.write(str(float(point[1])) + " " + str(float(point[2])) + " " + str(float(point[0])) + " " + str(colors[i][0]) + " " + str(colors[i][1]) + " " + str(colors[i][2]) + "\n")
  
  
  theta1 = random.randint(0, hough_spaces.shape[0] - 1)
  theta2 = random.randint(0, hough_spaces.shape[0] - 1)
  theta3 = random.randint(0, hough_spaces.shape[0] - 1)

  # choose 3 random hough spaces
  hough_space1 = hough_spaces[theta1]
  hough_space2 = hough_spaces[theta2]
  hough_space3 = hough_spaces[theta3]

  theta1 = (theta1 / hough_spaces.shape[0]) * 360
  theta2 = (theta2 / hough_spaces.shape[0]) * 360
  theta3 = (theta3 / hough_spaces.shape[0]) * 360


  # post process the hough spaces
  hough_space_processed1 = post_process_hough(hough_space1)
  hough_space_processed2 = post_process_hough(hough_space2)
  hough_space_processed3 = post_process_hough(hough_space3)

  # plot the hough spaces in a 2x3 grid
  fig, axs = plt.subplots(2, 3)
  axs[0, 0].imshow(hough_space1, cmap='inferno')
  axs[0, 1].imshow(hough_space2, cmap='inferno')
  axs[0, 2].imshow(hough_space3, cmap='inferno')
  axs[1, 0].imshow(hough_space_processed1, cmap='inferno')
  axs[1, 1].imshow(hough_space_processed2, cmap='inferno')
  axs[1, 2].imshow(hough_space_processed3, cmap='inferno')
  axs[0, 0].set_title(f"{theta1}°")
  axs[0, 1].set_title(f"{theta2}°")
  axs[0, 2].set_title(f"{theta3}°")

  plt.show()

  """




def main():
  #test_img = cv2.imread("test/cylinder_test.py", 0)
  #plt.imshow(test_img, cmap="gray")
  #plt.show()
#
  test_hough()
  return

  #dir = os.path.join("..", "scratch", "diagonal_angle", "cropped")
  #dir = os.path.join("..", "scratch", "60d_100mm_10mm_voronoi", "rows")
  #dir = os.path.join("..", "scratch", "circle_moving")
  dir = os.path.join("..", "scratch", "75d_100mm_10mm_honey")
  dir = os.path.join("..", "scratch", "full_1440")
  full_hough_to_ply(dir)

if __name__ == "__main__":
  main()