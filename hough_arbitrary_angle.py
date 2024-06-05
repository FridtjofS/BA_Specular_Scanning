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
  ellipse_ratio = 0.5#.258
  threshold = 20

  frame_names = os.listdir(dir)
  # sort the frames named 0.png, 1.png, ..., 100.png, ...
  # filter out every frame thats not an int
  frame_names = [frame for frame in frame_names if frame.split('.')[0].isdigit()]
  frame_names = sorted(frame_names, key=lambda x: int(x.split('.')[0]))
  #frame_names = frame_names[::80]
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
    #np.save("hough_spaces_voronoi_70d.npy", hough_spaces)
    #return

    plt.imshow(hough_spaces[0], cmap='jet')
    plt.show()


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

  #return hough_spaces

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

  source_points = np.argwhere(source > 0)
  target_points = np.argwhere(target > 0)

  if len(source_points) > len(target_points):
    temp = source_points
    source_points = target_points
    target_points = temp

  theta_max = source.shape[0]

  # plot the source and target points in 3d
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='r', label='Source')
  ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='b', label='Target')
  ax.legend()
  plt.show()

  source_points = estimate_transformation_helper(source_points, target_points, theta_max, iterations=iterations)

  # plot the source and target points in 3d
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='r', label='Source')
  ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='b', label='Target')
  ax.legend()
  plt.show()

  return source_points, target_points
  
    

  # plot the source and target points and the source points after transformation
  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection='3d')
  #ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='r', label='Source')
  #ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='b', label='Target')
  #ax.legend()
  #plt.show()

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
  
def filter_points(source, target, indices, distances):
    # Dictionary to store the minimum distances and corresponding indices
    min_distances = {}
    
    for i, (idx, dist) in enumerate(zip(indices, distances)):
        if idx not in min_distances or dist < min_distances[idx][1]:
            min_distances[idx] = (i, dist)
    
    # Lists to store the filtered points and corresponding distances
    new_source = []
    new_target = []
    new_indices = []
    new_distances = []

    for idx, (i, dist) in min_distances.items():
        new_source.append(source[i])
        new_target.append(target[idx])
        new_indices.append(idx)
        new_distances.append(dist)

    new_source = np.array(new_source)
    new_target = np.array(new_target)
    new_indices = np.array(new_indices)
    new_distances = np.array(new_distances)
    
    return new_source, new_target, new_indices, new_distances

def estimate_transformation_helper(source_points, target_points, theta_max,  iterations=10):

  scale_y = 1
  scale_z = 1
  translation_z = 0
  translation_y = 0

  for iter in range(iterations):
    distances = []
    indices = []

    # find corresponding points
    # for each point in source find the nearest point in target
    # use the nearest neighbor algorithm
    distances, indices = find_corresponding_points(source_points, target_points, theta_max)

    distances = np.array(distances)
    indices = np.array(indices)

    # only take the indices where the distance is the smallest for the index

    #source_points_, target_points_, indices_, distances = filter_points(source_points, target_points, indices, distances)

    #print(f"Number of corresponding points: (before {len(source_points)}) after: {len(source_points_)}, indices: {len(indices_)}")



    # normalize the distances
    #distances = distances / np.max(distances)

    # only take the target and source points where their distance is below a threshold
    threshold = 30.0
    source_points_thresh = source_points[distances < threshold]

    print(f"Number of points: {len(source_points_thresh)}")
    indices = indices[distances < threshold].astype(int)
    #indices = indices[distances < threshold]
    #target_points_thresh = target_points_[indices][distances < threshold]

    #center_source = np.mean(source_points_thresh, axis=0)
    #center_target = np.mean(target_points_thresh, axis=0)

    #source_points_thresh = source_points_thresh - center_source
    #target_points_thresh = target_points_thresh - center_target

    distances = distances[distances < threshold]

    print(f"Error: {np.mean(distances)}")


    distances = distances / np.max(distances)

    #weights = np.array([source_points[i][0] + target_points[indices[i]][0] for i in range(len(source_points))])
    #weights = weights / np.max(weights)
    #weights = weights / (distances + 1e-6)
    
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
    #scale = np.average(np.array([target_points[indices[i]] / source_points[i] for i in range(len(source_points))]), axis=0, weights=distances)
    scale = np.average(np.array([target_points[indices[i]] / source_points_thresh[i] for i in range(len(source_points_thresh))]), axis=0, weights=distances)
    #epsilon = 1e-6
    #scale = np.mean(np.array([(np.abs(target_points[indices[i]] - target_points[indices[i+1]]) / (np.abs(source_points_thresh[i] - source_points_thresh[i+1]) + epsilon)) for i in range(0, len(source_points_thresh) - 1, 2)]), axis=0)
    #scale = np.mean(np.array([target_points[indices[i]] / source_points[i] for i in range(len(source_points))]), axis=0, weights=weights)
    print(f"Scale: {scale}")
    if scale[1] < 0.5 or scale[2] < 0.5 or scale[1] > 2 or scale[2] > 2 or np.isnan(scale[1]) or np.isnan(scale[2]):
      print(f"Scale too small or too large, setting to 1... Scale: {scale}")
      scale = np.array([1, 1, 1])
  



    source_points[:, 1] = source_points[:, 1] * scale[1] # scale in y direction
    source_points[:, 2] = source_points[:, 2] * scale[2] # scale in z direction

    scale_y *= scale[1]
    scale_z *= scale[2]

    print(f"Scale: {scale}, Scale radius: {scale_y}, Scale height: {scale_z}")

    distances, indices = find_corresponding_points(source_points, target_points, theta_max)

    distances = np.array(distances)
    indices = np.array(indices)

    #source_points_, target_points_, indices, distances = filter_points(source_points, target_points, indices, distances)

    # only take the target and source points where their distance is below a threshold
    source_points_thresh = source_points[distances < threshold]
    indices = indices[distances < threshold].astype(int)
    #target_points_thresh = target_points[indices][distances < threshold]

    #center_source = np.mean(source_points_thresh, axis=0)
    #center_target = np.mean(target_points_thresh, axis=0)
#
    #source_points_thresh = source_points_thresh - center_source
    #target_points_thresh = target_points_thresh - center_target

    distances = distances[distances < threshold]
    distances = distances / np.max(distances)


    # translation (not in y direction)
    #translation = np.average(np.array([target_points[indices[i]] - source_points[i] for i in range(len(source_points))]), axis=0, weights=distances)
    translation = np.average(np.array([target_points[indices[i]] - source_points_thresh[i] for i in range(len(source_points_thresh))]), axis=0, weights=distances)
    #translation = np.mean(np.array([target_points[indices[i]] - source_points[i] for i in range(len(source_points))]), axis=0)#, weights=weights)

    #translation_theta = np.mean(np.array([target_points[indices[i]][0] - source_points[i][0] for i in range(len(source_points))]), axis=0)
    translation_theta = np.average(np.array([target_points[indices[i]][0] - source_points_thresh[i][0] for i in range(len(source_points_thresh))]), axis=0, weights=distances)
    translation_theta = theta_max - translation_theta if np.abs(theta_max - translation_theta) < translation_theta else translation_theta
    translation_theta = theta_max + translation_theta if np.abs(theta_max + translation_theta) < translation_theta else translation_theta 
    
    translation[0] = translation_theta

    if np.isnan(translation[0]) or np.isnan(translation[2]):
      translation = np.array([0, 0, 0])
      print(f"Translation is nan, setting to 0... Translation: {translation}")
    
    source_points[:, 0] = (source_points[:, 0] + translation[0]) % theta_max
    source_points[:, 2] = source_points[:, 2] + translation[2]

    translation_z = (translation_z + translation[0]) % theta_max
    translation_y += translation[2]

    print(f"Translation: {translation}, Translation theta: {translation_z}, Translation height: {translation_y}")

    if iter == 0:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='r', label='Source')
      ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], c='b', label='Target')
      for i in range(len(source_points_thresh)):
        #ax.plot([source_points[i, 0], target_points[indices[i], 0]], [source_points[i, 1], target_points[indices[i], 1]], [source_points[i, 2], target_points[indices[i], 2]], c='g')
        # only plot the lines where the distance is below a threshold

        ax.plot([source_points_thresh[i, 0], target_points[indices[i], 0]], [source_points_thresh[i, 1], target_points[indices[i], 1]], [source_points_thresh[i, 2], target_points[indices[i], 2]], c='g')

        #if distances[i] < threshold:
        #  ax.plot([source_points[i, 0], target_points[indices[i], 0]], [source_points[i, 1], target_points[indices[i], 1]], [source_points[i, 2], target_points[indices[i], 2]], c='g')
      ax.legend()
      plt.show()

  return source_points#, target_points
    






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

def align_hough_spaces_gui(hough_spaces1, hough_spaces2):
  import sys
  from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QGridLayout
  from PyQt6.QtWidgets import QLabel, QPushButton, QCheckBox, QComboBox, QFileDialog, QButtonGroup, QSpinBox
  from PyQt6 import QtGui, QtCore
  import open3d as o3d
  import win32gui
  import numpy as np
  import matplotlib.pyplot as plt
  import os
  from PIL import Image, ImageQt

  # using pyqt6
  app = QApplication([])
  window = QWidget()
  layout = QVBoxLayout()

  # show both hough spaces overlayed in a pixmap
  rand = np.random.randint(0, hough_spaces1.shape[0])
  hough_space1 = hough_spaces1[rand]
  hough_space2 = hough_spaces2[rand]

  hough_space1 = hough_space1 / np.max(hough_space1)
  hough_space2 = hough_space2 / np.max(hough_space2)

  hough_space1 = hough_space1 * 255
  hough_space2 = hough_space2 * 255

  # set hough space 1 to red and hough space 2 to blue
  hough_space1 = np.stack((hough_space1, np.zeros(hough_space1.shape), np.zeros(hough_space1.shape)), axis=-1)
  hough_space2 = np.stack((np.zeros(hough_space2.shape), np.zeros(hough_space2.shape), hough_space2), axis=-1)

  hough_space = hough_space1 + hough_space2

  hough_space = hough_space.astype(np.uint8)

  # convert the hough space to an image
  hough_space_img = ImageQt.ImageQt(Image.fromarray(hough_space))

  pixmap = QtGui.QPixmap.fromImage(hough_space_img)

  label = QLabel()
  label.setPixmap(pixmap)
  layout.addWidget(label)

  # add a button to start the alignment
  scale_x = QSlider(QtCore.Qt.Orientation.Horizontal)
  scale_x.setRange(50, 200)
  scale_x.setValue(100)
  scale_x.setTickInterval(1)
  scale_x.setTickPosition(QSlider.TickPosition.TicksBelow)
  layout.addWidget(scale_x)

  scale_y = QSlider(QtCore.Qt.Orientation.Horizontal)
  scale_y.setRange(50, 200)
  scale_y.setValue(100)
  scale_y.setTickInterval(1)
  scale_y.setTickPosition(QSlider.TickPosition.TicksBelow)
  layout.addWidget(scale_y)

  translation_x = QSlider(QtCore.Qt.Orientation.Horizontal)
  translation_x.setRange(-300, 300)
  translation_x.setValue(0)
  translation_x.setTickInterval(1)
  translation_x.setTickPosition(QSlider.TickPosition.TicksBelow)
  layout.addWidget(translation_x)

  # add a button to start the alignment
  button = QPushButton("Align")
  layout.addWidget(button)
  button.clicked.connect(lambda: label.setPixmap(adjust_scale_translation(hough_space1, hough_space2, scale_x.value() / 100, scale_y.value() / 100, translation_x.value())))

  def adjust_scale_translation(hough_space1, hough_space2, scale_x, scale_y, translation_x):
    
    hough_space = np.zeros(hough_space2.shape)

    for a in range(hough_space2.shape[0]):
      for z in range(hough_space2.shape[0]):
        a2 = int(a * scale_y)
        z2 = int(z * scale_x + translation_x)
        if a2 >= 0 and a2 < hough_space1.shape[0] and z2 >= 0 and z2 < hough_space1.shape[1]:
          hough_space[a, z] = hough_spaces1[a2, z2]

    hough_space = np.stack((hough_space, np.zeros(hough_space.shape), hough_spaces2), axis=-1)

    hough_space = hough_space / np.max(hough_space)
    hough_space = hough_space * 255

    hough_space = hough_space.astype(np.uint8)

    # convert the hough space to an image
    hough_space_img = ImageQt.ImageQt(Image.fromarray(hough_space))

    pixmap = QtGui.QPixmap.fromImage(hough_space_img)

    return pixmap

  



  # start the application
  window.setLayout(layout)
  window.show()
  sys.exit(app.exec())






def align_hough_spaces_manually(hough_spaces1, hough_spaces2):
  # show both hough spaces and let the user select the corresponding points in both hough spaces
  # the user has to select the same amount of points in both hough spaces (at least 2 points)

  correspondences = []


  for i in range(4):
    # get user input to click on the points
    fig, ax = plt.subplots(1, 2)
    rand = np.random.randint(0, hough_spaces1.shape[0])
    ax[0].imshow(hough_spaces1[rand], cmap='jet')
    ax[1].imshow(hough_spaces2[rand], cmap='jet')
    point1 = plt.ginput(n=1, timeout=0)
    point2 = plt.ginput(n=1, timeout=0)
    print(f"Point 1: {point1}, Point 2: {point2}")
    plt.close(fig)
    correspondences.append(point1)
    correspondences.append(point2)

    #fig, ax = plt.subplots(1, 2)
    #rand = np.random.randint(0, hough_spaces1.shape[0])
    #ax[0].imshow(hough_spaces1[rand], cmap='jet')
    #ax[1].imshow(hough_spaces2[rand], cmap='jet')
    #point1 = plt.ginput(n=1, timeout=0)
    #point2 = plt.ginput(n=1, timeout=0)
    #print(f"Point 1: {point1}, Point 2: {point2}")
    #plt.close(fig)
    #correspondences.append(point1)
    #correspondences.append(point2)

  
  scales = []
  translations = []

  # find the transformations between the points
  # first scale in x and y direction by measuring the distance between two points in the x and y direction
  # then find the translation by measuring the distance between the two points in the x and y direction
  for i in range(0, len(correspondences), 4):

    corr_h1_1 = correspondences[i][0]
    corr_h2_1 = correspondences[i + 1][0]
    corr_h1_2 = correspondences[i + 2][0]
    corr_h2_2 = correspondences[i + 3][0]

    distance_x1 = np.abs(corr_h1_1[0] - corr_h1_2[0])
    distance_x2 = np.abs(corr_h2_1[0] - corr_h2_2[0])
    distance_y1 = np.abs(corr_h1_1[1] - corr_h1_2[1])
    distance_y2 = np.abs(corr_h2_1[1] - corr_h2_2[1])

    scale_x = distance_x1 / distance_x2
    scale_y = distance_y1 / distance_y2
    scales.append([scale_x, scale_y])
  	
    translation_x1 = corr_h1_1[0] - (corr_h2_1[0] * scale_x)
    translations.append(translation_x1)
    translation_x2 = corr_h1_2[0] - (corr_h2_2[0] * scale_x)
    translations.append(translation_x2)

  # get the average scale and translation
  print(f"Scales: {scales}, Translations: {translations}")

  scale = np.mean(scales, axis=0)
  translation = np.mean(translations, axis=0)
  print(f"Scale: {scale}, Translation: {translation}")

  hough_spaces = fuse_hough_spaces(hough_spaces1, hough_spaces2, scale, translation)

  plt.imshow(hough_spaces[10], cmap='jet')
  plt.show()
  plt.imshow(hough_spaces[100], cmap='jet')
  plt.show()
  plt.imshow(hough_spaces[120], cmap='jet')
  plt.show()

  hough_spaces = post_process_hough_3d(hough_spaces)

  hough_spaces = hough_spaces / np.max(hough_spaces)

  extrema = np.argwhere(hough_spaces > 0)

  points = []
  colors = []

  for theta_base, a, z  in extrema:
        theta = (theta_base / (hough_spaces.shape[0] - 1)) * 2 * np.pi
        # get the x and y values
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        points.append([x, y, z])
        colors.append([hough_spaces[theta_base, a, z], 0, 0])

  point_cloud = o3d.geometry.PointCloud()
  point_cloud.points = o3d.utility.Vector3dVector(points)
  #cmap = plt.get_cmap('inferno')
  #colors = cmap(colors)[:, :3]
  point_cloud.colors = o3d.utility.Vector3dVector(colors)

  o3d.visualization.draw_geometries([point_cloud])

  return



  hough_spaces1 = post_process_hough_3d(hough_spaces1)
  hough_spaces2 = post_process_hough_3d(hough_spaces2)

  extrema1 = np.argwhere(hough_spaces1 > 0)
  extrema2 = np.argwhere(hough_spaces2 > 0)

  points1 = []

  for theta_base, a, z  in extrema1:
        theta = (theta_base / hough_spaces1.shape[0]) * 2 * np.pi
        # get the x and y values
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        points1.append([x, y, z])

  points2 = []

  for theta_base, a, z  in extrema2:
        z = z * scale[0] + translation
        a = a * scale[1]

        theta = (theta_base / hough_spaces2.shape[0]) * 2 * np.pi
        # get the x and y values
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        points2.append([x, y, z])

  point_cloud1 = o3d.geometry.PointCloud()
  point_cloud1.points = o3d.utility.Vector3dVector(points1)
  point_cloud1.paint_uniform_color([1, 0, 0])

  point_cloud2 = o3d.geometry.PointCloud()
  point_cloud2.points = o3d.utility.Vector3dVector(points2)
  point_cloud2.paint_uniform_color([0, 0, 1])

  o3d.visualization.draw_geometries([point_cloud1, point_cloud2])




  # apply the scale and translation to the source points
  #source_points = hough_spaces1 * scale + translation
#
  ## plot the source and target points and the source points after transformation
  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection='3d')
  #ax.scatter(source_points[:, 0], source_points[:, 1], source_points[:, 2], c='r', label='Source')
  #ax.scatter(hough_spaces2[:, 0], hough_spaces2[:, 1], hough_spaces2[:, 2], c='b', label='Target')
  #ax.legend()
  #plt.show()

def fuse_hough_spaces(hough_spaces1, hough_spaces2, scale, translation):
  #hough_spaces1 = hough_spaces1 / np.max(hough_spaces1)
  #hough_spaces2 = hough_spaces2 / np.max(hough_spaces2)
  for a in range(hough_spaces2.shape[1]):
    for z in range(hough_spaces2.shape[2]):
      a2 = int(a * scale[1])
      z2 = int(z * scale[0] + translation)
      if a2 >= 0 and a2 < hough_spaces1.shape[1] and z2 >= 0 and z2 < hough_spaces1.shape[2]:
        hough_spaces2[:, a, z] += hough_spaces1[:, a2, z2]
  
  return hough_spaces2

def test_hough():
  
  # load the hough spaces from a file
  #hough_spaces1 = np.load("hough_spaces_voronoi_60d.npy")
  #hough_spaces1 = hough_spaces1[::4]
  #hough_spaces2 = np.load("hough_spaces_voronoi_90d.npy")
  #hough_spaces2 = hough_spaces2[::4]
  ##hough_spaces2 = hough_spaces1
  #print(hough_spaces1.shape)
  #print(hough_spaces2.shape)
#
  #hough_spaces1 = post_process_hough_3d(hough_spaces1)
  #hough_spaces2 = post_process_hough_3d(hough_spaces2)

  hough_spaces1 = np.load("post_processed_voronoi_60d.npy", hough_spaces1)
  hough_spaces2 = np.load("post_processed_voronoi_90d.npy", hough_spaces2)


  #align_hough_spaces_manually(hough_spaces1, hough_spaces2)

  #align_hough_spaces_gui(hough_spaces1, hough_spaces2)

  #return

  #hough_spaces1 = np.zeros((100, 100, 100))
  #hough_spaces2 = np.zeros((100, 100, 100))
#
  ## set some points in the hough spaces
  #for i in range(20, 80, 10):
  #  for j in range(20, 80, 10):
  #    for k in range(20, 80, 10):
  #      hough_spaces1[i, j, k] = 1
  #      hough_spaces2[i - 20, j, k] = 1


  hough_spaces1 = hough_spaces1 / np.max(hough_spaces1)
  hough_spaces2 = hough_spaces2 / np.max(hough_spaces2)

  extrema1 = np.argwhere(hough_spaces1 > 0)
  extrema2 = np.argwhere(hough_spaces2 > 0)


  points1 = []

  for theta_base, a, z  in extrema1:
        theta = (theta_base / (hough_spaces1.shape[0] - 1)) * 2 * np.pi
        # get the x and y values
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        points1.append([x, y, z])

  points2 = []

  for theta_base, a, z  in extrema2:
        theta = (theta_base / (hough_spaces2.shape[0] - 1)) * 2 * np.pi
        # get the x and y values
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        points2.append([x, y, z])

  point_cloud1 = o3d.geometry.PointCloud()
  point_cloud1.points = o3d.utility.Vector3dVector(points1)
  point_cloud1.paint_uniform_color([1, 0, 0])

  point_cloud2 = o3d.geometry.PointCloud()
  point_cloud2.points = o3d.utility.Vector3dVector(points2)
  point_cloud2.paint_uniform_color([0, 0, 1])

  o3d.visualization.draw_geometries([point_cloud1, point_cloud2])

  extrema1, extrema2 = estimate_transformation(hough_spaces1, hough_spaces2, iterations=50)

  #extrema1 = np.argwhere(hough_spaces1 > 0)
  #extrema2 = np.argwhere(hough_spaces2 > 0)

  points1 = []

  for theta_base, a, z  in extrema1:
        theta = (theta_base / hough_spaces1.shape[0]) * 2 * np.pi
        # get the x and y values
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        points1.append([x, y, z])

  points2 = []

  for theta_base, a, z  in extrema2:
        theta = (theta_base / hough_spaces2.shape[0]) * 2 * np.pi
        # get the x and y values
        x = a * np.cos(theta)
        y = a * np.sin(theta)
        points2.append([x, y, z])

  point_cloud1 = o3d.geometry.PointCloud()
  point_cloud1.points = o3d.utility.Vector3dVector(points1)
  point_cloud1.paint_uniform_color([1, 0, 0])

  point_cloud2 = o3d.geometry.PointCloud()
  point_cloud2.points = o3d.utility.Vector3dVector(points2)
  point_cloud2.paint_uniform_color([0, 0, 1])

  o3d.visualization.draw_geometries([point_cloud1, point_cloud2])

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
  """

  #extrema1 = np.load(os.path.join("..", "post_processed_mirror_60d.npy"))
  #extrema2 = np.load(os.path.join("..", "post_processed_mirror_90d.npy"))

  #hough_spaces1 = np.zeros((1440, 960, 1080))
  #hough_spaces2 = np.zeros((1440, 960, 1080))

  #hough_spaces1[extrema1[:, 0], extrema1[:, 1], extrema1[:, 2]] = 1
  #hough_spaces2[extrema2[:, 0], extrema2[:, 1], extrema2[:, 2]] = 1

  ## source and target are images with possibly different sizes
  #extrema_source = np.argwhere(hough_spaces1[100] > 0)
  ##extrema_target = np.argwhere(target > 0)
  #extrema_target = np.argwhere(hough_spaces2[100] > 0)
  ## translate the target points by 100
  ##extrema_target = extrema_target + 20
#
  #estimate_transformation(extrema_source, extrema_target, iterations=10)
  #return

  points1 = []
  points2 = []

  for theta_base, a, z  in extrema1:
      theta = (theta_base / hough_spaces1.shape[0]) * 2 * np.pi
      # get the x and y values
      x = a * np.cos(theta)
      y = a * np.sin(theta)
      points1.append([x, y, z])

  for theta_base, a, z  in extrema2:
      theta = (theta_base / hough_spaces2.shape[0]) * 2 * np.pi
      # get the x and y values
      x = a * np.cos(theta)
      y = a * np.sin(theta)
      points2.append([x, y, z])

  # use the hough spaces as source and target point clouds
  source = o3d.geometry.PointCloud()
  source.points = o3d.utility.Vector3dVector(points1)

  target = o3d.geometry.PointCloud()
  target.points = o3d.utility.Vector3dVector(points2)
  #target = o3d.io.read_point_cloud("point_cloud_1080_bam.ply")
  ## turn by 180 degrees
  #target.rotate([[1, 0, 0], [0, -1, 0], [0, 0, -1]])



  # load two point clouds
  #source = o3d.io.read_point_cloud("75d_honey_thresholded.ply")
  #source.scale(0.82, center=source.get_center())
  #target = o3d.io.read_point_cloud("60d_orthographic_voronoi_1440.ply")

  # iterative closest point algorithm
  
  trans_init = np.asarray([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
  print("Initial alignment")
  draw_registration_result(source, target, trans_init)
  print("Apply point-to-point ICP")

  estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
  estimation_method.with_scaling = True

  #convergence_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)

  reg_p2p = o3d.pipelines.registration.registration_icp(
      source, target, 50, estimation_method=estimation_method#, criteria=convergence_criteria
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

from scipy.optimize import least_squares

def optimization_test():
  np.random.seed(0)

  x = np.random.randint(0, 100, 100) # sample dataset for independent variables
  y = np.random.randint(0, 100, 100) # sample dataset for dependent variables

  def func(theta, x, y):
      # Return residual = fit-observed
      return (theta[0]*x**2 + theta[1]*x + theta[2]) - y

  # Initial parameter guess
  theta0 = np.array([0.5, -0.1, 0.3])

  # Compute solution providing initial guess theta0, x input, and y input
  sol = least_squares(func, theta0, args=(x,y), verbose=2)
  print(sol)


def align_hough_spaces_optimization(hough_spaces1, hough_spaces2):
  # using scipy.optimize.least_squares to find the optimal transformation between the two hough spaces

  #hough_spaces1 = post_process_hough_3d(hough_spaces1)
  #hough_spaces2 = post_process_hough_3d(hough_spaces2)

  theta_max = hough_spaces1.shape[0]

  # convert the hough spaces to point clouds
  extrema1 = np.argwhere(hough_spaces1 > 0)
  extrema2 = np.argwhere(hough_spaces2 > 0)



  def fun(params, points1, points2):
    scale_radius, scale_height, translation_angle, translation_height = params

    #scale_radius = scale_radius**2
    #scale_height = scale_height**2
    #translation_angle *= 1000000000
    #translation_height *= 1000000000

    print(f"Scale radius: {scale_radius}, Scale height: {scale_height}, Translation angle: {translation_angle}, Translation height: {translation_height}")
    # apply the transformation to the points
    points1[:, 0] = (points1[:, 0] + translation_angle) % theta_max
    points1[:, 1] = points1[:, 1] * scale_radius
    points1[:, 2] = points1[:, 2] * scale_height + translation_height

    # calculate the distances between the points
    distances, indices = find_corresponding_points(points1, points2, theta_max)

    distances = []
    for i, point in enumerate(points1):
      dist = (point[0] - points2[indices[i]][0]) ** 2 + (point[1] - points2[indices[i]][1]) ** 2 + (point[2] - points2[indices[i]][2]) ** 2

      distances.append(dist)

    print(np.mean(distances))
    return distances
  
  # initial guess
  params = np.array([1, 1, 0, 0])

  bounds = ([0.5, 0.5, -theta_max/2, -500], [2, 2, theta_max/2, 500])

  distances, indices = find_corresponding_points(extrema1, extrema2, theta_max)

  distances = np.array(distances)
  indices = np.array(indices)

  extrema1 = extrema1[distances < 50]
  extrema2 = extrema2[indices[distances < 50]]


  # normalize the points
  #extrema1_ = extrema1
  #extrema1_[:, 0] = extrema1_[:, 0]
  #extrema1_[:, 1] = extrema1_[:, 1]
  #extrema1_[:, 2] = extrema1_[:, 2]
#
  #extrema2_ = extrema2
  #extrema2_[:, 0] = extrema2_[:, 0]
  #extrema2_[:, 1] = extrema2_[:, 1]
  #extrema2_[:, 2] = extrema2_[:, 2]


  def jacobian(params, points1, points2):
    scale_radius, scale_height, translation_angle, translation_height = params

    points1[:, 0] = (points1[:, 0] + translation_angle) % theta_max
    points1[:, 1] = points1[:, 1] * scale_radius
    points1[:, 2] = points1[:, 2] * scale_height + translation_height

    jacobian = np.zeros((len(points1), 4))

    distances, indices = find_corresponding_points(points1, points2, theta_max)

    for i, point in enumerate(points1):
      jacobian[i, 0] = 2 * (point[0] - points2[indices[i]][0]) * points2[indices[i]][0]
      jacobian[i, 1] = 2 * (point[1] - points2[indices[i]][1]) * points2[indices[i]][1]
      jacobian[i, 2] = 2 * (point[2] - points2[indices[i]][2]) * points2[indices[i]][2]
      jacobian[i, 3] = 2 * (point[0] - points2[indices[i]][0]) * points2[indices[i]][2]

    
    return jacobian


  # optimize the transformation
  res = least_squares(fun, params, args=(extrema1, extrema2), verbose=2, jac=jacobian, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=1000, bounds=bounds)

  print(res)

  scale_radius, scale_height, translation_angle, translation_height = res.x

  print(f"Scale radius: {scale_radius}, Scale height: {scale_height}, Translation angle: {translation_angle}, Translation height: {translation_height}")

  points1 = []
  points2 = []

  for theta_base, a, z  in extrema1:
    theta_base = (theta_base + translation_angle) % theta_max
    a = a * scale_radius
    z = z * scale_height + translation_height

    theta = (theta_base / (hough_spaces1.shape[0] - 1)) * 2 * np.pi
    # get the x and y values
    x = a * np.cos(theta)
    y = a * np.sin(theta)
    points1.append([x, y, z])

  for theta_base, a, z  in extrema2:
    theta = (theta_base / (hough_spaces2.shape[0] - 1)) * 2 * np.pi
    # get the x and y values
    x = a * np.cos(theta)
    y = a * np.sin(theta)
    points2.append([x, y, z])

  point_cloud1 = o3d.geometry.PointCloud()
  point_cloud1.points = o3d.utility.Vector3dVector(points1)
  point_cloud1.paint_uniform_color([1, 0, 0])

  point_cloud2 = o3d.geometry.PointCloud()
  point_cloud2.points = o3d.utility.Vector3dVector(points2)
  point_cloud2.paint_uniform_color([0, 0, 1])

  o3d.visualization.draw_geometries([point_cloud1, point_cloud2])

def align_hough_spaces_pytorch(hough_spaces1, hough_spaces2):
  import torch
  
  #hough_spaces1 = post_process_hough_3d(hough_spaces1)
  #hough_spaces2 = post_process_hough_3d(hough_spaces2)

  theta_max = hough_spaces1.shape[0]

  # convert the hough spaces to point clouds
  extrema1 = np.argwhere(hough_spaces1 > 0)
  extrema2 = np.argwhere(hough_spaces2 > 0)
  #extrema2 = extrema1.copy()

  #extrema2[:, 0] = (extrema2[:, 0] + 10) % theta_max
  #extrema2[:, 1] = extrema2[:, 1] * 1.5
  #extrema2[:, 2] = (extrema2[:, 2] + 10) * 1.5


  distances, indices = find_corresponding_points(extrema1, extrema2, theta_max)

  distances = np.array(distances)
  indices = np.array(indices)

  extrema1_ = extrema1[distances < 50].astype(np.float32)
  extrema2_ = extrema2[indices[distances < 50]].astype(np.float32)

  #extrema1 = extrema1.astype(np.float32)
  #extrema2 = extrema2.astype(np.float32)

  # normalize the points
  #extrema1_ = extrema1
  #extrema1_[:, 0] = (extrema1_[:, 0] / theta_max) * 2 * np.pi
  #extrema1_[:, 1] = extrema1_[:, 1] / hough_spaces1.shape[1]
  #extrema1_[:, 2] = extrema1_[:, 2] / hough_spaces1.shape[2]
#
  ##extrema2_ = extrema2
  #extrema2_[:, 0] = (extrema2_[:, 0] / theta_max) * 2 * np.pi
  #extrema2_[:, 1] = extrema2_[:, 1] / hough_spaces1.shape[1]
  #extrema2_[:, 2] = extrema2_[:, 2] / hough_spaces1.shape[2]


  extrema1_tensor = torch.tensor(extrema1_, requires_grad=True)
  extrema2_tensor = torch.tensor(extrema2_, requires_grad=True)


  class Model(torch.nn.Module):
    def __init__(self):
      super(Model, self).__init__()
      self.scale_1 = torch.nn.Parameter(torch.tensor(1.0)) # this parameter will not be optimized by the optimizer, set to 1.0
      self.scale_2 = torch.nn.Parameter(torch.tensor(1.0)) # the true value of this parameter is 2.0
      self.scale_3 = torch.nn.Parameter(torch.tensor(1.0)) # the true value of this parameter is 3.0

      self.translation_1 = torch.nn.Parameter(torch.tensor(0.0)) # the true value of this parameter is 1.5
      self.translation_2 = torch.nn.Parameter(torch.tensor(0.0)) # this parameter will not be optimized by the optimizer, set to 0.0
      self.translation_3 = torch.nn.Parameter(torch.tensor(0.0)) # the true value of this parameter is -1.0

    def forward(self, x):
      # x is a tensor of shape (batch_size, 3)
      # x[:, 0] is the first feature
      # x[:, 1] is the second feature
      # x[:, 2] is the third feature
      # return a tensor of shape (batch_size, 3)
      return torch.stack([
        (self.scale_1 * x[:, 0] + self.translation_1),# % 2 * np.pi,
        self.scale_2 * x[:, 1] + self.translation_2,
        self.scale_3 * x[:, 2] + self.translation_3,
      ], dim=1)
    
  def pairwise_distances(x, y):
    # Calculate the pairwise distances between two sets of points
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    y_square = torch.sum(y ** 2, dim=1, keepdim=True)
    xy = torch.mm(x, y.t())
    dist1 = x_square - 2 * xy + y_square.t()

    return dist1
    x2 = torch.stack([x[:, 0] + theta_max, x[:, 1], x[:, 2]])
    x_square = torch.sum(x2 ** 2, dim=1, keepdim=True)
    xy = torch.mm(x2, y.t())
    dist2 = x_square - 2 * xy + y_square.t()

    x3 = torch.stack([torch.abs(x[:, 0] - theta_max), x[:, 1], x[:, 2]])
    x_square = torch.sum(x3 ** 2, dim=1, keepdim=True)
    xy = torch.mm(x3, y.t())
    dist3 = x_square - 2 * xy + y_square.t()

    dist = torch.min(torch.min(dist1, dist2), dist3)

    # Modify the distance for the first channel (angle)
    #dist[:, 0] = torch.min(dist[:, 0], torch.abs(dist[:, 0] - theta_max))
    
    #return torch.stack([
    #  torch.min(dist[:, 0], torch.abs(dist[:, 0] - theta_max)),
    #  dist[:, 1], 
    #  dist[:, 2]
    #  ], dim=1)

    return dist

  def nearest_neighbor_loss(x, y):
      # convert to cartesian coordinates
      x = torch.stack([x[:, 1] * torch.cos(x[:, 0]), x[:, 1] * torch.sin(x[:, 0]), x[:, 2]])
      y = torch.stack([y[:, 1] * torch.cos(y[:, 0]), y[:, 1] * torch.sin(y[:, 0]), y[:, 2]])


      # Calculate pairwise distances
      dist_matrix = pairwise_distances(x, y)

      # Find the nearest neighbors in y for each point in x
      nearest_dist_x_to_y, _ = torch.min(dist_matrix, dim=1)

      # Find the nearest neighbors in x for each point in y
      nearest_dist_y_to_x, _ = torch.min(dist_matrix, dim=0)

      # check for channel 0 if the distance is smaller if the angle is treated as a circle
      nearest_dist_x_to_y = torch.min(nearest_dist_x_to_y, torch.abs(nearest_dist_x_to_y - theta_max))
      nearest_dist_y_to_x = torch.min(nearest_dist_y_to_x, torch.abs(nearest_dist_y_to_x - theta_max))



      # Compute the mean nearest neighbor distance
      loss = torch.mean(nearest_dist_x_to_y) + torch.mean(nearest_dist_y_to_x)

      return loss
    

  # only include the parameters scale_2, scale_3, translation_1, translation_3 in the optimizer
  model = Model()

  lr_scale = 0.0000000001
  lr_translation = 0.00000001

  optimizer = torch.optim.SGD([
      {'params': [model.scale_2, model.scale_3], 'lr': lr_scale},
      {'params': [model.translation_1, model.translation_3], 'lr': lr_translation}
  ], lr=0.1)


  epochs = 100
  #best_loss = np.inf
  #batch_size = 36

  for epoch in range(epochs):
    #for i in range(0, len(extrema1_tensor), batch_size):
        # get a batch of data
        #x_batch = x[i:i+batch_size]

        points1_batch = extrema1_tensor#[i:i+batch_size]
        points2_batch = extrema2_tensor#[i:i+batch_size]

        points1_modified = model(points1_batch)



        #loss = torch.functional.F.mse_loss(points1_modified, points2_batch)
        loss = nearest_neighbor_loss(points1_modified, points2_batch)


        # compute the output of the model
        #y_batch = model(x_batch)

        # compute the loss
       # loss = torch.mean((y_batch - y[i:i+batch_size])**2)
        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # update the parameters
        optimizer.step()
        if epoch % 5 == 0:
          print(f'epoch {epoch}, loss {loss.item()}')
          print(f"params: {model.scale_1.item()}, {model.scale_2.item()}, {model.scale_3.item()}, {model.translation_1.item()}, {model.translation_2.item()}, {model.translation_3.item()}")

  #print('scale_1', model.scale_1.item())
  #print('scale_2', model.scale_2.item())
  #print('scale_3', model.scale_3.item())
  #print('transp_1', model.translation_1.item() * theta_max)
  #print('transp_2', model.translation_2.item() * hough_spaces1.shape[1])
  #print('transp_3', model.translation_3.item() * hough_spaces1.shape[2])
 

  scale_radius = model.scale_2.item()
  scale_height = model.scale_3.item()
  translation_angle = (model.translation_1.item() / (2 * np.pi)) * theta_max
  translation_height = model.translation_3.item()# * hough_spaces1.shape[2]

  print(f"Scale radius: {scale_radius}, Scale height: {scale_height}, Translation angle: {translation_angle}, Translation height: {translation_height}")
  
  points1 = []
  points2 = []

  for theta_base, a, z  in extrema1:
    theta_base = (theta_base + translation_angle) % theta_max
    a = a * scale_radius
    z = z * scale_height + translation_height

    theta = (theta_base / (hough_spaces1.shape[0] - 1)) * 2 * np.pi
    # get the x and y values
    x = a * np.cos(theta)
    y = a * np.sin(theta)
    points1.append([x, y, z])

  for theta_base, a, z  in extrema2:
    theta = (theta_base / (hough_spaces2.shape[0] - 1)) * 2 * np.pi
    # get the x and y values
    x = a * np.cos(theta)
    y = a * np.sin(theta)
    points2.append([x, y, z])

  point_cloud1 = o3d.geometry.PointCloud()
  point_cloud1.points = o3d.utility.Vector3dVector(points1)
  point_cloud1.paint_uniform_color([1, 0, 0])

  point_cloud2 = o3d.geometry.PointCloud()
  point_cloud2.points = o3d.utility.Vector3dVector(points2)
  point_cloud2.paint_uniform_color([0, 0, 1])

  o3d.visualization.draw_geometries([point_cloud1, point_cloud2])


def single_point_to_hough_spaces():
  theta_max = 144#0 # number of angles
  a_max = 96#0 # number of radii
  z_max = 108#0 # number of heights  

  hough_spaces = np.zeros((theta_max, a_max, z_max))

  points = [] 


  points.append([54, 43])
  points.append([54, 43])
  points.append([54, 43])
  points.append([54, 43])
  #points.append([0, 0])

  for i, point in enumerate(points):
    point_x, point_y = point
    # create an image
    grad_mag = np.zeros((a_max * 2, z_max))
  
    grad_mag[point_x, point_y] = 1
  
    theta_base = 0
    ellipse_ratio = i * 0.5
    threshold = 0
  
    hough_spaces = hough_helper(grad_mag, hough_spaces, theta_base, ellipse_ratio, threshold, a_max, z_max, theta_max, 192, 108)


  # get the extrema
  extrema = np.argwhere(hough_spaces > 0)

  #points = []
  #colors = []
#
  #for theta_base, a, z  in extrema:
  #  theta = (theta_base / (theta_max - 1)) * 2 * np.pi
  #  # get the x and y values
  #  x = a * np.cos(theta)
  #  y = a * np.sin(theta)
  #  points.append([x, y, z])
  #  colors.append([a, 0, 0])
  #
  #colors = np.array(colors)
  #colors = colors / np.max(colors)
#
  ## create a point cloud
  #pcd = o3d.geometry.PointCloud()
  #pcd.points = o3d.utility.Vector3dVector(points)
  #pcd.colors = o3d.utility.Vector3dVector(colors)
#
  #o3d.visualization.draw_geometries([pcd])

  # plot the hough space in 3d
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  for theta_base, a, z  in extrema:
    ax.scatter(a, z, theta_base, c='r', marker='o')

  ax.set_xlabel('a')
  ax.set_ylabel('z')
  ax.set_zlabel('theta_base')

  plt.show()


def hough_coordinate_with_known_angle(ellipse_ratio, phi, theta_base, theta_max, rad_max, x, y):

  x_rel = x - rad_max
   
  alpha = (phi * (2 * np.pi / theta_max) + theta_base - np.pi)
    
  # compute y_im
  y_im = x_rel / np.tan(alpha) if np.tan(alpha) != 0 else 0

  # compute a
  a = np.sqrt(x_rel**2 + y_im**2)

  # compute z
  b = a * ellipse_ratio
  y_rel = (b * np.sqrt(a**2 - x_rel**2)) / a if a != 0 else 0
  z = y - y_rel

  theta = (theta_base + phi) % theta_max 

  return theta, a, z


def calibrate_hough_spaces(hough_spaces, ellipse_ratio, point1, point2, point1_true, point2_true):
  # point1 and point2 are of shape x, y, phi
  # point1_true and point2_true are of shape theta, a, z


  point1_hough_coords = hough_coordinate_with_known_angle(ellipse_ratio, point1[2], point1_true[0], hough_spaces.shape[0], hough_spaces.shape[1], point1[0], point1[1])
  point2_hough_coords = hough_coordinate_with_known_angle(ellipse_ratio, point2[2], point2_true[0], hough_spaces.shape[0], hough_spaces.shape[1], point2[0], point2[1])

  print(f"Point 1 hough coords: {point1_hough_coords}, Point 2 hough coords: {point2_hough_coords}")

  if False:#abs(point1_hough_coords[1] - point2_hough_coords[1]) != 0:
    scale_radius = (abs(point1_true[1] - point2_true[1]) / abs(point1_hough_coords[1] - point2_hough_coords[1]))
  else:
    if point1_hough_coords[1] != 0 and point2_hough_coords[1] != 0:
      scale_radius = ((point1_true[1] / point1_hough_coords[1]) + (point2_true[1] / point2_hough_coords[1])) / 2
      #print(point1_true[1] / point1_hough_coords[1], point2_true[1] / point2_hough_coords[1])
    else:
      scale_radius = 1
      print("Scale radius is 1")

  if abs(point1_hough_coords[2] - point2_hough_coords[2]) != 0:
    scale_height = ((point1_true[2] - point2_true[2]) / (point1_hough_coords[2] - point2_hough_coords[2]))
  else:
    scale_height = 1
  translation_angle = point1_true[0] - point1_hough_coords[0]
  translation_height = ((point1_true[2] - (point1_hough_coords[2] * scale_height)) + (point2_true[2] - (point2_hough_coords[2] * scale_height))) / 2

  print(f"Scale radius: {scale_radius}, Scale height: {scale_height}, Translation angle: {translation_angle}, Translation height: {translation_height}")

  return scale_radius, scale_height, translation_angle, translation_height

def combine_hough_spaces(hough_spaces1, hough_spaces2, transformation1, transformation2):
  #hough_spaces1 = (hough_spaces1 / np.max(hough_spaces1)).astype(np.float32)
  #hough_spaces2 = (hough_spaces2 / np.max(hough_spaces1)).astype(np.float32)

  scale_radius1, scale_height1, translation_angle1, translation_height1 = transformation1
  scale_radius2, scale_height2, translation_angle2, translation_height2 = transformation2

  translation_angle1 = int(translation_angle1)
  translation_angle2 = int(translation_angle2)

  # merge hough_spaces2 into hough_spaces1
  scale_radius = scale_radius1 / scale_radius2
  scale_height = scale_height1 / scale_height2
  translation_angle = translation_angle1 - translation_angle2
  translation_height = (translation_height2 - translation_height1) / scale_height2
  print(f"Scale radius: {scale_radius}, Scale height: {scale_height}, Translation angle: {translation_angle}, Translation height: {translation_height}")

  theta_max = hough_spaces1.shape[0]
  a_max = min(hough_spaces1.shape[1], hough_spaces2.shape[1])
  z_max = min(hough_spaces1.shape[2], hough_spaces2.shape[2])

  hough_spaces = np.zeros((theta_max, a_max, z_max), dtype=np.uint32)

  with tqdm(total=theta_max) as pbar:
    for i in range(theta_max):
      layer1 = hough_spaces1[i]
      hough_spaces[i] = layer1[:min(a_max, layer1.shape[0]), :min(z_max, layer1.shape[1])]
      layer2 = hough_spaces2[(i + translation_angle) % theta_max]
      for j in range(min(a_max, layer2.shape[0])):
        for k in range(min(z_max, layer2.shape[1])):
          x = int(j * scale_radius)
          y = int(k * scale_height - translation_height)
          if x >= 0 and x < layer1.shape[0] and y >= 0 and y < layer1.shape[1]:
            hough_spaces[i, j, k] += layer2[x, y]

      pbar.update(1)

  return hough_spaces
    
  
  


def main():

  if True:
    hough_spaces1 = np.load("hough_spaces_voronoi_60d.npy")#np.load("post_processed_voronoi_60d.npy")
    hough_spaces2 = np.load("hough_spaces_voronoi_90d.npy")#np.load("post_processed_voronoi_90d.npy")
    hough_spaces1 = hough_spaces1[::4]
    hough_spaces2 = hough_spaces2[::4]
    
    print(hough_spaces1.shape)
    print(hough_spaces2.shape)

    transformation1 = calibrate_hough_spaces(hough_spaces1, 0.5, [713, 971, 45], [713, 1008, 45], [0, 0.4, -0.405], [0, 0.4, -0.455]) # outer: [713, 971, 45], inner: [744, 953, 45]
    transformation2 = calibrate_hough_spaces(hough_spaces2, 0, [710, 898, 45], [710, 942, 45], [0, 0.4, -0.405], [0, 0.4, -0.455]) # outer: [710, 898, 45], inner: [741, 897, 45]

    hough_spaces = combine_hough_spaces(hough_spaces1, hough_spaces2, transformation1, transformation2)

    hough_spaces = post_process_hough_3d(hough_spaces)

    extremas = np.argwhere(hough_spaces > 0)
    scale_radius, scale_height, translation_angle, translation_height = transformation1

    points = []

    for theta_base, a, z  in extremas:

      # apply transformation1
      theta_base = (theta_base + translation_angle) % hough_spaces.shape[0]
      a = a * scale_radius
      z = z * scale_height + translation_height

      theta = (theta_base / (hough_spaces.shape[0] - 1)) * 2 * np.pi
      # get the x and y values
      x = a * np.cos(theta)
      y = a * np.sin(theta)
      points.append([x, y, z])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([point_cloud])

    return


    #hough_spaces1 = hough_spaces1 / np.max(hough_spaces1)
    #hough_spaces2 = hough_spaces2 / np.max(hough_spaces2)

    extrema1 = np.argwhere(hough_spaces1 > 0)
    extrema2 = np.argwhere(hough_spaces2 > 0)

    points1 = []
    points2 = []

    colors1 = []
    colors2 = []

    for theta_base, a, z  in extrema1:
      color = hough_spaces1[theta_base, a, z]
      theta_base = (theta_base + translation_angle1) % hough_spaces1.shape[0]
      a = a * scale_radius1
      z = z * scale_height1 + translation_height1

      theta = (theta_base / (hough_spaces1.shape[0] - 1)) * 2 * np.pi
      # get the x and y values
      x = a * np.cos(theta)
      y = a * np.sin(theta)
      points1.append([x, y, z])
      colors1.append(color)

    for theta_base, a, z  in extrema2:
      color = hough_spaces2[theta_base, a, z]
      theta_base = (theta_base + translation_angle2) % hough_spaces2.shape[0]
      a = a * scale_radius2
      z = z * scale_height2 + translation_height2

      theta = (theta_base / (hough_spaces2.shape[0] - 1)) * 2 * np.pi
      # get the x and y values
      x = a * np.cos(theta)
      y = a * np.sin(theta)
      points2.append([x, y, z])
      colors2.append(color)

    colors1 = np.array(colors1)
    colors1 = colors1 / np.max(colors1)

    colors2 = np.array(colors2)
    colors2 = colors2 / np.max(colors2)

    point_cloud1 = o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(points1)
    #point_cloud1.colors = o3d.utility.Vector3dVector(colors1)
    point_cloud1.paint_uniform_color([1, 0, 0])

    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(points2)
    point_cloud2.paint_uniform_color([0, 0, 1])
    
    o3d.visualization.draw_geometries([point_cloud1, point_cloud2])

    # save the point clouds

    filename ="test_cloud_scaled.ply"

    # write the points to a ply file
    #with open(filename, "w") as file:
    #  file.write("ply\n")
    #  file.write("format ascii 1.0\n")
    #  file.write("element vertex " + str(len(points1)) + "\n")
    #  file.write("property float32 x\n")
    #  file.write("property float32 y\n")
    #  file.write("property float32 z\n")
    #  file.write("property uint8 red\n")
    #  file.write("property uint8 green\n")
    #  file.write("property uint8 blue\n")
    #  file.write("end_header\n")
    #  for i, point in enumerate(points1):
    #    file.write(str(float(point[0])) + " " + str(float(point[1])) + " " + str(float(point[2])) + " 255 0 0\n")

    # write the points to a ply file
    with open(filename, "w") as file:
      file.write("ply\n")
      file.write("format ascii 1.0\n")
      file.write("element vertex " + str(len(points2)) + "\n")
      file.write("property float32 x\n")
      file.write("property float32 y\n")
      file.write("property float32 z\n")
      file.write("property uint8 red\n")
      file.write("property uint8 green\n")
      file.write("property uint8 blue\n")
      file.write("end_header\n")
      for i, point in enumerate(points2):
        file.write(str(float(point[0])) + " " + str(float(point[1])) + " " + str(float(point[2])) + " 0 0 255\n")
    


    


  if False: # ICP open3d ---------------------------------------------------------------------------------------------------
    # load the point clouds
    source_hough = np.load("post_processed_voronoi_60d.npy")
    target_hough = np.load("post_processed_voronoi_90d.npy")

    # convert the hough spaces to point clouds
    extrema1 = np.argwhere(source_hough > 0)
    extrema2 = np.argwhere(target_hough > 0)

    points1 = []
    points2 = []

    for theta_base, a, z  in extrema1:
          theta = (theta_base / (source_hough.shape[0] - 1)) * 2 * np.pi
          # get the x and y values
          x = a * np.cos(theta)
          y = a * np.sin(theta)
          points1.append([x, y, z])

    for theta_base, a, z  in extrema2:
          theta = (theta_base / (target_hough.shape[0] - 1)) * 2 * np.pi
          # get the x and y values
          x = a * np.cos(theta)
          y = a * np.sin(theta)
          points2.append([x, y, z])

    point_cloud1 = o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(points1)
    point_cloud1.paint_uniform_color([1, 0, 0])

    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(points2)
    point_cloud2.paint_uniform_color([0, 0, 1])

    #o3d.visualization.draw_geometries([point_cloud1, point_cloud2])

    trans_init = np.asarray([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
    print("Initial alignment")
    draw_registration_result(point_cloud1, point_cloud2, trans_init)
    print("Apply point-to-point ICP")

    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    #estimation_method.with_scaling = True

    reg_p2p = o3d.pipelines.registration.registration_icp(
        point_cloud1, point_cloud2, 50, estimation_method=estimation_method
    )

    print(reg_p2p)

    draw_registration_result(point_cloud1, point_cloud2, reg_p2p.transformation)


  if False: # selfmade ICP --------------------------------------------------------------------------------------------
    # load the point clouds
    source_hough = np.load("post_processed_voronoi_60d.npy")
    target_hough = np.load("post_processed_voronoi_90d.npy")

    # convert the hough spaces to point clouds
    extrema1 = np.argwhere(source_hough > 0)
    extrema2 = np.argwhere(target_hough > 0)

    points1 = []
    points2 = []

    for theta_base, a, z  in extrema1:
          theta = (theta_base / (source_hough.shape[0] - 1)) * 2 * np.pi
          # get the x and y values
          x = a * np.cos(theta)
          y = a * np.sin(theta)
          points1.append([x, y, z])

    for theta_base, a, z  in extrema2:
          theta = (theta_base / (target_hough.shape[0] - 1)) * 2 * np.pi
          # get the x and y values
          x = a * np.cos(theta)
          y = a * np.sin(theta)
          points2.append([x, y, z])

    point_cloud1 = o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(points1)
    point_cloud1.paint_uniform_color([1, 0, 0])

    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(points2)
    point_cloud2.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([point_cloud1, point_cloud2])

    extrema1, extrema2 = estimate_transformation(source_hough, target_hough, iterations=50)

    points1 = []
    points2 = []

    for theta_base, a, z  in extrema1:
          theta = (theta_base / (source_hough.shape[0] - 1)) * 2 * np.pi
          # get the x and y values
          x = a * np.cos(theta)
          y = a * np.sin(theta)
          points1.append([x, y, z])

    for theta_base, a, z  in extrema2:
          theta = (theta_base / (target_hough.shape[0] - 1)) * 2 * np.pi
          # get the x and y values
          x = a * np.cos(theta)
          y = a * np.sin(theta)
          points2.append([x, y, z])

    point_cloud1 = o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(points1)
    point_cloud1.paint_uniform_color([1, 0, 0])

    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(points2)
    point_cloud2.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([point_cloud1, point_cloud2])


  if False: # manual alignment --------------------------------------------------------------------------------------------
    # load the hough spaces from a file
    hough_spaces1 = np.load("hough_spaces_voronoi_60d.npy")
    hough_spaces1 = hough_spaces1[::8]
    hough_spaces2 = np.load("hough_spaces_voronoi_90d.npy")
    hough_spaces2 = hough_spaces2[::8]

    align_hough_spaces_manually(hough_spaces1, hough_spaces2)











  #test_img = cv2.imread("test/cylinder_test.py", 0)
  #plt.imshow(test_img, cmap="gray")
  #plt.show()
#
  #test_hough()
  #return

  #dir = os.path.join("..", "scratch", "diagonal_angle", "cropped")
  #dir = os.path.join("..", "scratch", "60d_100mm_10mm_voronoi", "rows")
  #dir = os.path.join("..", "scratch", "circle_moving")
  #dir = os.path.join("..", "scratch", "75d_100mm_10mm_honey")
  #dir = os.path.join("..", "scratch", "full_1440")
  #dir = os.path.join("..", "scratch", "dual_angle", "voronoi_60d")
  #dir = os.path.join("..", "scratch", "test")
  #full_hough_to_ply(dir)

if __name__ == "__main__":
  main()