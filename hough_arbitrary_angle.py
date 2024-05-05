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

def full_hough_to_ply(dir):
  ellipse_ratio = 0.5
  threshold = 20

  frame_names = os.listdir(dir)
  # sort the frames named 0.png, 1.png, ..., 100.png, ...
  frame_names = sorted(frame_names, key=lambda x: int(x.split('.')[0]))
  frame_names = frame_names[::10]
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
  hough_space = (hough_space - np.min(hough_space)) / (np.max(hough_space) - np.min(hough_space))

  hough_space = hough_space * 255

  #return hough_space
  
  # gradient in y direction
  gradient_hough = np.abs(cv2.Sobel(hough_space, cv2.CV_64F, 0, 1, ksize=5))

  # normalize the gradient
  if gradient_hough.max() > 0: 
    gradient_hough = (gradient_hough - np.min(gradient_hough)) / (np.max(gradient_hough) - np.min(gradient_hough))

  gradient_hough = gradient_hough * 255

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
  

def test_hough():
  # load the hough spaces from a file
  hough_spaces = np.load("hough_spaces.npy")

  # post process the hough spaces
  with tqdm(total=hough_spaces.shape[0], desc="Post processing hough spaces") as pbar:
    for i, hough_space in enumerate(hough_spaces):  
      hough_spaces[i] = post_process_hough(hough_space)
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

  o3d.visualization.draw_geometries([pcd])

  """
  colors *= 255
  colors = colors.astype(np.uint8)

  # write the points to a ply file
  with open("test_arbitrary_angle.ply", "w") as file:
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

  #test_hough()
  #return

  #dir = os.path.join("..", "scratch", "diagonal_angle", "cropped")
  dir = os.path.join("..", "scratch", "60d_100mm_10mm_voronoi")
  #dir = os.path.join("..", "scratch", "circle_moving")
  full_hough_to_ply(dir)

if __name__ == "__main__":
  main()