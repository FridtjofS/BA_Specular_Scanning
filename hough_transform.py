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

def calculate_hough_spaces(image):
  edges = cv2.Canny(image, 2, 5)

  gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
  gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

  return calculate_hough_spaces_helper(image, edges, gradient_x, gradient_y)


@jit(nopython=True)
def calculate_hough_spaces_helper(image, edges, gradient_x, gradient_y):
  num_rotations, width = image.shape
  x_center = width // 2

  # initialize hough spaces
  hough_space_h1 = np.zeros((x_center, num_rotations))
  hough_space_h2 = np.zeros((x_center, num_rotations))


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
            hough_space_h1[amplitude, phi] += 1
          else:
            # calculate the corresponding hough space phi
            phi = np.arccos((x - x_center) / amplitude) - ((theta / num_rotations) * 2 * np.pi) + (np.pi / 2)
            phi = phi % (2 * np.pi)
            phi = int(phi / (2 * np.pi) * (num_rotations - 1))
            # add the amplitude to the corresponding hough space
            hough_space_h2[amplitude, phi] += 1


  return hough_space_h1, hough_space_h2

def perspective_hough_transform(images, f, dist):
  # perform canny on all images
  edges = [cv2.Canny(image, 2, 5) for image in images]

  # perform sobel on all images
  gradients_x = [cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) for image in images]
  gradients_y = [cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5) for image in images]

  # calculate the hough spaces for all images
  hough_spaces = perspective_hough_transform_helper(images, edges, gradients_x, gradients_y, f, dist)

@jit(nopython=True)
def perspective_hough_transform_helper(images, edges, gradients_x, gradients_y, f, dist):
    """
    This function calculates the hough space for a perspective transformation
    f: focal length
    dist: distance from the camera to the center of rotation
    """
    #edges = cv2.Canny(image, 2, 5)

    #gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    #gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    num_rotations, width = images[0].shape
    height = len(images)
    x_center = width // 2
    hough_spaces_h1 = [np.zeros((x_center, num_rotations)) for _ in range(height)]
    hough_spaces_h2 = [np.zeros((x_center, num_rotations)) for _ in range(height)]

    for y in range(height):
      for x in range(1, width - 1):
        for theta in range(1, num_rotations):
          for amplitude in range(1, x_center):
            temp = 1 + (amplitude / dist) * np.cos(theta)
            if temp == 0:
              continue
            proj_x = (-f * (amplitude/dist) * np.sin(theta)) / temp
            proj_x = int(proj_x)
            proj_y = (f * (y/dist)) / temp
            proj_y = int(proj_y)
            if proj_x < 0 or proj_x >= width or proj_y < 0 or proj_y >= height:
              continue
            if edges[y][proj_x, proj_y] > 0:
              hough_spaces_h1[y][amplitude, theta] += 1

            
    return hough_spaces_h1, hough_spaces_h2
   
    #for x in range(1, width - 1):
    #  for theta in range(1, num_rotations):
    #    if edges[theta, x] > 0:
    #      for amplitude in range(1, x_center):
    #        if amplitude == 0:
    #          continue
    #        if gradient_x[theta, x] * gradient_y[theta, x] < 0:
    #          phi = np.atan((f * amplitude - np.sqrt(f**2 * amplitude**2 + x**2 * (amplitude**2 - dist**2))) / (x * (amplitude - dist)))
    #          phi = phi - ((theta/num_rotations) * 2 * np.pi)
    #          phi = (phi + 2 * np.pi) * num_rotations / (2.5 * np.pi)
    #          phi = int(phi)
    #          hough_space_h1[amplitude, phi] += 1



            #else:
            #  phi = np.arccos((x - x_center) / amplitude) - ((theta / num_rotations) * 2 * np.pi) + (np.pi / 2)
            #  phi = phi % (2 * np.pi)
            #  phi = int(phi / (2 * np.pi) * (num_rotations - 1))
            #  hough_space_h2[amplitude, phi] += 1

    return

def post_process_hough_space(hough_space_h1, hough_space_h2):
    hough_space_h1 = hough_space_h1 + hough_space_h2
    hough_space_h2 = hough_space_h1

    sum_h1 = np.sum(hough_space_h1)
    sum_h2 = np.sum(hough_space_h2)
    size = hough_space_h1.shape[0] * hough_space_h1.shape[1]
    temp1 = size / sum_h1 if sum_h1 > 0 else 1
    if temp1 == np.inf:
      temp1 = 0.9
    temp2 = size / sum_h2 if sum_h2 > 0 else 1
    if temp2 == np.inf:
      temp2 = 0.9
    r1 = int(temp1 * min(hough_space_h1.shape[0], hough_space_h1.shape[1]))
    r2 = int(temp2 * min(hough_space_h2.shape[0], hough_space_h2.shape[1]))

    # remove low frequencies by subtracting the lowpass filtered image
    fspace_h1 = np.fft.fft2(hough_space_h1)
    fspace_h2 = np.fft.fft2(hough_space_h2)
    fspace_h1 = np.fft.fftshift(fspace_h1)
    fspace_h2 = np.fft.fftshift(fspace_h2)
    rows, cols = hough_space_h1.shape
    crow, ccol = rows // 2, cols // 2
    # create a mask first, center circle is 1, remaining all zeros
    mask1 = np.zeros((rows, cols), np.uint8)
    mask2 = np.zeros((rows, cols), np.uint8)
    #r1 = 50
    #r2 = 100
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area_1 = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r1**2
    mask_area_2 = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r2**2
    mask1[mask_area_1] = 1
    mask2[mask_area_2] = 1
    fshift_h1 = fspace_h1 * mask1
    fshift_h2 = fspace_h2 * mask2

    f_ishift_h1 = np.fft.ifftshift(fshift_h1)
    f_ishift_h2 = np.fft.ifftshift(fshift_h2)
    img_back_h1 = np.fft.ifft2(f_ishift_h1)
    img_back_h2 = np.fft.ifft2(f_ishift_h2)
    img_back_h1 = np.abs(img_back_h1)
    img_back_h2 = np.abs(img_back_h2)

    hough_space_h1 = hough_space_h1 - img_back_h1
    hough_space_h2 = hough_space_h2 - img_back_h2

    # clip all values below 0 to 0
    hough_space_h1[hough_space_h1 < 0] = 0
    hough_space_h2[hough_space_h2 < 0] = 0
    

    # Weigh the matrix with weights e^(-0.001)*[1,2,...,A_max]
    for amplitude in range(1, hough_space_h1.shape[0]):
        hough_space_h1[amplitude, :] = hough_space_h1[amplitude, :] * np.exp(-0.001 * amplitude)
        hough_space_h2[amplitude, :] = hough_space_h2[amplitude, :] * np.exp(-0.001 * amplitude)

    # clip all values below 0 to 0
    hough_space_h1[hough_space_h1 < 0] = 0
    hough_space_h2[hough_space_h2 < 0] = 0

    # normalize the matrix (can be negative after weighing)
    if hough_space_h1.max() > 0:
        hough_space_h1 = (hough_space_h1 - np.min(hough_space_h1)) / (np.max(hough_space_h1) - np.min(hough_space_h1))
    else:
        hough_space_h1 = hough_space_h1 * 0
    if hough_space_h2.max() > 0:
        hough_space_h2 = (hough_space_h2 - np.min(hough_space_h2)) / (np.max(hough_space_h2) - np.min(hough_space_h2))
    else:
        hough_space_h2 = hough_space_h2 * 0
    

    hough_space_h1 = hough_space_h1 * 255
    hough_space_h2 = hough_space_h2 * 255

    

    # gradient in y direction
    gradient_h1 = np.abs(cv2.Sobel(hough_space_h1, cv2.CV_64F, 0, 1, ksize=5))
    #gradient_h1 = hough_space_h1 * hough_space_h1
    gradient_h2 = np.abs(cv2.Sobel(hough_space_h2, cv2.CV_64F, 0, 1, ksize=5))
    #gradient_h2 = hough_space_h2 * hough_space_h2

    # normalize the gradient
    if gradient_h1.max() > 0: 
      gradient_h1 = (gradient_h1 - np.min(gradient_h1)) / (np.max(gradient_h1) - np.min(gradient_h1))
    if gradient_h2.max() > 0:
      gradient_h2 = (gradient_h2 - np.min(gradient_h2)) / (np.max(gradient_h2) - np.min(gradient_h2))

    gradient_h1 = gradient_h1 * 255
    gradient_h2 = gradient_h2 * 255

    #gradient_h1[gradient_h1 < 10] = 0
    #gradient_h2[gradient_h2 < 10] = 0

    # gaussian blur
    #hough_space_h1 = cv2.GaussianBlur(hough_space_h1.astype(np.uint8), (9, 9), 0)
    #hough_space_h2 = cv2.GaussianBlur(hough_space_h2.astype(np.uint8), (9, 9), 0)

    
    
    

    # min filter to remove small horizontal noise
    #hough_space_h1 = cv2.erode(hough_space_h1.astype(np.uint8), np.ones((1, 5), np.uint8), iterations=1)
    #hough_space_h2 = cv2.erode(hough_space_h2.astype(np.uint8), np.ones((1, 5), np.uint8), iterations=1)

    # median filter
    #hough_space_h1 = cv2.medianBlur(hough_space_h1.astype(np.uint8), 3)
    #hough_space_h2 = cv2.medianBlur(hough_space_h2.astype(np.uint8), 3)


    

    t1, _ = cv2.threshold(gradient_h1.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    t2, _ = cv2.threshold(gradient_h2.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    
    hough_space_h1[gradient_h1 < t1] = 0
    hough_space_h2[gradient_h2 < t2] = 0	

    #hough_space_h1[ccordinatesh1[:, 0], ccordinatesh1[:, 1]] = 255
    #hough_space_h2[ccordinatesh2[:, 0], ccordinatesh2[:, 1]] = 255
    

    # repress values below a certain threshold using Otsu's thresholding
    #t1, _ = cv2.threshold(hough_space_h1.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    #t2, _ = cv2.threshold(hough_space_h2.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
#
    #hough_space_h1[hough_space_h1 < t1] = 0
    #hough_space_h2[hough_space_h2 < t2] = 0	


    # maybeee???
    #hough_space_h1 = cv2.medianBlur(hough_space_h1.astype(np.uint8), 3)
    #hough_space_h2 = cv2.medianBlur(hough_space_h2.astype(np.uint8), 3)

    # blur to get a heatmap
    #hough_space_h1 = cv2.GaussianBlur(hough_space_h1.astype(np.uint8), (19, 19), 0)
    #hough_space_h2 = cv2.GaussianBlur(hough_space_h2.astype(np.uint8), (19, 19), 0)

    #hough_space_h1 = cv2.erode(hough_space_h1.astype(np.uint8), np.ones((1, 7), np.uint8), iterations=1)
    #hough_space_h2 = cv2.erode(hough_space_h2.astype(np.uint8), np.ones((1, 7), np.uint8), iterations=1)

    coordinates_h1 = sk.feature.peak_local_max(hough_space_h1, threshold_rel = 0.2, min_distance=5)
    coordinates_h2 = sk.feature.peak_local_max(hough_space_h2, threshold_rel = 0.2, min_distance=5)

    #hough_space_h1 = cv2.polylines(hough_space_h1, [coordinates_h1], False, 255, 1)
    #hough_space_h2 = cv2.polylines(hough_space_h2, [coordinates_h2], False, 255, 1)

    #coordinates_h1 = sk.feature.peak_local_max(gradient_h1, min_distance=10)
    #coordinates_h2 = sk.feature.peak_local_max(gradient_h2, min_distance=10)
#
    h1_temp = np.zeros(hough_space_h1.shape)
    h2_temp = np.zeros(hough_space_h2.shape)

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


    h1_temp[coordinates_h1[:, 0], coordinates_h1[:, 1]] = 1
    h2_temp[coordinates_h2[:, 0], coordinates_h2[:, 1]] = 1
    ## set those coordinates which dont! have a local maximum to 0 by multiplying with the temp array
    hough_space_h1 = hough_space_h1 * h1_temp
    hough_space_h2 = hough_space_h2 * h2_temp
    return hough_space_h1, hough_space_h2



def plot_curves_from_hough_spaces(hough_space_h1, hough_space_h2, image):
  edges = cv2.Canny(image, 2, 5)

  gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
  gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

  num_rotations, width = image.shape
  x_center = width // 2

  indices_h1 = np.argwhere(hough_space_h1 > 50)
  indices_h2 = np.argwhere(hough_space_h2 > 80)

  # sort the indices by their amplitude
  indices_h1 = sorted(indices_h1, key=lambda x: x[1], reverse=True)
  indices_h2 = sorted(indices_h2, key=lambda x: x[1], reverse=False)

  output_h1 = np.zeros((num_rotations, width))
  output_h2 = np.zeros((num_rotations, width))

  for i in range(len(indices_h1)):
    amplitude, phase = indices_h1[i]
    phase = (phase / num_rotations) * 2 * np.pi

    for y in range(1, num_rotations - 1):

      theta = (y / num_rotations) * 2 * np.pi
      occlusion = (theta + phase) % (2 * np.pi)
      if occlusion > np.pi / 2 and occlusion < 3 * np.pi / 2:
        continue
      x = int(x_center + amplitude * np.sin(theta + phase))
      if True:#gradient_x[y, x] * gradient_y[y, x] < 0: #and edges[y, x] > 0:
        if output_h1[y, x] == 0:
          mode = 3
          if mode == 0:
            output_h1[y, x] = 255
          elif mode == 1:
            # phase
            output_h1[y, x] = int(phase / (2 * np.pi) * 255)
          elif mode == 2:
            # amplitude
            output_h1[y, x] = int(amplitude / width * 255)
          elif mode == 3:
            # depth
            # depending on the amplitude and offset the depth is calculated
            # the depth is then normalized to [0, 255]
            offset = theta + phase
            depth = np.abs(np.cos(offset) * amplitude)
            depth = int((depth / x_center) * 255)
            output_h1[y, x] = depth
            

  for i in range(len(indices_h2)):
    amplitude, phase = indices_h2[i]
    phase = (phase / num_rotations) * 2 * np.pi

    for y in range(1, num_rotations - 1):
      theta = (y / num_rotations) * 2 * np.pi
      occlusion = (theta + phase) % (2 * np.pi)
      if occlusion <= np.pi / 2 or occlusion >= 3 * np.pi / 2:
        continue
      x = int(x_center + amplitude * np.sin(theta + phase))
      if True:#gradient_x[y, x] * gradient_y[y, x] >= 0:# and edges[y, x] > 0:
        if output_h2[y, x] == 0:
          mode = 3
          if mode == 0:
            output_h2[y, x] = 255
          elif mode == 1:
            # phase
            output_h2[y, x] = int(phase / (2 * np.pi) * 255)
          elif mode == 2:
            # amplitude
            output_h2[y, x] = int(amplitude / width * 255)
          elif mode == 3:
            # depth
            # depending on the amplitude and offset the depth is calculated
            # the depth is then normalized to [0, 255]
            offset = theta + phase
            depth = np.abs(np.cos(offset) * amplitude)
            depth = int((depth / x_center) * 255)
            output_h2[y, x] = 255 - depth

  plt.imshow(output_h1, cmap='jet', interpolation='none')
  plt.colorbar()
  plt.show()

  plt.imshow(output_h2, cmap='jet', interpolation='none')
  plt.colorbar()
  plt.show()

  # image to grayscale in one channel
  #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  output = np.zeros((num_rotations, width, 3), dtype=np.uint8)
  output[:, :, 0] = edges
  output[:, :, 1] = output_h1
  #output[:, :, 1] = output_h2
  

  plt.imshow(output)
  plt.show()

      

    
     




def get_coordinates_from_input(hough_space_h1):
   # returns a numpy array with the clicked coordinates	in the shape of hough_space_h1
    # on the clicked coordinates the new array is set to 1, else 0
    # the array is then returned
   
    def onclick(event):
      global ix, iy
      ix, iy = event.xdata, event.ydata
      print ('x = %d, y = %d'%(
          ix, iy))
      
      coords.append((int(ix), int(iy)))
      #if len(coords) == 10:
      #  fig.canvas.mpl_disconnect(cid)
      #return coords
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(hough_space_h1, cmap='jet')
    global coords
    coords = []
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title('Please click the points to plot the sine waves')
    plt.show()

    new = np.zeros(hough_space_h1.shape)
    for i in range(len(coords)):
      new[coords[i][1], coords[i][0]] = 1

    return new




def plot_hough_transform1point(image, x, theta):
    width, num_rotations = image.shape  # Anzahl der Rotationen

    # Schritt 1: Kanten im Bild erkennen
    edges = cv2.Canny(image, 30, 100)

    #edges = cv2.imread('cat_test_canny.png',0)
    # invert the image
    #edges = cv2.bitwise_not(edges)

    # plot edges
    plt.imshow(edges, cmap='inferno')
    plt.colorbar()
    plt.show()

    x_center = width // 2

    hough_space_h1 = np.zeros((width // 2, num_rotations))
    hough_space_h2 = np.zeros((width // 2, num_rotations))


    for amplitude in range(1, width // 2, 50):
        if amplitude == 0:
            continue
        elif amplitude < np.abs(x - x_center):
            continue
        try:
            phi = 0
            if (int(image[x, theta]) - int(image[x - 1, theta])) > 0:
                print("sdfh")
                phi = np.arcsin(((x - x_center) / amplitude))# - ((theta/num_rotations) * 2 * np.pi)

                # plot sine wave with amplitude and phase
                t = np.arange(0, 2 * np.pi, 0.01)
                y = amplitude * np.sin(t + phi)
                plt.plot(t, y)
    
                # find local maxima and minima
                maxima, _ = find_peaks(y)
                minima, _ = find_peaks(-y)
    
                # highlight local maxima and minima
                plt.plot(t[maxima], y[maxima], 'ro')
                plt.plot(t[minima], y[minima], 'bo')

                phi = phi - ((theta/num_rotations) * 2 * np.pi)
                # stretch phi from [-2pi, pi/2] to [0, num_rotations]
                phi = (phi + 2 * np.pi) * num_rotations / (2.5 * np.pi)
                phi = int(phi)

                hough_space_h1[amplitude, phi] += 1

                for i in range (-5, 5):
                   for j in range(-5, 5):
                      if amplitude + i < 0 or amplitude + i >= width // 2 or phi + j < 0 or phi + j >= num_rotations:
                          continue
                      hough_space_h1[amplitude + i, phi + j] += 1

            else:
                phi = np.arccos((x - x_center) / amplitude)# - ((theta/num_rotations) * 2 * np.pi) + (np.pi / 2)

                # plot sine wave with amplitude and phase
                t = np.arange(0, 2 * np.pi, 0.01)
                y = amplitude * np.sin(t + phi)
                plt.plot(t, y)
    
                # find local maxima and minima
                maxima, _ = find_peaks(y)
                minima, _ = find_peaks(-y)
    
                # highlight local maxima and minima
                plt.plot(t[maxima], y[maxima], 'ro')
                plt.plot(t[minima], y[minima], 'bo')

                phi = phi - ((theta/num_rotations) * 2 * np.pi) + (np.pi / 2)

                # stretch phi from [-3pi/2, pi] to [0, num_rotations]
                phi = (phi + 3 * np.pi / 2) * num_rotations / (2.5 * np.pi)
                phi = int(phi)

                hough_space_h2[amplitude, phi] += 1

                for i in range (-5, 5):
                   for j in range(-5, 5):
                      if amplitude + i < 0 or amplitude + i >= width // 2 or phi + j < 0 or phi + j >= num_rotations:
                          continue
                      hough_space_h2[amplitude + i, phi + j] += 1
                         

        except:
            continue
    plt.show()

    # plot Hough-Räume
    plt.imshow(hough_space_h1, cmap='jet')
    plt.colorbar()
    plt.show()

    plt.imshow(hough_space_h2, cmap='jet')
    plt.colorbar()
    plt.show()

@jit(nopython=True)
def compute_3d_points(hough_space_h1, hough_space_h2, indices_h1, indices_h2, image):
  init_width, num_rotations = image.shape  # Anzahl der Rotationen
  width = init_width * 2
  
  # initialize the 3d points as array of width x width (top view)
  points = np.zeros((width, width))

  # iterate over all indices
  for i in range(len(indices_h1[0])):
    # Extract the amplitude and phase for h1
    amplitude_h1, phase_h1 = indices_h1[0][i], indices_h1[1][i]
    # polar coordinates to cartesian coordinates
    x = int(amplitude_h1 * np.cos((phase_h1 / init_width) * 2 * np.pi)) - init_width
    y = int(amplitude_h1 * np.sin((phase_h1 / init_width) * 2 * np.pi)) - init_width


    # set the point in the 3d points array
    points[x, y] = hough_space_h1[amplitude_h1, phase_h1]

  #
  for i in range(len(indices_h2[0])):
    # Extract the amplitude and phase for h2
    amplitude_h2, phase_h2 = indices_h2[0][i], indices_h2[1][i]

    # polar coordinates to cartesian coordinates
    x = int(amplitude_h2 * np.cos((phase_h2 / init_width) * 2 * np.pi)) - width // 2
    y = int(amplitude_h2 * np.sin((phase_h2 / init_width) * 2 * np.pi)) - width // 2

    # set the point in the 3d points array
    points[x, y] = hough_space_h2[amplitude_h2, phase_h2]

  return points

def compute_3d_coordinates(hough_space_h1, hough_space_h2, indices_h1, indices_h2, image, z):
  init_width, num_rotations = image.shape  # Anzahl der Rotationen
  width = init_width * 2
  
  points = []
  for i in range(len(indices_h1[0])):
    amplitude_h1, phase_h1 = indices_h1[0][i], indices_h1[1][i]
    x = amplitude_h1 * np.cos((phase_h1 / init_width) * 2 * np.pi)
    y = amplitude_h1 * np.sin((phase_h1 / init_width) * 2 * np.pi)
    points.append((x, y, z, hough_space_h1[amplitude_h1, phase_h1]))

  for i in range(len(indices_h2[0])):
    amplitude_h2, phase_h2 = indices_h2[0][i], indices_h2[1][i]
    x = amplitude_h2 * np.cos((phase_h2 / init_width) * 2 * np.pi)
    y = amplitude_h2 * np.sin((phase_h2 / init_width) * 2 * np.pi)
    points.append((x, y, z, hough_space_h2[amplitude_h2, phase_h2]))

  return np.array(points)

def full_hough_transform_3d(dir):
  # iterate over all images in the directory
  
  frames = [cv2.imread(os.path.join(dir, frame), 0) for frame in os.listdir(dir)][::20]
  frames = frames[::-1]
  
  width = frames[0].shape[1]
  points_3d = np.zeros((len(frames), width, width))
  

  for i, frame in tqdm(enumerate(frames)):

    hough_space_h1, hough_space_h2 = calculate_hough_spaces(frame)
    hough_space_h1, hough_space_h2 = post_process_hough_space(hough_space_h1, hough_space_h2)

    #plt.imshow(hough_space_h1, cmap='jet')
    #plt.colorbar()
    #plt.show()

    #indices_h1 = np.array(np.unravel_index(np.argsort(hough_space_h1.ravel())[-500:], hough_space_h1.shape))
    #indices_h2 = np.array(np.unravel_index(np.argsort(hough_space_h2.ravel())[-327:], hough_space_h2.shape))


    indices_h1 = np.transpose(np.argwhere(hough_space_h1 > 0))
    indice1_max = hough_space_h1.shape[1] // 5
    if len(indices_h1[0]) > indice1_max:
      # get the most prominent points in the hough spaces
      indices_h1 = np.array(np.unravel_index(np.argsort(hough_space_h1.ravel())[-indice1_max:], hough_space_h1.shape))


    #indices_h2 = np.array(np.unravel_index(np.argsort(hough_space_h2.ravel())[-10:], hough_space_h2.shape))
    indices_h2 = np.transpose(np.argwhere(hough_space_h2 > 0))
    indice2_max = hough_space_h2.shape[1] // 5
    if len(indices_h2[0]) > indice2_max:
      # get the most prominent points in the hough spaces
      indices_h2 = np.array(np.unravel_index(np.argsort(hough_space_h2.ravel())[-indice2_max:], hough_space_h2.shape))


    #indices_h2 = np.argwhere(hough_space_h2 > 50)

    points = compute_3d_points(hough_space_h1, hough_space_h2, indices_h1, indices_h2, frame)

    # add the points to the 3d points array, with the frames index being the z coordinate
    for x in range(width):
      for y in range(width):
        points_3d[i, x, y] = points[x, y]
  # transpose the array to have the shape (width, width, num_frames)
  points_3d = points_3d.transpose(1, 2, 0)

  return points_3d

def full_hough_to_ply(dir):
    # iterate over all images in the directory
    frames = [cv2.imread(os.path.join(dir, frame), 0) for frame in os.listdir(dir)]
    z_max = len(frames)
    frames = frames[::20]
    frames = frames[::-1]

    width = frames[0].shape[1]
    # floating points for ply point cloud
    points_coordinates = []

    with tqdm(total=len(frames), desc="Processing frames") as pbar:
      for i, frame in enumerate(frames):
        hough_space_h1, hough_space_h2 = calculate_hough_spaces(frame)
        hough_space_h1, hough_space_h2 = post_process_hough_space(hough_space_h1, hough_space_h2)

        indices_h1 = np.transpose(np.argwhere(hough_space_h1 > 0))
        indice1_max = hough_space_h1.shape[1] // 5
        if len(indices_h1[0]) > indice1_max:
          # get the most prominent points in the hough spaces
          indices_h1 = np.array(np.unravel_index(np.argsort(hough_space_h1.ravel())[-indice1_max:], hough_space_h1.shape))

        indices_h2 = np.transpose(np.argwhere(hough_space_h2 > 0))
        indice2_max = hough_space_h2.shape[1] // 5
        if len(indices_h2[0]) > indice2_max:
          # get the most prominent points in the hough spaces
          indices_h2 = np.array(np.unravel_index(np.argsort(hough_space_h2.ravel())[-indice2_max:], hough_space_h2.shape))

        z = i / len(frames) * z_max
        points = compute_3d_coordinates(hough_space_h1, hough_space_h2, indices_h1, indices_h2, frame, z)
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

      


def double_image(image):
  # double the image by interpolating every second slice
  # shape
  width, height = image.shape
  new_image = np.zeros((2 * width, height), dtype=np.uint8)
  for i in range(width):
    new_image[2 * i] = image[i]
    if i < width - 1:
      new_image[2 * i + 1] = (image[i] + image[i + 1]) / 2
  return new_image


def main():
  if False: # perspective test
    img = cv2.imread(os.path.join("..", "scratch", "perspective_50mm_2m", "combined.png"), 0)
    
    hough_space_h1, hough_space_h2 = perspective_hough_transform(img, 5, 200)
    plt.imshow(hough_space_h1, cmap='jet')
    plt.colorbar()
    plt.show()


  if False: # compare 3 images
    # get 3 random integers between 120 and 980
    imgs_indices = np.random.randint(120, 980, 3)
    imgs = [cv2.imread(os.path.join("..", "scratch", "vonroi_wulsd", str(i) + ".png"), 0) for i in imgs_indices]

    h1_1, h2_1 = calculate_hough_spaces(imgs[0])
    h1_2, h2_2 = calculate_hough_spaces(imgs[1])
    h1_3, h2_3 = calculate_hough_spaces(imgs[2])

    # plot the hough spaces in a 2x3 grid
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(h1_1, cmap='jet')
    ax[0, 1].imshow(h1_2, cmap='jet')
    ax[0, 2].imshow(h1_3, cmap='jet')
    ax[1, 0].imshow(h2_1, cmap='jet')
    ax[1, 1].imshow(h2_2, cmap='jet')
    ax[1, 2].imshow(h2_3, cmap='jet')
    ax[0, 0].set_title(str(imgs_indices[0]) + ".png")
    ax[0, 1].set_title(str(imgs_indices[1]) + ".png")
    ax[0, 2].set_title(str(imgs_indices[2]) + ".png")
    plt.show()

    h1_1, h2_1 = post_process_hough_space(h1_1, h2_1)
    h1_2, h2_2 = post_process_hough_space(h1_2, h2_2)
    h1_3, h2_3 = post_process_hough_space(h1_3, h2_3)

    # plot the hough spaces in a 2x3 grid
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(h1_1, cmap='jet')
    ax[0, 1].imshow(h1_2, cmap='jet')
    ax[0, 2].imshow(h1_3, cmap='jet')
    ax[1, 0].imshow(h2_1, cmap='jet')
    ax[1, 1].imshow(h2_2, cmap='jet')
    ax[1, 2].imshow(h2_3, cmap='jet')
    ax[0, 0].set_title(str(imgs_indices[0]) + ".png")
    ax[0, 1].set_title(str(imgs_indices[1]) + ".png")
    ax[0, 2].set_title(str(imgs_indices[2]) + ".png")
    plt.show()

  if False: # curve plotting
    

    img = cv2.imread(os.path.join("..", "scratch", "vonroi_wulsd", "580.png"), 0)
    #img = cv2.imread("rendered/Test_lord_rabbit.png", 0)
    #img = double_image(img)

    plt.imshow(img, cmap='jet')
    plt.colorbar()
    plt.show()
    

    hough_space_h1, hough_space_h2 = calculate_hough_spaces(img)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(hough_space_h1, cmap='jet')
    ax1.set_title('Hough Space 1')
    ax2.imshow(hough_space_h2, cmap='jet')
    ax2.set_title('Hough Space 2')
    plt.show()

    hough_space_h1, hough_space_h2 = post_process_hough_space(hough_space_h1, hough_space_h2)

    # move the image by 500 pixels to the right
    #hough_space_h1 = np.roll(hough_space_h1, 1000, axis=1)
    #hough_space_h2 = np.roll(hough_space_h2, 1000, axis=1)

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(121)
    ax2 = fig2.add_subplot(122)
    ax1.imshow(hough_space_h1, cmap='jet')
    ax1.set_title('Hough Space 1')
    ax2.imshow(hough_space_h2, cmap='jet')
    ax2.set_title('Hough Space 2')
    plt.show()

    #plot_curves_from_hough_spaces(hough_space_h1, hough_space_h2, img)

    indices_h1 = np.transpose(np.argwhere(hough_space_h1 > 0))
    if len(indices_h1[0]) > 5000:
      # get the 5000 most prominent points in the hough spaces
      indices_h1 = np.array(np.unravel_index(np.argsort(hough_space_h1.ravel())[-5000:], hough_space_h1.shape))

    indices_h2 = np.transpose(np.argwhere(hough_space_h2 > 0))
    if len(indices_h2[0]) > 5000:
      # get the 5000 most prominent points in the hough spaces
      indices_h2 = np.array(np.unravel_index(np.argsort(hough_space_h2.ravel())[-5000:], hough_space_h2.shape))

    points = compute_3d_points(hough_space_h1, hough_space_h2, indices_h1, indices_h2, img)
    #points = cv2.GaussianBlur(points.astype(np.uint8), (9, 9), 0)
    plt.imshow(points, cmap='jet')
    plt.colorbar()
    plt.show()
    
     
  if False: # 3d Model Plotting
    dir = os.path.join("..", "scratch", "vonroi_wulsd")
    points_3d = full_hough_transform_3d(dir)

    # plot the 3d points of shape (width, width, num_frames) in 3d space
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = points_3d.nonzero()
    c = points_3d[x, y, z]
    ax.scatter(x, y, z, c=c, cmap='jet', alpha=1)
    plt.show()

  if True: # ply file creation
    #dir = os.path.join("..", "scratch", "full_persp_50mm_2.5m", "rows")
    dir = os.path.join("..", "scratch", "vonroi_wulsd")
    full_hough_to_ply(dir)
  
  if True: # 3d Model Plotting
    #cloud = o3d.io.read_point_cloud("point_cloud_1080.ply")
    cloud = o3d.io.read_point_cloud("point_cloud_test.ply")
    # normalize the colors
    colors = np.asarray(cloud.colors)
    max = np.max(colors[:, 2])
    min = np.min(colors[:, 2])
    colors[:, 2] = (colors[:, 2] - min) / (max - min)

    points = o3d.utility.Vector3dVector(cloud.points)
    # remove the points where blue is below 0.2
    #indices = np.argwhere(colors[:, 2] < 0.2)
    #indices = indices.flatten()
    #points = np.delete(points, indices, axis=0)
    #colors = np.delete(colors, indices, axis=0)
    #cloud.points = o3d.utility.Vector3dVector(points)



    # use colormap to color the point cloud
    cmap = plt.get_cmap('inferno')
    colors = cmap(colors[:, 2])[:, :3]
    cloud.colors = o3d.utility.Vector3dVector(colors)


    


    

    # Get the certainty values from the point cloud
    pcd = o3d.visualization.draw_geometries([cloud])


    


if __name__ == "__main__":
    main()