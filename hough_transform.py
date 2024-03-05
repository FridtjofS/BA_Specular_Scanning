import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
from numba import jit
import os
from image_augmentation import combine_slices

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

def test_hough_space():
    img = np.zeros((1440, 1440))
    for x in range(1440):
       for y in range(1440):
          try:
            phi = np.arcsin(((x - 720) / 581)) - ((y/1440) * 2 * np.pi)
            phi = phi % (2 * np.pi)
            phi = int(phi * 1440 / (2 * np.pi))
            if phi == 188:
              img[x, y] = 255
          except:
            continue
    plt.imshow(img, cmap='jet')
    plt.colorbar()
    plt.show()
   


def post_process_hough_space(hough_space_h1, hough_space_h2):
    # remove low frequencies by subtracting the lowpass filtered image
    hough_space_h1 = hough_space_h1 - cv2.GaussianBlur(hough_space_h1, (7, 7), 0)
    hough_space_h2 = hough_space_h2 - cv2.GaussianBlur(hough_space_h2, (7, 7), 0)

    # clip all values below 0 to 0
    hough_space_h1[hough_space_h1 < 0] = 0
    hough_space_h2[hough_space_h2 < 0] = 0

    # Weigh the matrix with weights e^(-0.001)*[1,2,...,A_max]
    for amplitude in range(1, hough_space_h1.shape[0]):
        hough_space_h1[amplitude, :] = hough_space_h1[amplitude, :] * np.exp(-0.001 * amplitude)
        hough_space_h2[amplitude, :] = hough_space_h2[amplitude, :] * np.exp(-0.001 * amplitude)

    # normalize the matrix (can be negative after weighing)
    hough_space_h1 = (hough_space_h1 - np.min(hough_space_h1)) / (np.max(hough_space_h1) - np.min(hough_space_h1))
    hough_space_h2 = (hough_space_h2 - np.min(hough_space_h2)) / (np.max(hough_space_h2) - np.min(hough_space_h2))

    hough_space_h1 = hough_space_h1 * 255
    hough_space_h2 = hough_space_h2 * 255

    # repress values below a certain threshold using Otsu's thresholding
    t1, h1 = cv2.threshold(hough_space_h1.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    t2, h2 = cv2.threshold(hough_space_h2.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)

    hough_space_h1[hough_space_h1 < t1] = 0
    hough_space_h2[hough_space_h2 < t2] = 0	

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
      if gradient_x[y, x] * gradient_y[y, x] < 0: #and edges[y, x] > 0:
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
      if gradient_x[y, x] * gradient_y[y, x] >= 0:# and edges[y, x] > 0:
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
  output[:, :, 0] = output_h1
  output[:, :, 1] = output_h2
  output[:, :, 2] = image

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


def compute_3d_points(hough_space_h1, hough_space_h2, image, original_image):
  width, num_rotations = image.shape  # Anzahl der Rotationen
  
  # initialize the 3d points as array of width x width (top view)
  points = np.zeros((width, width))


  indices_h1 = np.unravel_index(np.argsort(hough_space_h1.ravel()), hough_space_h1.shape)
  indices_h2 = np.unravel_index(np.argsort(hough_space_h2.ravel()), hough_space_h2.shape)

  # iterate over all indices
  for i in range(len(indices_h1[0])):
    # Extract the amplitude and phase for h1
    amplitude_h1, phase_h1 = indices_h1[0][i], indices_h1[1][i]

    # polar coordinates to cartesian coordinates
    x = int(amplitude_h1 * np.cos((phase_h1 / num_rotations) * 2 * np.pi)) - width // 2
    y = int(amplitude_h1 * np.sin((phase_h1 / num_rotations) * 2 * np.pi)) - width // 2

    # set the point in the 3d points array
    points[x, y] = hough_space_h1[amplitude_h1, phase_h1]

  #
  for i in range(len(indices_h2[0])):
    # Extract the amplitude and phase for h2
    amplitude_h2, phase_h2 = indices_h2[0][i], indices_h2[1][i]

    # polar coordinates to cartesian coordinates
    x = int(amplitude_h2 * np.cos((phase_h2 / num_rotations) * 2 * np.pi)) - width // 2
    y = int(amplitude_h2 * np.sin((phase_h2 / num_rotations) * 2 * np.pi)) - width // 2

    # set the point in the 3d points array
    points[x, y] = hough_space_h2[amplitude_h2, phase_h2]
  
  # plot the 3d points
  plt.imshow(points, cmap='jet')

  #fig = plt.figure()
  #ax = fig.add_subplot(111, projection='3d')
  #x = np.arange(0, width, 1)
  #y = np.arange(0, width, 1)
  #X, Y = np.meshgrid(x, y)
  #ax.plot_surface(X, Y, points, cmap='jet')
  plt.show()

def main():
  #test_hough_space()
  #return

  image = cv2.imread(os.path.join("rendered", "textured.png"),0)
  edges = cv2.Canny(image, 2, 5)
  # plot edges
  plt.imshow(edges, cmap='jet')
  plt.colorbar()
  plt.show()
  
  #edges = cv2.imread("rendered/orthographic_theta_derivative.png",0)
  #edges[edges < 10] = 0

  #edges = cv2.imread("rendered/sine_curves.png",0)
  # plot edges
  #plt.imshow(edges, cmap='jet')
  #plt.colorbar()
  #plt.show()
  

  hough_space_h1, hough_space_h2 = calculate_hough_spaces(image)

  plt.imshow(hough_space_h1, cmap='jet')
  plt.colorbar()
  plt.show()

  plt.imshow(hough_space_h2, cmap='jet')
  plt.colorbar()
  plt.show()

  hough_space_h1, hough_space_h2 = post_process_hough_space(hough_space_h1, hough_space_h2)

  plt.imshow(hough_space_h1, cmap='jet')
  plt.colorbar()
  plt.show()

  plt.imshow(hough_space_h2, cmap='jet')
  plt.colorbar()
  plt.show()
  

  #extrema_image = get_coordinates_from_input(hough_space_h1)

  hough_space_h2 = hough_space_h1

  #plot_curves_from_hough_spaces(hough_space_h1, hough_space_h2, image)

  compute_3d_points(hough_space_h1, hough_space_h2, image, image)
   
    


if __name__ == "__main__":
    main()