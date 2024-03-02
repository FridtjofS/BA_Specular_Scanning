import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
from numba import jit
import os
from image_augmentation import combine_slices

@jit(nopython=True)
def calculate_hough_spaces(image):
    
    width, num_rotations = image.shape  # Anzahl der Rotationen

    # Schritt 1: Kanten im Bild erkennen
    #edges = cv2.Canny(image, 30, 100)
    edges = image
    # clip edges below a certain threshold
    #edges[edges < 10] = 0
    #for i in range(width):
    #  for j in range(num_rotations):
    #    if edges[i, j] < 10:
    #      edges[i, j] = 0

    #edges = cv2.imread('cat_test_canny.png',0)
    # invert the image
    #edges = cv2.bitwise_not(edges)

    # plot edges
    #plt.imshow(edges, cmap='inferno')
    #plt.colorbar()
    #plt.show()

    x_center = width // 2

    # Schritt 2: Hough-Räume initialisieren
    hough_space_h1 = np.zeros((width // 2, num_rotations))
    hough_space_h2 = np.zeros((width // 2, num_rotations))

    #plt.imshow(hough_space_h1, cmap='inferno')
    #plt.colorbar()
    #plt.show()

    for x in range(1,width - 1):
        for theta in range(1, num_rotations):
            if edges[x, theta] > 0:
                for amplitude in range(1, width // 2):
                    if amplitude == 0:
                        continue
                    elif amplitude < np.abs(x - x_center):
                        #print("amplitude < np.abs(x - x_center), amplitude: " + str(amplitude) + ", x: " + str(x))
                        continue
                    try:
                      phi = 0
                      
                      if ((int(image[x, theta]) - int(image[x - 1, theta])) / (int(image[x, theta]) - int (image[x, theta - 1]))) > 0: 
                        phi = np.arcsin(((x - x_center) / amplitude)) - ((theta/num_rotations) * 2 * np.pi)
                        # stretch phi from [-2pi, pi/2] to [0, num_rotations]
                        phi = (phi + 2 * np.pi) * num_rotations / (2.5 * np.pi)
                        phi = int(phi)

                        hough_space_h1[amplitude, phi] += 1


                      else:
                        phi = np.arccos((x - x_center) / amplitude) - ((theta/num_rotations) * 2 * np.pi) + (np.pi / 2)
                        # stretch phi from [-3pi/2, pi] to [0, num_rotations]
                        phi = (phi + 3 * np.pi / 2) * num_rotations / (2.5 * np.pi)
                        phi = int(phi)

                        hough_space_h2[amplitude, phi] += 1
                      


                      
                      #hough_space_h1[amplitude, phi] += 1
                      
                      #hough_space_h2[amplitude, phi] += 1
                    except:
                      #print("Error: " + str(x) + ", " + str(theta) + ", " + str(amplitude))
                      continue


    # remove low frequencies by subtracting the lowpass filtered image
    #hough_space_h1 = hough_space_h1 - cv2.GaussianBlur(hough_space_h1, (5, 5), 0)
    #hough_space_h2 = hough_space_h2 - cv2.GaussianBlur(hough_space_h2, (5, 5), 0)

    # Weigh the matrix with weights e^(-0.001)*[1,2,...,A_max]
    for amplitude in range(1, width // 2):
        hough_space_h1[amplitude, :] = hough_space_h1[amplitude, :] * np.exp(-0.001 * amplitude)
        hough_space_h2[amplitude, :] = hough_space_h2[amplitude, :] * np.exp(-0.001 * amplitude)

    # normalize the matrix
    hough_space_h1 = hough_space_h1 / np.max(hough_space_h1)
    hough_space_h2 = hough_space_h2 / np.max(hough_space_h2)

    # repress values below a certain threshold
    #hough_space_h1[hough_space_h1 < 0.3] = 0
    #hough_space_h2[hough_space_h2 < 0.3] = 0

    # plot Hough-Räume
    #plt.imshow(hough_space_h1, cmap='jet')
    #plt.colorbar()
    #plt.show()

    #plt.imshow(hough_space_h2, cmap='jet')
    #plt.colorbar()
    #plt.show()

    # save the hough spaces as images
    #cv2.imwrite('hough_space_h1_2.png', hough_space_h1 * 255)
    #cv2.imwrite('hough_space_h2_2.png', hough_space_h2 * 255)

    return hough_space_h1, hough_space_h2


def plot_curves_from_hough_spaces(hough_space_h1, hough_space_h2, image, original_image, extrema_image = None):
    width, num_rotations = image.shape  # Anzahl der Rotationen
    x_center = width // 2

    # find the smallest non zero x value in the image
    x = width
    for i in range(num_rotations - 1):
       for j in range(width - 1):
          if original_image[j, i] > 0:
              if j < x:
                x = j
                #print(j)
              break

    #plt.imshow(original_image, cmap='jet')
    #plt.colorbar()
    #plt.show()

    #print (x)
    x = 90
    


    # set values to 0 which have a bigger y than (x_center - x)
    #hough_space_h1[x_center - x:, :] = 0
    #hough_space_h2[x_center - x:, :] = 0


    # ignore values above 100
    #hough_space_h1[hough_space_h1 > 80] = 0
    #hough_space_h2[hough_space_h2 > 80] = 0

    if extrema_image is None:
      plt.imshow(hough_space_h1, cmap='jet')
      plt.colorbar()
      plt.show()

      # Calculate the Euclidean distance between two points
      def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
      

      # Set the threshold for minimum distance between extrema
      min_distance_threshold = 20

      # Get the indices of the 20 most prominent extrema in hough_space_h1
      indices_h1 = np.unravel_index(np.argsort(hough_space_h1.ravel())[-5000:], hough_space_h1.shape)

      # Get the indices of the 20 most prominent extrema in hough_space_h2
      indices_h2 = np.unravel_index(np.argsort(hough_space_h2.ravel())[-5000:], hough_space_h2.shape)

      # Filter out extrema that are very close together
      filtered_indices_h1 = []
      filtered_indices_h2 = []

      for i in range(len(indices_h1[0])):
        point1 = np.array([indices_h1[0][i], indices_h1[1][i]])
        is_close = False

        for j in range(len(filtered_indices_h1)):
          point2 = np.array([filtered_indices_h1[j][0], filtered_indices_h1[j][1]])
          distance = euclidean_distance(point1, point2)

          if distance < min_distance_threshold:
            is_close = True
            break

        if not is_close:
          filtered_indices_h1.append(point1)

      for i in range(len(indices_h2[0])):
        point1 = np.array([indices_h2[0][i], indices_h2[1][i]])
        is_close = False

        for j in range(len(filtered_indices_h2)):
          point2 = np.array([filtered_indices_h2[j][0], filtered_indices_h2[j][1]])
          distance = euclidean_distance(point1, point2)

          if distance < min_distance_threshold:
            is_close = True
            break

        if not is_close:
          filtered_indices_h2.append(point1)

      # overwrite with filtered indices being the extrema from every column
      
      #for i in range(hough_space_h1.shape[1]):
      #  max_index = np.argmax(hough_space_h1[:, i])
      #  filtered_indices_h1.append(np.array([max_index, i]))

          
      extrema_image = np.zeros(hough_space_h1.shape)
      for x,y in filtered_indices_h1:
        extrema_image[x,y] = 1


    plt.imshow(extrema_image, cmap='jet')
    plt.colorbar()
    plt.show()

    filtered_indices_h1 = []
    filtered_indices_h2 = []
    for i in range(extrema_image.shape[0]):
      for j in range(extrema_image.shape[1]):
        if extrema_image[i, j] > 0:
          filtered_indices_h1.append((i, j))
          filtered_indices_h2.append((i, j))

    
    min_phi = 1000
    max_phi = 0
    min_phase = 1000
    max_phase = 0
    min_theta = 1000
    max_theta = 0

    image2 = np.zeros((width, num_rotations))

    for i in range(len(filtered_indices_h1)):
      # Extract the amplitude and phase for h1
      max_amp_h1, max_phase_h1 = filtered_indices_h1[i][0], filtered_indices_h1[i][1]
      #max_amp_h1 = max_amp_h1 / (hough_space_h1.shape[0])
      max_phase_h1 = max_phase_h1 * 2 * np.pi / num_rotations

      # Extract the amplitude and phase for h2
      #max_amp_h2, max_phase_h2 = indices_h2[0][i], indices_h2[1][i]
      #max_amp_h2 = max_amp_h2 * 2 * np.pi / num_rotations
      #max_phase_h2 = max_phase_h2 * 2 * np.pi / num_rotations

      for y in range(num_rotations):
        theta = y * 2 * np.pi / num_rotations

        #if not((max_phase_h1 + theta)%(2*np.pi) <= np.pi / 2 or (max_phase_h1 + theta)%(2*np.pi) >= 3 * np.pi/2):
        #  continue 
        min_phi = max_phase_h1 + theta if max_phase_h1 + theta < min_phi else min_phi
        max_phi = max_phase_h1 + theta if max_phase_h1 + theta > max_phi else max_phi
        min_phase = max_phase_h1 if max_phase_h1 < min_phase else min_phase
        max_phase = max_phase_h1 if max_phase_h1 > max_phase else max_phase
        min_theta = theta if theta < min_theta else min_theta
        max_theta = theta if theta > max_theta else max_theta
        
        x = int(x_center + ((max_amp_h1 * np.sin((max_phase_h1 + theta)))))# * x_center))
        if x < 0 or x >= width:
          print("Error x: " + str(x))
          continue
        
        mode = 2
        if mode == 0:
          image2[y, x] += 1
        if mode == 1:
          image2[y, x] = max_amp_h1 if max_amp_h1 > image2[y, x] else image2[y, x]
        elif mode == 2:
          image2[y, x] = max_phase_h1 if max_phase_h1 > image2[y, x] else image2[y, x]
        else:
          image2[y, x] = 1
        #x2 = int(x_center + ((max_amp_h2 * np.cos(theta - max_phase_h2))) * x_center)
        #if x2 < 0 or x2 >= width:
        #  continue
        #image2[y, x2] = 1

    plt.imshow(image2, cmap='jet')
    plt.colorbar()
    plt.show()
    print ("min_phi: " + str(min_phi) + ", max_phi: " + str(max_phi))
    print ("min_phase: " + str(min_phase) + ", max_phase: " + str(max_phase))
    print ("min_theta: " + str(min_theta) + ", max_theta: " + str(max_theta))

    # canny edge detection
    edges = cv2.Canny(image, 30, 100)

    # add image2 to the edges
    #edges = (edges + image2 * 255) / 2

    edges_new = np.zeros((width, num_rotations, 3))

    edges_new[:,:,0] = edges
    edges_new[:,:,1] = image2 * 255

    
    # plot edges
    plt.imshow(edges_new)
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



def main():
  hough_space_h1, hough_space_h2 = calculate_hough_spaces(cv2.imread(os.path.join("rendered", "orthographic_theta_derivative.png"),0))

  plt.imshow(hough_space_h1, cmap='jet')
  plt.colorbar()
  plt.show()

  plt.imshow(hough_space_h2, cmap='jet')
  plt.colorbar()
  plt.show()
   
    


if __name__ == "__main__":
    main()