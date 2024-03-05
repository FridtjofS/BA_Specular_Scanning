import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from numba import jit

def calculate_hough_spaces(image):
  edges = cv2.Canny(image, 20, 50)

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
          if gradient_x[theta, x] * gradient_y[theta, x] > 0:
            # calculate the corresponding hough space phi
            phi = np.arcsin((x - x_center) / amplitude) - ((theta / num_rotations) * 2 * np.pi)
            phi = phi % (2 * np.pi)
            phi = int(phi / (2 * np.pi) * (num_rotations - 1))
            # add the amplitude to the corresponding hough space
            hough_space_h1[amplitude, phi] += 1
          else:
            # calculate the corresponding hough space phi
            phi = np.arcsin((x - x_center) / amplitude) - ((theta / num_rotations) * 2 * np.pi) + (np.pi / 2)
            phi = phi % (2 * np.pi)
            phi = int(phi / (2 * np.pi) * (num_rotations - 1))
            # add the amplitude to the corresponding hough space
            hough_space_h2[amplitude, phi] += 1


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

      # get the indices of the 20 most prominent extrema in hough_space_h1
      indices_h1 = np.unravel_index(np.argsort(hough_space_h1.ravel())[-1000:], hough_space_h1.shape)
      indices_h2 = np.unravel_index(np.argsort(hough_space_h2.ravel())[-500:], hough_space_h2.shape)
      extrema_image = np.zeros(hough_space_h1.shape)
      for i in range(len(indices_h1[0])):
        extrema_image[indices_h1[0][i], indices_h1[1][i]] = 1
         

      '''

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
    '''

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

    # sort the filtered_indices_h1 by the amplitude
    filtered_indices_h1 = sorted(filtered_indices_h1, key=lambda x: x[0], reverse=True)
    filtered_indices_h2 = sorted(filtered_indices_h2, key=lambda x: x[0], reverse=False)

    
    #min_phi = 1000
    #max_phi = 0
    #min_phase = 1000
    #max_phase = 0
    #min_theta = 1000
    #max_theta = 0

    image2 = np.zeros((width, num_rotations))

    for i in range(len(filtered_indices_h1)):
      # Extract the amplitude and phase for h1
      max_amp_h1, max_phase_h1 = filtered_indices_h1[i][0], filtered_indices_h1[i][1]
      #max_amp_h1 = max_amp_h1 / (hough_space_h1.shape[0])
      max_phase_h1 = max_phase_h1 * 2 * np.pi / num_rotations

      #print("max_amp_h1: " + str(max_amp_h1) + ", max_phase_h1: " + str(max_phase_h1))

      # Extract the amplitude and phase for h2
      #max_amp_h2, max_phase_h2 = indices_h2[0][i], indices_h2[1][i]
      #max_amp_h2 = max_amp_h2 * 2 * np.pi / num_rotations
      #max_phase_h2 = max_phase_h2 * 2 * np.pi / num_rotations

      for y in range(num_rotations):
        theta = (y * 2 * np.pi) / num_rotations
        occlusion = (theta + max_phase_h1) % (2 * np.pi) 
        if occlusion <= np.pi / 2 or occlusion >= 3 * np.pi / 2:
          x = int(x_center + ((max_amp_h1 * np.sin((max_phase_h1 + theta)))))
          mode = 2
          if mode == 0:
            image2[y, x] += 1
          if mode == 1:
            image2[y, x] = max_amp_h1 if max_amp_h1 > image2[y, x] else image2[y, x]
          elif mode == 2:
            image2[y, x] = max_phase_h1 if max_phase_h1 > image2[y, x] else image2[y, x]
          else:
            image2[y, x] = 1
           

        #if not((max_phase_h1 + theta)%(2*np.pi) <= np.pi / 2 or (max_phase_h1 + theta)%(2*np.pi) >= 3 * np.pi/2):
        #  continue 
        #min_phi = max_phase_h1 + theta if max_phase_h1 + theta < min_phi else min_phi
        #max_phi = max_phase_h1 + theta if max_phase_h1 + theta > max_phi else max_phi
        #min_phase = max_phase_h1 if max_phase_h1 < min_phase else min_phase
        #max_phase = max_phase_h1 if max_phase_h1 > max_phase else max_phase
        #min_theta = theta if theta < min_theta else min_theta
        #max_theta = theta if theta > max_theta else max_theta
        
        #x = int(x_center + ((max_amp_h1 * np.sin((max_phase_h1 + theta)))))# * x_center))
        #if x < 0 or x >= width:
        #  print("Error x: " + str(x))
        #  continue
        #
        #mode = 2
        #if mode == 0:
        #  image2[y, x] += 1
        #if mode == 1:
        #  image2[y, x] = max_amp_h1 if max_amp_h1 > image2[y, x] else image2[y, x]
        #elif mode == 2:
        #  image2[y, x] = max_phase_h1 if max_phase_h1 > image2[y, x] else image2[y, x]
        #else:
        #  image2[y, x] = 1
        #x2 = int(x_center + ((max_amp_h2 * np.cos(theta - max_phase_h2))) * x_center)
        #if x2 < 0 or x2 >= width:
        #  continue
        #image2[y, x2] = 1

    plt.imshow(image2, cmap='jet')
    plt.colorbar()
    plt.show()
    # save the image2 as an image
    cv2.imwrite('rendered/sine curves.png', image2 * 255)

    #print ("min_phi: " + str(min_phi) + ", max_phi: " + str(max_phi))
    #print ("min_phase: " + str(min_phase) + ", max_phase: " + str(max_phase))
    #print ("min_theta: " + str(min_theta) + ", max_theta: " + str(max_theta))

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
  
def main():

  image = cv2.imread(os.path.join("rendered", "textured.png"),0)
  #edges = cv2.Canny(image, 20, 50)
  # pad the image with 100 pixels of black on left and right
  edges = np.pad(image, ((0, 0), (300, 300)), 'constant', constant_values=0)

  image = edges
  hough_space_h1, hough_space_h2 = calculate_hough_spaces(image)
  plt.imshow(hough_space_h1, cmap='jet')
  plt.colorbar()
  plt.show()

  plt.imshow(hough_space_h2, cmap='jet')
  plt.colorbar()
  plt.show()

if __name__ == "__main__":
  main()