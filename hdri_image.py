from PIL import Image
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import find_peaks
import numpy as np


# create a random HDRI image, which is noisy horizontally, but stable vertically
# so it's random vertical lines

def create_hdri_image(width, height):
    img = Image.new('RGB', (width, height), "black")
    pixels = img.load()
    vertical_values = [random.randint(0, 255) for _ in range(width)]
    for x in range(width):
        for y in range(height):
            pixels[x, y] = (vertical_values[x], vertical_values[x], vertical_values[x])
    return img

# upscale hdr image to factor of its size, by repeating pixels
def upscale_hdri_image(img, factor):
    width, height = img.size
    new_img = Image.new('RGB', (width * factor, height * factor), "black")
    pixels = new_img.load()
    for x in range(width):
        for y in range(height):
            for i in range(factor):
                for j in range(factor):
                    pixels[x * factor + i, y * factor + j] = img.getpixel((x, y))
    return new_img

def create_gradient_image(width, height):
    img = Image.new('RGB', (width, height), "black")
    pixels = img.load()
    # make the gradient go from black to white to black
    for x in range(width):
        for y in range(height):
            if x < width // 2:
                pixels[x, y] = ( int(x / (width / 2) * 255), int(x / (width / 2) * 255), int(x / (width / 2) * 255))
            else:
                pixels[x, y] = ( int((width - x) / (width / 2) * 255), int((width - x) / (width / 2) * 255), int((width - x) / (width / 2) * 255))
    return img

def create_2d_gradient_image(width, height):
    img = Image.new('RGB', (width, height), "black")
    pixels = img.load()
    # make the gradient go from red to blue to red horizontally, and from green to yellow to green vertically
    for x in range(width):
        for y in range(height):
            if x < width // 2:
              if y < height // 2:
                pixels[x, y] = (int(x / (width / 2) * 255), 255 - int(y / (height / 2) * 255), 255)
              else:
                pixels[x, y] = (int(x / (width / 2) * 255), 255 - int((height - y) / (height / 2) * 255), 255)

            else:
              if y < height // 2:
                pixels[x, y] = (int((width - x) / (width / 2) * 255), 255 - int(y / (height / 2) * 255), 255)
              else:
                pixels[x, y] = (int((width - x) / (width / 2) * 255), 255 - int((height - y) / (height / 2) * 255), 255)
    return img

# combine frames, which consist of a single pixel slice, into a single image 
def combine_frames(dir):
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

def combine_vertical_frames(dir):
  frames = [Image.open(dir + '/' + frame) for frame in os.listdir(dir)]
  width, height = frames[0].size
  new_img = Image.new('RGBA', (width * len(frames), height), "black")
  pixels = new_img.load()
  for i, frame in enumerate(frames):
    for x in range(width):
      for y in range(height):
        pixel = frame.getpixel((x, y))
        pixels[i * width + x, y] = pixel
  return new_img

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

def map_to_gradient(img):
  #map intesity to gradient from black to red to yellow to white
  width, height = img.size
  pixels = img.load()
  new_img = Image.new('RGBA', (width, height), "black")
  new_pixels = new_img.load()
  for x in range(width):
    for y in range(height):
      intensity = pixels[x, y][0]
      if intensity < 64:
        new_pixels[x, y] = (intensity * 4, 0, 0, 255)
      elif intensity < 128:
        new_pixels[x, y] = (255, intensity * 4 - 255, 0, 255)
      elif intensity < 192:
        new_pixels[x, y] = (255, 255, intensity * 4 - 255 * 2, 255)
      else:
        new_pixels[x, y] = (255, 255, 255, 255)
  return new_img

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

def estimate_normals(img):
   # estimate normals from the image
   # the enviroment map is assumed to be a gradient
    # the normals are estimated by taking the gradient of the image
    # and normalizing it
    # the gradient is estimated by taking the difference between the pixel and the pixel to the left
    # and the pixel and the pixel above it
    # the gradient is then normalized
   
    width, height = img.size
    pixels = img.load()
    normals = Image.new('RGBA', (width, height), "black")
    normal_pixels = normals.load()
    for x in range(width):
        for y in range(height):
            if x == 0 or y == 0:
                normal_pixels[x, y] = (0, 0, 0, 0)
            else:
                dx = (pixels[x, y][0] - pixels[x - 1, y][0], pixels[x, y][1] - pixels[x - 1, y][1], pixels[x, y][2] - pixels[x - 1, y][2])
                dy = (pixels[x, y][0] - pixels[x, y - 1][0], pixels[x, y][1] - pixels[x, y - 1][1], pixels[x, y][2] - pixels[x, y - 1][2])
                magnitude = (dx[0] ** 2 + dx[1] ** 2 + dx[2] ** 2 + dy[0] ** 2 + dy[1] ** 2 + dy[2] ** 2) ** 0.5
                if magnitude == 0:
                    normal_pixels[x, y] = (0, 0, 0, 0)
                else:
                  normal_pixels[x, y] = (int(dx[0] / magnitude * 255), int(dx[1] / magnitude * 255), int(dx[2] / magnitude * 255), 255)
    return normals


def canny_edge_detection(img):
  width, height = img.size
  pixels = img.load()
  new_img = Image.new('RGBA', (width, height), "black")
  new_pixels = new_img.load()
  for x in range(width):
    for y in range(height):
      if x == 0 or y == 0 or x == width - 1 or y == height - 1:
        new_pixels[x, y] = (0, 0, 0, 255)
      else:
        dx = (pixels[x + 1, y][0] - pixels[x - 1, y][0], pixels[x + 1, y][1] - pixels[x - 1, y][1], pixels[x + 1, y][2] - pixels[x - 1, y][2])
        dy = (pixels[x, y + 1][0] - pixels[x, y - 1][0], pixels[x, y + 1][1] - pixels[x, y - 1][1], pixels[x, y + 1][2] - pixels[x, y - 1][2])
        magnitude = (dx[0] ** 2 + dx[1] ** 2 + dx[2] ** 2 + dy[0] ** 2 + dy[1] ** 2 + dy[2] ** 2) ** 0.5
        if magnitude > 10:
          new_pixels[x, y] = (255, 255, 255, 255)
        else:
          new_pixels[x, y] = (0, 0, 0, 255)
  return new_img


def calculate_hough_spaces(image):
    
    width, num_rotations = image.shape  # Anzahl der Rotationen

    # Schritt 1: Kanten im Bild erkennen
    #edges = cv2.Canny(image, 30, 100)
    edges = image
    # clip edges below a certain threshold
    edges[edges < 10] = 0

    #edges = cv2.imread('cat_test_canny.png',0)
    # invert the image
    #edges = cv2.bitwise_not(edges)

    # plot edges
    plt.imshow(edges, cmap='inferno')
    plt.colorbar()
    plt.show()

    x_center = width // 2

    # Schritt 2: Hough-Räume initialisieren
    hough_space_h1 = np.zeros((width // 2, num_rotations))
    hough_space_h2 = np.zeros((width // 2, num_rotations))

    #plt.imshow(hough_space_h1, cmap='inferno')
    #plt.colorbar()
    #plt.show()

    for x in range(1,width - 1):
        for theta in range(num_rotations):
            if edges[x, theta] > 0:
                for amplitude in range(1, width // 2):
                    if amplitude == 0:
                        continue
                    elif amplitude < np.abs(x - x_center):
                        #print("amplitude < np.abs(x - x_center), amplitude: " + str(amplitude) + ", x: " + str(x))
                        continue
                    try:
                      phi = 0
                      
                      if (int(image[x, theta]) - int(image[x - 1, theta])) > 0:
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
    plt.imshow(hough_space_h1, cmap='jet')
    plt.colorbar()
    plt.show()

    plt.imshow(hough_space_h2, cmap='jet')
    plt.colorbar()
    plt.show()

    # save the hough spaces as images
    cv2.imwrite('hough_space_h1_noCanny.png', hough_space_h1 * 255)
    cv2.imwrite('hough_space_h2_noCanny.png', hough_space_h2 * 255)

    return hough_space_h1, hough_space_h2


def plot_curves_from_hough_spaces(hough_space_h1, hough_space_h2, image, original_image):
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
    hough_space_h1[x_center - x:, :] = 0
    hough_space_h2[x_center - x:, :] = 0

    # ignore values above 100
    hough_space_h1[hough_space_h1 > 80] = 0
    hough_space_h2[hough_space_h2 > 80] = 0


    plt.imshow(hough_space_h1, cmap='jet')
    plt.colorbar()
    plt.show()

    # Calculate the Euclidean distance between two points
    def euclidean_distance(point1, point2):
      return np.sqrt(np.sum((point1 - point2) ** 2))
    

    # Set the threshold for minimum distance between extrema
    min_distance_threshold = 5

    # Get the indices of the 20 most prominent extrema in hough_space_h1
    indices_h1 = np.unravel_index(np.argsort(hough_space_h1.ravel())[-3000:], hough_space_h1.shape)

    # Get the indices of the 20 most prominent extrema in hough_space_h2
    indices_h2 = np.unravel_index(np.argsort(hough_space_h2.ravel())[-3000:], hough_space_h2.shape)

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

        
    extrema_image = np.zeros(hough_space_h1.shape)
    for x,y in filtered_indices_h1:
      extrema_image[x,y] = 1

    plt.imshow(extrema_image, cmap='jet')
    plt.colorbar()
    plt.show()

    


    image2 = np.zeros((width, num_rotations))

    for i in range(len(filtered_indices_h1)):
      # Extract the amplitude and phase for h1
      max_amp_h1, max_phase_h1 = indices_h1[0][i], indices_h1[1][i]
      max_amp_h1 = max_amp_h1 / (hough_space_h1.shape[0])
      max_phase_h1 = max_phase_h1 * 2 * np.pi / num_rotations

      # Extract the amplitude and phase for h2
      #max_amp_h2, max_phase_h2 = indices_h2[0][i], indices_h2[1][i]
      #max_amp_h2 = max_amp_h2 * 2 * np.pi / num_rotations
      #max_phase_h2 = max_phase_h2 * 2 * np.pi / num_rotations

      for y in range(num_rotations):
        theta = y * 2 * np.pi / num_rotations
        x = int(x_center + ((max_amp_h1 * np.sin(-theta - max_phase_h1)) * x_center))
        if x < 0 or x >= width:
          print("Error x: " + str(x))
          continue
        image2[y, x] = 1
        #x2 = int(x_center + ((max_amp_h2 * np.cos(theta - max_phase_h2))) * x_center)
        #if x2 < 0 or x2 >= width:
        #  continue
        #image2[y, x2] = 1

    plt.imshow(image2, cmap='jet')
    plt.colorbar()
    plt.show()

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
    #img = create_hdri_image(256, 256)
    #img = upscale_hdri_image(img, 5)
    #img.save('hdri_image.png')

    #gradient = create_2d_gradient_image(256, 256)
    #gradient = upscale_hdri_image(gradient, 5)
    #gradient.save('gradient2d.png')


    #img = combine_frames('rendered/orthographic')
    #img.show()
    #img.save('rendered/orthographic.png')

    #img = Image.open('rendered/orthographic.png').convert("RGBA")

    ##horiz = horiz_derivative(img)
    ##vert = vert_derivative(img)

    ## img to np array
    #img = np.array(img)
    ## apply canny edge detection
    #canny = cv2.Canny(img, 100, 200)
    ## show the image
    #plt.imshow(canny, cmap='gray')

    #horiz.show()
    #vert.show()
    #div.show()


    #map_to_gradient(img).show()

    #canny_edge_detection(img).show()

    #normals = estimate_normals(img)
    #normals.show()

    #depth = integrate_normals(normals)
    #depth.show()

    #horiz = horiz_derivative(img)
    #horiz.show()
    #horiz.save('rendered/orthographic_x_derivative.png')

    #vert = vert_derivative(img)
    #vert.show()
    #vert.save('rendered/orthographic_theta_derivative.png')
    #width, height = img.size
    
          


    #img.save('rendered/test2.png')

    #slice = get_pixel_slice(img, img.size[1] // 2)
    #slice_deriv = get_slice_derivative(slice)
    #map_to_gradient(widen_slice(slice_deriv, 100)).show()

    #slice = get_pixel_slice(img, img.size[1] // 2, False)
    #slice_deriv = get_slice_derivative(slice, True)
    #map_to_gradient(widen_slice(slice, 100, False)).show()
    #slice = diff_and_integrate_slice(slice)
    #slice2 = get_pixel_slice(img, img.size[0] // 2 + 100)
    #slice2 = diff_and_integrate_slice(slice2)

    #plt.plot(slice)

    #slice3 = get_pixel_slice(img, img.size[0] // 2 + 2)
    #slice3 = diff_and_integrate_slice(slice3)

    # plot all slices in one graph
    #plt.plot(slice)
    #plt.plot(slice2)
    #plt.plot(slice3)
    #plt.show()
    #stitched = stitch_slice_diffs(img, False)
    #difference = [slice[i] - stitched[i] for i in range(len(slice))]
    #plt.plot(stitched)
    #plt.plot(difference)
    #plt.show()

    img = cv2.imread('orthographic_theta_derivative_testtt.png',0)
    #img = cv2.imread('cat_test.png',0)
    #img = cv2.imread('rendered/netball_env_test.png',0)

    original_image = cv2.imread('rendered/orthographic.png',0)

    #hough_space_h1, hough_space_h2 = calculate_hough_spaces(img)

    hough_space_h1 = cv2.imread('hough_space_h1_noCanny.png',0)
    hough_space_h2 = cv2.imread('hough_space_h2_noCanny.png',0)

    plot_curves_from_hough_spaces(hough_space_h1, hough_space_h2, img, original_image)


    #plot_hough_transform1point(img, 673, 55)





   #plt.imshow(img, cmap='inferno')
   #plt.colorbar()
   #plt.show()




    


if __name__ == '__main__':
    main()