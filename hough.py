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
    
    hough_space[gradient_hough <= t] = 0

    hough_space = cv2.GaussianBlur(hough_space, (5, 5), 0)

    coordinates_hough = sk.feature.peak_local_max(hough_space, threshold_rel = 0.25, min_distance=10)

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
    

    #hough_temp = smooth_hough_space(hough_temp, coordinates_hough)
    hough_space = hough_space * hough_temp

    return hough_space

def smooth_hough_space(hough_space, coordinates):
    
    # smooth out the coordinates by considering the neighbourhood
    hough_temp = np.zeros(hough_space.shape)

    for x, y in coordinates:
        window_size = 40
        # consider all x and y values in the window
        new_x = 0
        new_y = 0
        count = 0

        for i in range(-window_size, window_size):
            for j in range(-window_size, window_size):
                if x + i < 0 or x + i >= hough_space.shape[0] or y + j < 0 or y + j >= hough_space.shape[1]:
                    continue
                if hough_space[x + i, y + j] > 0:
                    new_x += x + i
                    new_y += y + j
                    count += 1

        new_x = new_x // count
        new_y = new_y // count
        if new_x >= 0 and new_x < hough_space.shape[0] and new_y >= 0 and new_y < hough_space.shape[1]:
            hough_temp[int(new_x), int(new_y)] = 1

    return hough_temp
        

def curve_fitting(hough_space, splits, indices):
    from scipy.interpolate import splprep, splev

    #tck, u = splprep([indices[:, 1], indices[:, 0]], s=0.0, per=False, k=2)
    #u_new = np.linspace(u.min(), u.max(), 1000)
    #x_new, y_new = splev(u_new, tck, der=0)
    x_new = []
    y_new = []
    if len(splits) == 1:
        tck, u = splprep([indices[:, 1], indices[:, 0]], s=0.0, per=False, k=2)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x, y = splev(u_new, tck, der=0)
        x_new.extend(x)
        y_new.extend(y)
    else: 
      for i in range(len(splits) - 1):
          if splits[i + 1][0] - splits[i][0] > 2:
              loop = True if splits[i][0] > 0 else False
              tck, u = splprep([indices[splits[i][0]:splits[i + 1][0], 1], indices[splits[i][0]:splits[i + 1][0], 0]], s=0.0, per=loop, k=2)
              u_new = np.linspace(u.min(), u.max(), hough_space.shape[1] - 1)
              x, y = splev(u_new, tck, der=0)
              x_new.extend(x)
              y_new.extend(y)

    # create a new hough space with the fitted curve
    fitted_hough_space = np.zeros(hough_space.shape)
    for i in range(len(x_new)):
        fitted_hough_space[int(y_new[i]), int(x_new[i])] = 1

    for index in indices:
        fitted_hough_space[int(index[0]), int(index[1])] = 2

    return fitted_hough_space




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
    frame_names = os.listdir(dir)
    z_max = len(frame_names)
    # sort the frames named 0.png, 1.png, ..., 100.png, ...
    frame_names.sort(key=lambda x: int(str(os.path.basename(str(x))).split('.')[0]))
    #frame_names = frame_names[::20]
    frames = [cv2.imread(os.path.join(dir, frame), 0) for frame in frame_names]
    
    #frames = frames[::20]
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

    #points_coordinates = smooth_point_cloud(points_coordinates)    

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

def smooth_point_cloud(points):
    return points
    avg = 0
    # smooth the point cloud by considering the neighbourhood
    for i in tqdm(range(len(points))):
        x, y, z, intensity = points[i]
        window_size = 50
        # consider all x and y values in the window
        new_x = x
        new_y = y
        new_z = z
        count = 1

        # coordinates are real numbers
        for j in range(len(points)):
            if i == j:
                continue
            x2, y2, z2, intensity2 = points[j]
            dist = np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + (z - z2) ** 2)
            if dist < window_size:
                new_x += x2
                new_y += y2
                new_z += z2
                count += 1

        new_x = new_x / count
        new_y = new_y / count
        new_z = new_z / count
        points[i] = (new_x, new_y, new_z, intensity)
        avg += count

    print(avg / len(points))
    return points


def main():

  if False: # curve Fitting Test
    # get 3 random integers between 120 and 980
    imgs_indices = np.random.randint(120, 980, 3)
    imgs = [cv2.imread(os.path.join("..", "scratch", "vonroi_wulsd", str(i) + ".png"), 0) for i in imgs_indices]

    hough_space_1 = calculate_hough_space(imgs[0])
    hough_space_2 = calculate_hough_space(imgs[1])
    hough_space_3 = calculate_hough_space(imgs[2])

    # plot the hough spaces in a 2x3 grid
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(hough_space_1, cmap='jet')
    ax[1].imshow(hough_space_2, cmap='jet')
    ax[2].imshow(hough_space_3, cmap='jet')
    ax[0].set_title(str(imgs_indices[0]) + ".png")
    ax[1].set_title(str(imgs_indices[1]) + ".png")
    ax[2].set_title(str(imgs_indices[2]) + ".png")
    plt.show()

    hough_space_1 = post_process_hough_space(hough_space_1)
    hough_space_2 = post_process_hough_space(hough_space_2)
    hough_space_3 = post_process_hough_space(hough_space_3)

    indices_1 = sk.feature.peak_local_max(hough_space_1, threshold_rel = 0.1, min_distance=10)
    indices_2 = sk.feature.peak_local_max(hough_space_2, threshold_rel = 0.1, min_distance=10)
    indices_3 = sk.feature.peak_local_max(hough_space_3, threshold_rel = 0.1, min_distance=10)

    # sort by distance to each point starting from the leftmost point
    indices_1 = indices_1[indices_1[:, 1].argsort()]
    indices_2 = indices_2[indices_2[:, 1].argsort()]
    indices_3 = indices_3[indices_3[:, 1].argsort()]

    def sort_by_lowest_cost(indices, threshold=100):
        """
          Check every order of indices and return the one with the lowest cost
          we define the cost as the sum of the distances between the points
        """
        order = np.arange(len(indices))
        # sort by distance to each point starting from the leftmost point
        for i in range(1, len(indices) - 4):
          min_dist = np.inf
          min_index = 0
          next_indices = order.copy()[i: i+5]
          # get distance matrix
          matrix = np.zeros((5, 5))
          for j in range(5):
            for k in range(5):
              matrix[j, k] = np.sqrt((indices[next_indices[j], 0] - indices[next_indices[k], 0]) ** 2 + (indices[next_indices[j], 1] - indices[next_indices[k], 1]) ** 2)
          # get the order with the lowest cost
          for j in range(5):
            for k in range(5):
              for l in range(5):
                for m in range(5):
                  if j != k and k != l and l != m and j != l and j != m and k != m:
                    dist = matrix[0, j] + matrix[j, k] + matrix[k, l] + matrix[l, m]
                    if dist < min_dist:
                      min_dist = dist
                      min_index = j
          order[i] = next_indices[min_index]
          order[next_indices[min_index]] = next_indices[0]

        print(order)
        indices = indices[order]
        splits = add_splits(indices, threshold)
        return splits, indices

    def add_splits(indices, threshold=100):
        splits = []
        splits.append([0, 0])
        splits.append([len(indices), 0])
        return splits
        for i in range(1, len(indices)):
            dist = np.sqrt((indices[i - 1, 0] - indices[i, 0]) ** 2 + (indices[i - 1, 1] - indices[i, 1]) ** 2)
            if dist > threshold:
                last_split = splits[-1][0]
                dist = np.sqrt((indices[last_split, 0] - indices[i, 0]) ** 2 + (indices[last_split, 1] - indices[i, 1]) ** 2)
                if dist < threshold:
                    splits.append([i, 1])
                else:
                    splits.append([i, 0])
        last_split = splits[-1][0]
        dist = np.sqrt((indices[last_split, 0] - indices[-1, 0]) ** 2 + (indices[last_split, 1] - indices[-1, 1]) ** 2)
        if dist < threshold:
            splits.append([len(indices), 1])
        else:
            splits.append([len(indices), 0])

        return splits

    # now go through the points and sort them by distance to the previous point
    def sort_by_distance(indices, threshold=100):
        sorted_indices = np.zeros(indices.shape)
        splits = []
        splits.append([0, 0])
        sorted_indices[0] = indices[0]
        indices = np.delete(indices, 0, axis=0)
        for i in range(1, len(sorted_indices)):
            min_dist = 1000000
            min_index = 0
            for j in range(len(indices)):
                dist = np.sqrt((sorted_indices[i - 1, 0] - indices[j, 0]) ** 2 + (sorted_indices[i - 1, 1] - indices[j, 1]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            if min_dist > threshold:
                # check for closed loop
                last_split = splits[-1][0]
                dist = np.sqrt((sorted_indices[last_split, 0] - indices[0, 0]) ** 2 + (sorted_indices[last_split, 1] - indices[0, 1]) ** 2)
                if dist < threshold:
                  splits.append([i, 1])
                else:
                  splits.append([i, 0])
                # check for closed loop
            sorted_indices[i] = indices[min_index]
            indices = np.delete(indices, min_index, axis=0)

        # check for closed loop
        last_split = splits[-1][0]
        length = sorted_indices.shape[0] - 1
        dist = np.sqrt((sorted_indices[last_split, 0] - sorted_indices[length, 0]) ** 2 + (sorted_indices[last_split, 1] - sorted_indices[length, 1]) ** 2)
        if dist < threshold:
            splits.append([len(sorted_indices), 1])
        else:
            splits.append([len(sorted_indices), 0])
        return splits, sorted_indices

    splits_1, indices_1 = sort_by_lowest_cost(indices_1)
    splits_2, indices_2 = sort_by_lowest_cost(indices_2)
    splits_3, indices_3 = sort_by_lowest_cost(indices_3)

    hough_space_1_temp = curve_fitting(hough_space_1, splits_1, indices_1)
    hough_space_2_temp = curve_fitting(hough_space_2, splits_2, indices_2)
    hough_space_3_temp = curve_fitting(hough_space_3, splits_3, indices_3)

    #hough_space_1_temp[indices_1[:, 0], indices_1[:, 1]] = 2
    #hough_space_2_temp[indices_2[:, 0], indices_2[:, 1]] = 2
    #hough_space_3_temp[indices_3[:, 0], indices_3[:, 1]] = 2


    # plot the hough spaces in a 2x3 grid
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(hough_space_1, cmap='jet')
    ax[0, 1].imshow(hough_space_2, cmap='jet')
    ax[0, 2].imshow(hough_space_3, cmap='jet')
    ax[1, 0].imshow(hough_space_1_temp, cmap='jet')
    ax[1, 1].imshow(hough_space_2_temp, cmap='jet')
    ax[1, 2].imshow(hough_space_3_temp, cmap='jet')
    ax[0, 0].set_title(str(imgs_indices[0]) + ".png")
    ax[0, 1].set_title(str(imgs_indices[1]) + ".png")
    ax[0, 2].set_title(str(imgs_indices[2]) + ".png")
    plt.show()

  if False:
      # test 3 random pictures from the dir
      imgs_indices = np.random.randint(120, 980, 3)
      #dir = os.path.join("..", "scratch", "vonroi_wulsd")
      dir = os.path.join("..", "scratch", "copper_bg_100mm_10mm", "rows")
      imgs = [cv2.imread(os.path.join(dir, str(i) + ".png"), 0) for i in imgs_indices]

      hough_space_1 = calculate_hough_space(imgs[0])
      hough_space_2 = calculate_hough_space(imgs[1])
      hough_space_3 = calculate_hough_space(imgs[2])

      hough_space_processed_1 = post_process_hough_space(hough_space_1)
      hough_space_processed_2 = post_process_hough_space(hough_space_2)
      hough_space_processed_3 = post_process_hough_space(hough_space_3)

      coords1 = np.argwhere(hough_space_processed_1 != 0)
      coords2 = np.argwhere(hough_space_processed_2 != 0)
      coords3 = np.argwhere(hough_space_processed_3 != 0)

      hough_space_processed_1_smooth = smooth_hough_space(hough_space_processed_1, coords1)
      hough_space_processed_2_smooth = smooth_hough_space(hough_space_processed_2, coords2)
      hough_space_processed_3_smooth = smooth_hough_space(hough_space_processed_3, coords3)

      visualized_1 = np.zeros((hough_space_1.shape[0], hough_space_1.shape[1], 3))
      visualized_2 = np.zeros((hough_space_2.shape[0], hough_space_2.shape[1], 3))
      visualized_3 = np.zeros((hough_space_3.shape[0], hough_space_3.shape[1], 3))

      # normalize the hough spaces
      hough_space_1 = (hough_space_1 - np.min(hough_space_1)) / (np.max(hough_space_1) - np.min(hough_space_1))
      hough_space_2 = (hough_space_2 - np.min(hough_space_2)) / (np.max(hough_space_2) - np.min(hough_space_2))
      hough_space_3 = (hough_space_3 - np.min(hough_space_3)) / (np.max(hough_space_3) - np.min(hough_space_3))


      visualized_1[:, :, 0] = hough_space_1
      visualized_2[:, :, 0] = hough_space_2
      visualized_3[:, :, 0] = hough_space_3

      visualized_1[:, :, 1] = hough_space_processed_1_smooth
      visualized_2[:, :, 1] = hough_space_processed_2_smooth
      visualized_3[:, :, 1] = hough_space_processed_3_smooth

      visualized_1[:, :, 2] = hough_space_processed_1
      visualized_2[:, :, 2] = hough_space_processed_2
      visualized_3[:, :, 2] = hough_space_processed_3

      plt.imshow(visualized_1)
      plt.show()
      plt.imshow(visualized_2)
      plt.show()
      plt.imshow(visualized_3)
      plt.show()


      fig, ax = plt.subplots(3, 3)
      #ax[0, 0].imshow(hough_space_1, cmap='jet')
      #ax[0, 1].imshow(hough_space_2, cmap='jet')
      #ax[0, 2].imshow(hough_space_3, cmap='jet')
      #ax[0, 0].set_title(str(imgs_indices[0]) + ".png")
      #ax[0, 1].set_title(str(imgs_indices[1]) + ".png")
      #ax[0, 2].set_title(str(imgs_indices[2]) + ".png")
      #ax[1, 0].imshow(hough_space_processed_1, cmap='jet')
      #ax[1, 1].imshow(hough_space_processed_2, cmap='jet')
      #ax[1, 2].imshow(hough_space_processed_3, cmap='jet')

      #plt.show()
    


  if True:
    #dir = os.path.join("..", "scratch", "vonroi_wulsd")
    dir = os.path.join("..", "scratch", "copper_bg_100mm_10mm", "rows")
    #dir = os.path.join("..", "rendered_scratch", "1440_persp_100mm_10mm", "rows")
    #full_hough_to_ply(dir)

    cloud = o3d.io.read_point_cloud("point_cloud_test.ply")
    #cloud = o3d.io.read_point_cloud("point_cloud_1080_bam.ply")
    #cloud = o3d.io.read_point_cloud("point_cloud_perspective_100mm.ply")
    # normalize the colors
    colors = np.asarray(cloud.colors)
    max = np.max(colors[:, 2])
    min = np.min(colors[:, 2])
    colors[:, 2] = (colors[:, 2] - min) / (max - min)

    points = o3d.utility.Vector3dVector(cloud.points)

    # statistical outlier removal
    cl, ind = cloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=0.9)


    # go through all points in the cloud, and if its not in ind paint it green
    colors = np.asarray(cloud.colors)
    temp = colors.copy()
    colors[:] = [0, 1, 0]
    colors[ind] = temp[ind]
    cloud.colors = o3d.utility.Vector3dVector(colors)

    


    o3d.visualization.draw_geometries([cloud], window_name="With Outliers (green)")

    o3d.visualization.draw_geometries([cl], window_name="Outliers removed")



    # use colormap to color the point cloud
    #cmap = plt.get_cmap('inferno')
    #colors = cmap(colors[:, 2])[:, :3]
    #cloud.colors = o3d.utility.Vector3dVector(colors)
    #pcd = o3d.visualization.draw_geometries([cl])

if __name__ == "__main__":
  main()