import numpy as np
import cv2
import matplotlib.pyplot as plt
import math



def main():
  # load EPI image
  img = cv2.imread('rendered/orthographic.png',0)

  ## apply hough transform
  #lines = cv2.HoughLines(img,1,np.pi/180,200)
#
  ## draw lines
  #for rho,theta in lines[0]:
  #  a = np.cos(theta)
  #  b = np.sin(theta)
  #  x0 = a*rho
  #  y0 = b*rho
  #  x1 = int(x0 + 1000*(-b))
  #  y1 = int(y0 + 1000*(a))
  #  x2 = int(x0 - 1000*(-b))
  #  y2 = int(y0 - 1000*(a))
  #  cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

  # display result
  plt.plot(img)
  plt.show()

if __name__ == "__main__":
  main()