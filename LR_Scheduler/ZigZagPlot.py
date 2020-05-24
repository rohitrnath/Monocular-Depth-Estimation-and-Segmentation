import numpy as np
import math
import matplotlib.pyplot as plt

def ZigZagPlot( lr_min, lr_max, batch_size, MaxNumCycles):
  #lr_min,lr_max are in y direction
  #Cifar10 having 50000 images
  #MaxNumCycles = Number of cycles we need to draw

  #For Cifar10
  CIFAR10_num_images = 50000
  batch_iterations = CIFAR10_num_images/batch_size
  step_size  = 10*batch_iterations # any number between(2-10) * iterations
  
  x_points = []
  y_points = []
 
  plt.figure(figsize=(20,10))
  cycleCount = 0
  iteration = 0
  while(cycleCount <= MaxNumCycles):

    cycleCount = math.floor(1+(iteration/(2*step_size)))
    x = abs((iteration/step_size) - (2*(cycleCount)) + 1)
    lr_t = lr_min + ((lr_max - lr_min)*(1.0 - x))

    y_points.append(lr_t)
    x_points.append(iteration)

    iteration = iteration + 1

  plt.plot(x_points, y_points, '-')
  #plt.yscale("log")
  plt.xlabel('Iterations')
  plt.ylabel('Lr Range')
  plt.show()
