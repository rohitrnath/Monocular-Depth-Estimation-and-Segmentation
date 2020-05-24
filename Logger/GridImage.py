import torchvision
import matplotlib.pyplot as plt

def show(tesonrs, figsize=(10,10), *args, **kwargs):
  grid_tensor = torchvision.utils.make_grid( tesonrs[:10], *args, **kwargs)
  grid_image  = grid_tensor.permute(1,2,0)
  plt.figure(figsize=figsize)
  plt.imshow(grid_image)
  plt.xticks([])
  plt.yticks([])
  plt.show()