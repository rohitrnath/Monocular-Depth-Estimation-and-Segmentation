import math
import matplotlib.pyplot as plt
TOTAL_IMAGES = 10000
RATIO        = 0.7
class cyclicLR(object):
    def __init__(self, optimizer, lr_min, lr_max, batch_size, MaxNumCycles, epochs, startEpoch=0, constLR=0.0001, constEpochs= 0):
        self.optimizer = optimizer
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.maxCycles = MaxNumCycles
        self.num_images = TOTAL_IMAGES * RATIO
        self.epochs = epochs
        self.batch_iterations = self.num_images/batch_size
        self.step_size  = (self.epochs) * self.batch_iterations
        self.const_iterations = constEpochs * self.batch_iterations
        self.constLR = constLR
        self.startEpochIter = startEpoch * self.batch_iterations

    def step(self,epoch,i):
        lr_t = self.constLR
        iteration = epoch*self.batch_iterations + i - self.const_iterations - self.startEpochIter
        if(iteration >= 0):
          cycleCount = math.floor(1+(iteration/(2*self.step_size)))
          if cycleCount > self.maxCycles:
            if (iteration // self.step_size) == 0:
              self.constLR = self.constLR/10.0
              lr_t = self.constLR
          else:
            x = abs((iteration/self.step_size) - (2*(cycleCount)) + 1)
            lr_t = self.lr_min + ((self.lr_max - self.lr_min)*(1.0 - x))
          #lr_t = self.lr_min + ((self.lr_max - self.lr_min)*(x))
        else:
          lr_t = self.constLR

        self.optimizer.param_groups[0]['lr'] = lr_t
        
        return lr_t