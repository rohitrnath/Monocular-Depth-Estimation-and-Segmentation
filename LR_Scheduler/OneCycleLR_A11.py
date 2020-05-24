from torch.optim import Optimizer


class OneCycleLR_A11:
    """ This is Custom OneCycleLR code supports EVA assignment 11.
    Based on the requirement:
    Total Epochs = 24
    Max at Epoch = 5
    LRMIN = LRMAX /10
    LRMAX = found with LRFinder rang-test
    NO Annihilation
    
    Args:
        optimizer:             (Optimizer) against which we apply this scheduler
        batch_size:             (int) batchSize used
        num_epochs:             (int) Number of epochs
        num_steps:             (int) of total number of steps/iterations
        lr_range:              (tuple) of min and max values of learning rate
        momentum_range:        (tuple) of min and max values of momentum
    """

    def __init__(self,
                 optimizer: Optimizer,
                 batch_size: int,
                 num_epochs: int,
                 lr_range: tuple = (0.1, 1.),
                 momentum_range: tuple = (0.85, 0.95),):
        # Sanity check
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer  = optimizer
        
        self.epochs     = num_epochs
        
        # Total number of images in Cifar10 = 50000
        #iterations per epoch
        self.iteration = 50000/batch_size
        #Total number of iterations/steps through out the training
        self.num_steps = num_epochs * self.iteration
        
        #Number of steps at 5th epoch
        self.steps_fifth_epoch = 5 * self.iteration

        self.min_lr, self.max_lr = lr_range[0], lr_range[1]
        assert self.min_lr < self.max_lr, \
            "Argument lr_range must be (min_lr, max_lr), where min_lr < max_lr"

        self.min_momentum, self.max_momentum = momentum_range[0], momentum_range[1]
        assert self.min_momentum < self.max_momentum, \
            "Argument momentum_range must be (min_momentum, max_momentum), where min_momentum < max_momentum"

        self.num_cycle_steps = self.num_steps # Total number of steps in the cycle

        self.last_step = -1
        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_momentum(self):
        return self.optimizer.param_groups[0]['momentum']

    def step(self):
        """Conducts one step of learning rate and momentum update
        """
        current_step = self.last_step + 1
        self.last_step = current_step

        if current_step <= self.steps_fifth_epoch:
            # Scale up phase
            scale = current_step / (self.steps_fifth_epoch)
            lr = self.min_lr + (self.max_lr - self.min_lr) * scale
            momentum = self.max_momentum - (self.max_momentum - self.min_momentum) * scale
        elif current_step <= self.num_cycle_steps:
            # Scale down phase
            scale = (current_step - self.steps_fifth_epoch) / (self.num_cycle_steps - self.steps_fifth_epoch)
            lr = self.max_lr - (self.max_lr - self.min_lr) * scale
            momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * scale
        else:
            # Exceeded given num_steps: do nothing
            return

        self.optimizer.param_groups[0]['lr'] = lr
        if momentum:
            self.optimizer.param_groups[0]['momentum'] = momentum