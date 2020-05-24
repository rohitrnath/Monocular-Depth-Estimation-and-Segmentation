from __future__ import print_function
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import copy
from torchsummary import summary
from torchvision import datasets, transforms
import numpy as np

def train_model(model, device, train_loader, optimizer, scheduler, epoch,train_losses,train_acc,criteria, store_mode ='epoch', doL1 = 0,doL2 = 0,LAMBDA = 0):
  print('L1=',doL1,';L2=',doL2,';LAMBDA=',LAMBDA,'epoch=',epoch)
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)
    #print('data=',len(data),';target=',len(target))

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    #print('y_pred=',len(y_pred.dataset),'target=',len(target.dataset))
    #loss = F.nll_loss(y_pred, target)
    #criteria = nn.CrossEntropyLoss()
    loss = criteria(y_pred, target) 
    reg_loss=0
    if (doL1 == 1):
      for p in model.parameters():  
        reg_loss += torch.sum(torch.abs(p.data))
    if (doL2 == 1):
      for p in model.parameters():
        reg_loss += torch.sum(p.data.pow(2))    
    
    loss+=LAMBDA*reg_loss
    
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    #One Cyclec LR step
    scheduler.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    if store_mode == 'mini_batch':  # Store loss and accuracy
        batch_accuracy = 100 * correct / processed
        if not train_losses is None:
            train_losses.append(loss.item())
        if not train_acc is None:
            train_acc.append(batch_accuracy)
        
    if store_mode == 'epoch':   # Store loss and accuracy
        accuracy = 100 * correct / processed
        if not train_losses is None:
            train_losses.append(loss.item())
        if not accuracies is None:
            train_acc.append(accuracy)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    #train_acc.append(100*correct/processed)
    

def test_model(model, device, test_loader,test_losses,test_acc,criteria, correct_samples, incorrect_samples, sample_count=30, last_epoch=False):
    model.eval()
    test_loss = 0
    correct = 0
    #criteria = nn.CrossEntropyLoss()
            
    with torch.no_grad():
        for data, target in test_loader:
            img_batch = data
            data, target = data.to(device), target.to(device)
            #print('data=',len(data),';target=',len(target))
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            #test_loss += criteria(output, target).item()
            test_loss += criteria(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            result = pred.eq(target.view_as(pred))
            if last_epoch:
                #print('last_epoch=',last_epoch)
                for i in range(len(list(result))):
                    if not list(result)[i] and len(incorrect_samples) < sample_count:
                        incorrect_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': img_batch[i]
                            
                        })
                    elif list(result)[i] and len(correct_samples) < sample_count:
                        correct_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': img_batch[i]
                            
                        })
            correct += result.sum().item()
            #correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 
        100. * correct / len(test_loader.dataset)))
    test_acc.append(100. * correct / len(test_loader.dataset)) 
    return test_loss
#Global functions
def show_summary(model,input_size = (1, 28, 28)):
    summary(model.m_model, input_size)
    
def run_model(model, device, criteria = F.nll_loss, doL1 = 0, doL2 = 0, LAMBDA = 0, EPOCHS = 40,start=0):
    #scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(EPOCHS):
        print("EPOCH:", (start+epoch))
        train_model(model.m_model, device, model.m_train_loader, model.m_optimizer, epoch,model.m_train_losses,model.m_train_acc,criteria,doL1,doL2,LAMBDA)
        test_model(model.m_model, device, model.m_test_loader,model.m_test_losses,model.m_test_acc,criteria)

def run_model_with_entropy(model, device, criteria = nn.CrossEntropyLoss(), doL1 = 0, doL2 = 0, LAMBDA = 0, EPOCHS = 40,start=0):
    #scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(EPOCHS):
        print("EPOCH:", (start+epoch))
        train_model(model.m_model, device, model.m_train_loader, model.m_optimizer, model.m_scheduler, epoch,model.m_train_losses,model.m_train_acc,criteria,doL1,doL2,LAMBDA)
        #model.m_scheduler.step()
        last_epoch = False
        if(epoch == (EPOCHS-1)):
            last_epoch = True
        
        test_loss = test_model(model.m_model, device, model.m_test_loader, model.m_test_losses, model.m_test_acc, model.m_criterion, model.m_correct_samples, model.m_incorrect_samples, 30, last_epoch)
        
        #model.m_scheduler.step(test_loss) #Used for LR Plateou
        
def run_model_with_entropy_A11(model, device, criteria = nn.CrossEntropyLoss(), doL1 = 0, doL2 = 0, LAMBDA = 0, EPOCHS = 40,start=0):
    #scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
    for epoch in range(EPOCHS):
        print("EPOCH:", (start+epoch))
        print('\nOneCyclicLR:  steps: {}/{}, LR: {:.4f}, Momentum: {:.4f}%\n'.format(model.m_scheduler.last_step,
        model.m_scheduler.num_steps, model.m_scheduler.get_lr(), model.m_scheduler.get_momentum() ))
        train_model(model.m_model, device, model.m_train_loader, model.m_optimizer, model.m_scheduler, epoch,model.m_train_losses,model.m_train_acc,criteria,doL1,doL2,LAMBDA)
        
        last_epoch = False
        if(epoch == (EPOCHS-1)):
            last_epoch = True
        
        test_loss = test_model(model.m_model, device, model.m_test_loader, model.m_test_losses, model.m_test_acc, model.m_criterion, model.m_correct_samples, model.m_incorrect_samples, 30, last_epoch)

import matplotlib.pyplot as plt
def draw_accuracy_graph(model,metric,single_plot= True):
    #print('train_losses=',len(train_losses))
    #print('test_losses=',len(test_losses))
    if(single_plot == True):
        fig = plt.figure(figsize=(12, 6))
        plt.plot(model.m_train_acc,color='blue',label='Training Accuracy')
        plt.plot(model.m_test_acc,color='green',label='Test Accuracy')
        #plt.set_title("Training and validation accuracy")
        plt.legend(loc="center")
        plt.title(f'{metric}')
        # Label axes
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        return

def draw_accuracy_loss_change_graps(model_0,model_l1,model_l2,model_l1_l2, single_plot= True):
    fig, axs = plt.subplots(2,2,figsize=(30,20))
    #print('train_losses=',len(train_losses))
    #print('test_losses=',len(test_losses))
    if(single_plot == True):
        fig = plt.figure(figsize=(12, 6))
        plt.plot(model_l1_l2.m_train_acc,color='blue',label='Both L1 and L2 Regularization')
        plt.plot(model_l1_l2.m_test_acc,color='green',label='Both L1 and L2 Regularization')
        plt.set_title("Training and validation accuracy")
        plt.legend(loc="center")
        return
    
    axs[0,0].plot(model_0.m_test_losses,color='black',label='No Regularization')
    axs[0,0].plot(model_l1.m_test_losses,color='red',label='L1 Regularization')
    axs[0,0].plot(model_l2.m_test_losses,color='blue',label='L2 Regularization')
    axs[0,0].plot(model_l1_l2.m_test_losses,color='green',label='Both L1 and L2 Regularization')
    axs[0,0].set_title("Validation Loss Change")
    axs[0,0].legend(loc="center")

    axs[0,1].plot(model_0.m_test_acc,color='black',label='No Regularization')
    axs[0,1].plot(model_l1.m_test_acc,color='red',label='L1 Regularization')
    axs[0,1].plot(model_l2.m_test_acc,color='blue',label='L2 Regularization')
    axs[0,1].plot(model_l1_l2.m_test_acc,color='green',label='Both L1 and L2 Regularization')
    axs[0,1].set_title("Validation Accuracy Change")
    axs[0,1].legend(loc="center")

    axs[1,0].plot(model_0.m_train_losses,color='black',label='No Regularization')
    axs[1,0].plot(model_l1.m_train_losses,color='red',label='L1 Regularization')
    axs[1,0].plot(model_l2.m_train_losses,color='blue',label='L2 Regularization')
    axs[1,0].plot(model_l1_l2.m_train_losses,color='green',label='Both L1 and L2 Regularization')
    axs[1,0].set_title("Training Loss Change")
    axs[1,0].legend(loc="center")

    axs[1,1].plot(model_0.m_train_acc,color='black',label='No Regularization')
    axs[1,1].plot(model_l1.m_train_acc,color='red',label='L1 Regularization')
    axs[1,1].plot(model_l2.m_train_acc,color='blue',label='L2 Regularization')
    axs[1,1].plot(model_l1_l2.m_train_acc,color='green',label='Both L1 and L2 Regularization')
    axs[1,1].set_title("Training Accuracy Change")
    axs[1,1].legend(loc="center")

def unnormalize(image, mean, std, out_type='array'):
    """Un-normalize a given image.
    
    Args:
        image: A 3-D ndarray or 3-D tensor.
            If tensor, it should be in CPU.
        mean: Mean value. It can be a single value or
            a tuple with 3 values (one for each channel).
        std: Standard deviation value. It can be a single value or
            a tuple with 3 values (one for each channel).
        out_type: Out type of the normalized image.
            If `array` then ndarray is returned else if
            `tensor` then torch tensor is returned.
    """

    if type(image) == torch.Tensor:
        image = np.transpose(image.clone().numpy(), (1, 2, 0))
    
    normal_image = image * std + mean
    if out_type == 'tensor':
        return torch.Tensor(np.transpose(normal_image, (2, 0, 1)))
    elif out_type == 'array':
        return normal_image
    return None  # No valid value given


def to_numpy(tensor):
    """Convert 3-D torch tensor to a 3-D numpy array.
    Args:
        tensor: Tensor to be converted.
    """
    return np.transpose(tensor.clone().numpy(), (1, 2, 0))


def to_tensor(ndarray):
    """Convert 3-D numpy array to 3-D torch tensor.
    Args:
        ndarray: Array to be converted.
    """
    return torch.Tensor(np.transpose(ndarray, (2, 0, 1)))

def plot_accuracy_loss_graphs(data, metric):
    """Plot accuracy graph or loss graph.
    Args:
        data (list or dict): If only single plot then this is a list, else
            for multiple plots this is a dict with keys containing.
            the plot name and values being a list of points to plot
        metric (str): Metric name which is to be plotted. Can be either
            loss or accuracy.
    """

    single_plot = True
    if type(data) == dict:
        single_plot = False
    
    # Initialize a figure
    fig = plt.figure(figsize=(7, 5))

    # Plot data
    if single_plot:
        plt.plot(data)
    else:
        plots = []
        for value in data.values():
            plots.append(plt.plot(value)[0])

    # Set plot title
    plt.title(f'{metric} Change')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    if not single_plot: # Set legend
        location = 'upper' if metric == 'Loss' else 'lower'
        plt.legend(
            tuple(plots), tuple(data.keys()),
            loc=f'{location} right',
            shadow=True,
            prop={'size': 15}
        )

    # Save plot
    fig.savefig(f'{metric.lower()}_change.png')

def plot_predictions(data, classes, plot_title, plot_path):
    """Display data.
    Args:
        data (list): List of images, model predictions and ground truths.
            Images should be numpy arrays.
        classes (list or tuple): List of classes in the dataset.
        plot_title (str): Title for the plot.
        plot_path (str): Complete path for saving the plot.
    """

    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.suptitle(plot_title)

    for idx, result in enumerate(data):

        # If 25 samples have been stored, break out of loop
        if idx > 24:
            break
        
        label = result['label'].item()
        prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
        axs[row_count][idx % 5].imshow(result['image'])
    
    # Set spacing
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    # Save image
    fig.savefig(f'{plot_path}', bbox_inches='tight')

def save_and_show_result(classes, correct_pred=None, incorrect_pred=None, path=None):
    """Display network predictions.
    Args:
        classes (list or tuple): List of classes in the dataset.
        correct_pred (list, optional): Contains correct model predictions and labels.
            (default: None)
        incorrect_pred (list, optional): Contains incorrect model predictions and labels.
            (default: None)
        path (str, optional): Path where the results will be saved.
            (default: None)
    """

    # Create directories for saving predictions
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'predictions'
        )
    if not os.path.exists(path):
        os.makedirs(path)
    
    if not correct_pred is None:  # Plot correct predicitons
        plot_predictions(
            correct_pred, classes, 'Correct Predictions', f'{path}/correct_predictions.png'
        )

    if not incorrect_pred is None:  # Plot incorrect predicitons
        plot_predictions(
            incorrect_pred, classes, '\nIncorrect Predictions', f'{path}/incorrect_predictions.png'
        )