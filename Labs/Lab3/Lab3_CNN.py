'''
Owner: Mithun Jothiravi 
Student #: 1002321258
Purpose: Code contains the CNN implementating for Gesture Dataset - Lab 3B
'''

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import os
import torchvision.models

torch.manual_seed(100) # Set the seed to ensure reproducibility of results

########## Environment variables and classes ##########
class HandCNN(nn.Module):
    def __init__(self):
        super(HandCNN, self).__init__()
        self.name = "HandCNN"
        self.conv1 = nn.Conv2d(3, 5, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 5)

        self.fc1 = nn.Linear(10 * 53 * 53, 32)
        self.fc2 = nn.Linear(32, 9)

    def forward(self, img):
        x = self.pool(F.relu(self.conv1(img)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AlexNetClassifier(nn.Module):
    def __init__(self):
        super(AlexNetClassifier, self).__init__()
        self.name= "AlexNetClassifier"
        self.fc1 = nn.Linear(256 * 6 * 6, 10)
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6) #flatten feature data
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

########## Helper functions start here ##########
def load_dataset(batch_size=1, overfit_test=False):
    '''
    param: batch_size <int>
    param: overfit_test <bool> - This for is for question 1. c) of the lab where we want to overfit to the small dataset.
    return: 
        train_loader, val_loader, test_loader <torch.utils.data.dataloader.DataLoader> : These are the results of the  
            torch.DataLoader utility. These can be parsed using the same method as iterable-style Python datatypes.
    '''
    # Filepaths for training/val and holdout test data
    train_val_fp = './data/Lab_3b_Gesture_Dataset_train_val'
    holdout_test_fp = './data/Lab_3b_Gesture_Dataset_holdout_test'
    small_dataset_fp = './data/small_dataset'

    # Transform Settings - From Lab_3b_Sample_Training_Code
    transform = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.ToTensor()])

    # Generate train/val dataset
    if not overfit_test:
        dataset = torchvision.datasets.ImageFolder(
            root=train_val_fp,
            transform=transform
        )
    else:
        dataset = torchvision.datasets.ImageFolder(
            root=small_dataset_fp,
            transform=transform
        )
    dataset_indices = list(range(0,len(dataset)))


    # Split into train and validation
    np.random.seed(500) # Fixed numpy random seed for reproducible shuffling
    np.random.shuffle(dataset_indices) # Randomize the elements if the dataset_indices data structure

    # 80-20 split was recommended between training and test set: https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data
    split = int(len(dataset_indices) * 0.8) 

    # Split dataset into training and validation (val set will be spli further)
    relevant_train_indices, relevant_val_indices = dataset_indices[:split], dataset_indices[split:]  

    # SubsetRandomSampler: Samples elements randomly from a given list of indices, without replacement.
    train_sampler = SubsetRandomSampler(relevant_train_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=0, sampler=train_sampler)
    val_sampler = SubsetRandomSampler(relevant_val_indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=0, sampler=val_sampler)

    if not overfit_test:
        # Load holdout test dataset
        testset = torchvision.datasets.ImageFolder(
            root=holdout_test_fp,
            transform=transform
        )
        # Get the list of indices to sample from
        test_sampler = SubsetRandomSampler(testset)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                num_workers=0, sampler=test_sampler)

        return train_loader, val_loader, test_loader
    else:
        # For the overfit question, there is no need for a holdout test dataloader object.
        return train_loader, val_loader, None

def get_accuracy(model, data_loader, train_alexNet = False, ALNC = None):
    '''
    param: model <class> - This is the NN architecture
    param: data_loader <torch.utils.data> - This is the data loader object
    param: train_alexNet <bool> - AlexNET training requires being fed into a different pipeline.
    param: ALNC <class> - AlexNet without classifier, used is train_alexNet is True
    return: The accuracy results of the model
    Accuracy function was sourced from the Lab_3b_Sample_Training_Code.html handout
    '''
    correct = 0
    total = 0

    for imgs, labels in data_loader:
        
        if train_alexNet:
            output = model(ALNC(imgs))
        else:
            output = model(imgs)

        #select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += imgs.shape[0]

    return correct / total

def get_model_name(name, batch_size, learning_rate, epoch):
    '''
    param: name <str> - Name given by the user for this trial
    param: batch_size <int> - Training parameter specified by the user
    param: learning_rate <float> - Training parameter specified by the user
    param: epoch <int> - Training parameter specified by the user
    return: path <str> -  A string with the hyperparameter name and value concatenated
    Generate a name for the model consisting of all the hyperparameter values
    Function taken from Lab_2_Cats_vs_Dogs
    '''
    path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size, learning_rate, epoch)

    return path

def plot_curves(dataX, dataY, dataX_title, dataY_title, plot_title):
    '''
    param: dataX <list> - List of values to plot on the x-axis
    param: dataY <list> - List of values to plot on the y-axis
    param: dataX_title <str> - Title for the x-axis
    param: dataY_title <str> - Title for the y-axis
    param: plot_title <str> - Title for the entire plot
    return: None

    Given data, this function will generate a plot.
    '''
    plt.title(plot_title)
    plt.plot(dataX, dataY,'ro', label=dataX_title)
    plt.xlabel(dataX_title)
    plt.ylabel(dataY_title)
    plt.legend(loc='best')
    plt.show()
    print("Plot generated for: {} vs. {}".format(dataX_title, dataY_title))

    return

def test_net(model, test_loader, trial_name = None, batch_size=1, num_epochs=1, learn_rate = 0.001, momentum = 0.9, train_alexNet = False, ALNC = None, overfit_test = False, plot=False):
    '''
    param: model <class> - NN architecture specified by the user.
    param: test_loader <torch.utils.data> - This is the data loader object
    param: trial_name <str> - User can specify a trial name to store all the backup log files that are generated
    param: batch_size <int> - Training parameter specified by the user
    param: learning_rate <float> - Training parameter specified by the user
    param: epoch <int> - Training parameter specified by the use
    param: momentum <float> - Training parameter specified by the use
    param: train_alexNet <bool> - AlexNET training requires being fed into a different pipeline.
    param: ALNC <class> - AlexNet without classifier, used is train_alexNet is True
    param: overfit_false <bool> - If we want to perform overfit test as requested in question 1.c), set as True
    param: plot <bool> - If the user requests a plot for the trial.
    return: test_acc <float> - The testing accuracy

    Given a holdout test dataset, and a NN architecture, determine the accuracy of trained model.
    Code was referenced from Lab.
    '''
        # Training with the AlexNet feature pipeline requires different configs
    if train_alexNet:
        torch.manual_seed(333) # set the random seed
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)

        iters, losses, test_acc = [], [], []

        # training
        n = 0 # the number of iterations
        start_time=time.time()
        for epoch in range(num_epochs):
            mini_b=0
            mini_batch_correct = 0
            Mini_batch_total = 0
            for imgs, labels in iter(test_loader):
            
            #### ALNC is alexNet.features (AlexNet without classifier) ####
            
                out = model(ALNC(imgs))             # forward pass
                loss = criterion(out, labels) # compute the total loss
                loss.backward()               # backward pass (compute parameter updates)
                optimizer.step()              # make the updates for each parameter
                optimizer.zero_grad()         # a clean up step for PyTorch

            # save the current training information
                iters.append(n)
                losses.append(float(loss)/batch_size)             # compute *average* loss
                test_acc.append(get_accuracy(model, test_loader, train_alexNet = True, ALNC = ALNC))  # compute validation accuracy
                n += 1
                mini_b += 1
                print("Iteration: ",n,'Progress: % 6.2f ' % ((epoch * len(test_loader) + mini_b) / (num_epochs * len(test_loader))*100),'%', "Time Elapsed: % 6.2f s " % (time.time()-start_time))


            print ("Epoch %d Finished. " % epoch ,"Time per Epoch: % 6.2f s "% ((time.time()-start_time) / (epoch +1)))


        end_time= time.time()
        
        test_acc.append(get_accuracy(model, test_loader, train_alexNet = True, ALNC = ALNC))
        print("Final Testing Accuracy: {}".format(test_acc[-1]))
        print ("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % ( (end_time-start_time), ((end_time-start_time) / num_epochs) ))

        return

    else:

        # If a trial_name is defined, a directory will be created to store all the log files.
        if trial_name and not os.path.exists('./checkpoints/{}'.format(trial_name)):
            os.mkdir('./checkpoints/{}'.format(trial_name))

        torch.manual_seed(444)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

        test_acc = []

        # training
        print ("Training Started...")
        n = 0 # the number of iterations
        for epoch in range(num_epochs):
            for imgs, labels in iter(test_loader):
                
                out = model(imgs)             # forward pass
                loss = criterion(out, labels) # compute the total loss
                loss.backward()               # backward pass (compute parameter updates)
                optimizer.step()              # make the updates for each parameter
                optimizer.zero_grad()         # a clean up step for PyTorch
                n += 1
            
            # track accuracy
            test_acc.append(get_accuracy(model, test_loader))
            print("Epoch: {}, Testing Accuracy: {}".format(epoch, test_acc[-1])) 

            # Backup the checkpoint for this epoch
            model_path = get_model_name(model.name, batch_size, learn_rate, epoch)
            torch.save(model.state_dict(), "./checkpoints/{}/{}".format(trial_name, model_path)) if trial_name else torch.save(model.state_dict(), model_path)
        
        # Plot results if specified
        if plot:
            print('Generating plots for Test accuracy results.')
            plot_curves(list(range(num_epochs)), test_acc, "Epochs", "Testing Accuracy", "Relationship between Epochs and Testing Accuracy") # Plot training curve

        return test_acc

def train_net(model, train_loader, val_loader, trial_name = None, batch_size=1, num_epochs=1, learn_rate = 0.001, momentum= 0.9, train_alexNet = False, ALNC = None, overfit_test = False, plot=False):
    '''
    param: model <class> - NN architecture specified by the user.
    param: train_loader, test_loader <torch.utils.data> - This is the data loader object
    param: trial_name <str> - User can specify a trial name to store all the backup log files that are generated
    param: batch_size <int> - Training parameter specified by the user
    param: learning_rate <float> - Training parameter specified by the user
    param: epoch <int> - Training parameter specified by the use
    param: momentum <float> - Training parameter specified by the use
    param: train_alexNet <bool> - AlexNET training requires being fed into a different pipeline.
    param: ALNC <class> - AlexNet without classifier, used is train_alexNet is True
    param: overfit_false <bool> - If we want to perform overfit test as requested in question 1.c), set as True
    param: plot <bool> - If the user requests a plot for the trial.
    Training function was sourced from the Lab_3b_Sample_Training_Code.html handout.
    '''

    # Training with the AlexNet feature pipeline requires different configs
    if train_alexNet:
        torch.manual_seed(501) # set the random seed
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=momentum)

        iters, losses, train_acc, val_acc = [], [], [], []

        # training
        n = 0 # the number of iterations
        start_time=time.time()
        for epoch in range(num_epochs):
            mini_b=0
            mini_batch_correct = 0
            Mini_batch_total = 0
            for imgs, labels in iter(train_loader):
            
            #### ALNC is alexNet.features (AlexNet without classifier) ####
            
                out = model(ALNC(imgs))             # forward pass
                loss = criterion(out, labels) # compute the total loss
                loss.backward()               # backward pass (compute parameter updates)
                optimizer.step()              # make the updates for each parameter
                optimizer.zero_grad()         # a clean up step for PyTorch



                ##### Mini_batch Accuracy ##### We don't compute accuracy on the whole trainig set in every iteration!
                pred = out.max(1, keepdim=True)[1]
                mini_batch_correct = pred.eq(labels.view_as(pred)).sum().item()
                Mini_batch_total = imgs.shape[0]
                train_acc.append((mini_batch_correct / Mini_batch_total))
            ###########################

            # save the current training information
                iters.append(n)
                losses.append(float(loss)/batch_size)             # compute *average* loss
                val_acc.append(get_accuracy(model, val_loader, train_alexNet = True, ALNC = ALNC))  # compute validation accuracy
                n += 1
                mini_b += 1
                print("Iteration: ",n,'Progress: % 6.2f ' % ((epoch * len(train_loader) + mini_b) / (num_epochs * len(train_loader))*100),'%', "Time Elapsed: % 6.2f s " % (time.time()-start_time))


            print ("Epoch %d Finished. " % epoch ,"Time per Epoch: % 6.2f s "% ((time.time()-start_time) / (epoch +1)))


        end_time= time.time()
        
        train_acc.append(get_accuracy(model, train_loader, train_alexNet = True, ALNC = ALNC))
        print("Final Training Accuracy: {}".format(train_acc[-1]))
        print("Final Validation Accuracy: {}".format(val_acc[-1]))
        print ("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % ( (end_time-start_time), ((end_time-start_time) / num_epochs) ))

        return

    else:

        # If a trial_name is defined, a directory will be created to store all the log files.
        if trial_name and not os.path.exists('./checkpoints/{}'.format(trial_name)):
            os.mkdir('./checkpoints/{}'.format(trial_name))

        torch.manual_seed(1000)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

        train_acc, val_acc = [], []

        # training
        print ("Training Started...")
        n = 0 # the number of iterations
        for epoch in range(num_epochs):
            for imgs, labels in iter(train_loader):
                
                out = model(imgs)             # forward pass
                loss = criterion(out, labels) # compute the total loss
                loss.backward()               # backward pass (compute parameter updates)
                optimizer.step()              # make the updates for each parameter
                optimizer.zero_grad()         # a clean up step for PyTorch
                n += 1
            
            # track accuracy
            train_acc.append(get_accuracy(model, train_loader))
            val_acc.append(get_accuracy(model, val_loader))
            print("Epoch: {}, Training Accuracy: {}, Validation Accuracy: {}".format(epoch, train_acc[-1], val_acc[-1])) if not overfit_test else  print("Epoch: {}, Training Accuracy: {}".format(epoch, train_acc[-1]))

            # Backup the checkpoint for this epoch
            model_path = get_model_name(model.name, batch_size, learn_rate, epoch)
            torch.save(model.state_dict(), "./checkpoints/{}/{}".format(trial_name, model_path)) if trial_name else torch.save(model.state_dict(), model_path)
        
        # Plot results if specified
        if plot:
            print('Generating plots for Training and Validation accuracy results.')
            plot_curves(list(range(num_epochs)), train_acc, "Epochs", "Training Accuracy", "Relationship between Epochs and Training Accuracy") # Plot training curve
            plot_curves(list(range(num_epochs)), val_acc, "Epochs", "Validation Accuracy", "Relationship between Epochs and Validation Accuracy") # Plot validation curve

        return train_acc, val_acc

def main():
    '''
    Main task execution for machine learning pipeline.
    '''

    # Call the CNN and AlexNet architecture class (subset of nn.module)
    hand_net = HandCNN()
    alex_net = AlexNetClassifier()
    alex_net_config = torchvision.models.alexnet(pretrained=True)

    # Load the data for training, validation, and holdout test set
    train_data, val_data, test_data = load_dataset(batch_size=1) # Regular data
    #train_data, val_data, test_data = load_dataset(batch_size=1, overfit_test=True) # Small dataset meant for overfitting - Question 1. c)


    # Train the CNN network
    #train_net(hand_net, train_data, val_data, num_epochs=50, trial_name="overfit_test")
    train_net(hand_net, train_data, val_data, num_epochs=1, learn_rate=0.01, trial_name="plot_test", plot=True)


    # Train the AlexNet network
    #train_net(alex_net, train_data, val_data, num_epochs=2 , trial_name = 'AlexNet_Test', train_alexNet = True, ALNC = alex_net_config.features)

    # Test network with holdout set
    print('Testing network with holdout set.')
    test_net(hand_net, test_data, trial_name = 'TestData_1', plot=True)

    return

main()
