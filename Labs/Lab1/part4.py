import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt # for plotting
import torch.optim as optim
import numpy as np

torch.manual_seed(1) # set the random seed

# define a 2-layer artificial neural network
class Pigeon(nn.Module):
    def __init__(self):
        super(Pigeon, self).__init__()
        self.layer1 = nn.Linear(28 * 28, 30)
        self.layer2 = nn.Linear(30, 1)
    def forward(self, img):
        flattened = img.view(-1, 28 * 28) 
        activation1 = self.layer1(flattened)
        activation1 = F.relu(activation1)
        activation2 = self.layer2(activation1)
        return activation2

pigeon = Pigeon()

# load the data
mnist_data = datasets.MNIST('data', train=True, download=True)
mnist_data = list(mnist_data)
mnist_train = mnist_data[:1000]
mnist_val   = mnist_data[1000:2000]
img_to_tensor = transforms.ToTensor()
      
    
# simplified training code to train `pigeon` on the "small digit recognition" task
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(pigeon.parameters(), lr=0.005, momentum=0.9) 

'''
Parameters that we can adjust to manipulate training and testing data:
- Batch size: Make adjustments to weights after a certain batch has run through
- Learning rate, momentum
- Size of network: I'm going to ignore this one for now because I think changing the ANN's architecture is 
    outside the scope for this problem.
- Activation function
- Number of iterations
- Changing the optimizer perhaps (ex. Adam Optimizer)
'''

'''
#NOTE: Uncomment block to get the original problem statement

for (image, label) in mnist_train:
    # actual ground truth: is the digit less than 3?
    actual = torch.tensor(label < 3).reshape([1,1]).type(torch.FloatTensor)
    # pigeon prediction
    out = pigeon(img_to_tensor(image)) # step 1-2
    # update the parameters based on the loss
    loss = criterion(out, actual)      # step 3
    loss.backward()                    # step 4 (compute the updates for each parameter)
    optimizer.step()                   # step 4 (make the updates for each parameter)
    optimizer.zero_grad()              # a clean up step for PyTorch

# computing the error and accuracy on the training set
error = 0
for (image, label) in mnist_train:
    prob = torch.sigmoid(pigeon(img_to_tensor(image)))
    if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
        error += 1
print("Training Error Rate:", error/len(mnist_train))
print("Training Accuracy:", 1 - error/len(mnist_train))


# computing the error and accuracy on a test set
error = 0
for (image, label) in mnist_val:
    prob = torch.sigmoid(pigeon(img_to_tensor(image)))
    if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
        error += 1
print("Test Error Rate:", error/len(mnist_val))
print("Test Accuracy:", 1 - error/len(mnist_val))
'''

'''
#NOTE: Uncomment code block to see learning rate adjustment results.

# Test out different learning rate values
lr_increment = np.arange(0,0.1, 0.005).tolist() # Create a list of learning rate values
lr_training_val, lr_testing_val = list(), list() # Data structures to store the accuracy of our training and testing based on learning rate.

for lr in lr_increment:
    optimizer = optim.SGD(pigeon.parameters(), lr=lr, momentum=0.9) # Redefine the optimizer function with the incremented learning rate
    for (image, label) in mnist_train:
        # actual ground truth: is the digit less than 3?
        actual = torch.tensor(label < 3).reshape([1,1]).type(torch.FloatTensor)
        # pigeon prediction
        out = pigeon(img_to_tensor(image)) # step 1-2
        # update the parameters based on the loss
        loss = criterion(out, actual)      # step 3
        loss.backward()                    # step 4 (compute the updates for each parameter)
        optimizer.step()                   # step 4 (make the updates for each parameter)
        optimizer.zero_grad()              # a clean up step for PyTorch

    # computing the error and accuracy on the training set
    error = 0
    for (image, label) in mnist_train:
        prob = torch.sigmoid(pigeon(img_to_tensor(image))) 
        if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
            error += 1

    lr_training_val.append(1 - error/len(mnist_train))

    # computing the error and accuracy on a test set
    error = 0
    for (image, label) in mnist_val:
        prob = torch.sigmoid(pigeon(img_to_tensor(image)))
        if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
            error += 1

    lr_testing_val.append(1 - error/len(mnist_val))

# Plot the results of learning rate

plt.plot(lr_increment, lr_testing_val, "-b", label="Testing data accuracy")
plt.plot(lr_increment, lr_training_val, "-r", label="Training data accuracy")
plt.legend(loc="lower right")
plt.xlabel("Learning rate")
plt.ylabel("Prediction accuracy")

plt.show()
'''

'''
#NOTE: Uncomment code block to see momentum rate adjustment results.

# Test out different learning rate values
momentum_increment = np.arange(0,1, 0.05).tolist() # Create a list of momentum values
momentum_training_val, momentum_testing_val = list(), list() # Data structures to store the accuracy of our training and testing based on momentum.

for momentum in momentum_increment:
    optimizer = optim.SGD(pigeon.parameters(), lr=0.005, momentum=momentum) # Redefine the optimizer function with the incremented momentum
    for (image, label) in mnist_train:
        # actual ground truth: is the digit less than 3?
        actual = torch.tensor(label < 3).reshape([1,1]).type(torch.FloatTensor)
        # pigeon prediction
        out = pigeon(img_to_tensor(image)) # step 1-2
        # update the parameters based on the loss
        loss = criterion(out, actual)      # step 3
        loss.backward()                    # step 4 (compute the updates for each parameter)
        optimizer.step()                   # step 4 (make the updates for each parameter)
        optimizer.zero_grad()              # a clean up step for PyTorch

    # computing the error and accuracy on the training set
    error = 0
    for (image, label) in mnist_train:
        prob = torch.sigmoid(pigeon(img_to_tensor(image))) 
        if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
            error += 1

    momentum_training_val.append(1 - error/len(mnist_train))

    # computing the error and accuracy on a test set
    error = 0
    for (image, label) in mnist_val:
        prob = torch.sigmoid(pigeon(img_to_tensor(image)))
        if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
            error += 1

    momentum_testing_val.append(1 - error/len(mnist_val))

# Plot the results of learning rate

plt.plot(momentum_increment, momentum_testing_val, "-b", label="Testing data accuracy")
plt.plot(momentum_increment, momentum_training_val, "-r", label="Training data accuracy")
plt.legend(loc="lower right")
plt.xlabel("Momentum")
plt.ylabel("Prediction accuracy")

plt.show()
'''

'''
#NOTE: Uncomment code block to see results of batch size changes.

def create_sublist(l, increment):
    #param: l <list> : This is the list of training data
    #param: increment <int> : This is a valid factor of the length of the list to create a subset of.
    #return: result <list> : List contains sublists of the data split into specific batch sizes.
    increment_list = np.arange(0, len(l), increment)
    result = list()
    prev_i = 0

    for i in increment_list:
        result.append(l[prev_i:i])
        prev_i = i

    return result
    
factors = [1,2,4,5,8,10,20,25,40,50,100,125,200,250,500,1000] # Factors of 1000

batch_train_accur, batch_test_accur = list(), list()

for factor in factors:
    for train_data in create_sublist(mnist_train, factor): # Create a sublist of the training data based on the factors - used to determing batch size

        count = 1 # Keep track of the number of iterations 

        for (image, label) in mnist_train:
            if count == len(mnist_train):
                # actual ground truth: is the digit less than 3?
                actual = torch.tensor(label < 3).reshape([1,1]).type(torch.FloatTensor)
                # pigeon prediction
                out = pigeon(img_to_tensor(image)) # step 1-2
                # update the parameters based on the loss
                loss = criterion(out, actual)      # step 3
                loss.backward()                    # step 4 (compute the updates for each parameter)
                optimizer.step()                   # step 4 (make the updates for each parameter)
                optimizer.zero_grad()              # a clean up step for PyTorch

                break
            else:
                count += 1

    # computing the error and accuracy on the training set
    error = 0
    for (image, label) in mnist_train:
        prob = torch.sigmoid(pigeon(img_to_tensor(image))) 
        if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
            error += 1
    batch_train_accur.append(1 - error/len(mnist_train))

    # computing the error and accuracy on a test set
    error = 0
    for (image, label) in mnist_val:
        prob = torch.sigmoid(pigeon(img_to_tensor(image)))
        if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
            error += 1
    batch_test_accur.append((1 - error/len(mnist_val)))

# Plot the results of batch rate

# Plot the results of learning rate

print(batch_test_accur)
print(batch_train_accur)

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(factors, batch_test_accur, color='r')
ax.scatter(factors, batch_train_accur, color='b')
ax.set_xlabel('Batch divisions')
ax.set_ylabel('Prediction accuracy')
plt.show()
'''

#NOTE: Uncomment block to get Adam optimizer results
optimizer = optim.Adam(pigeon.parameters(), lr=0.005) 

for (image, label) in mnist_train:
    # actual ground truth: is the digit less than 3?
    actual = torch.tensor(label < 3).reshape([1,1]).type(torch.FloatTensor)
    # pigeon prediction
    out = pigeon(img_to_tensor(image)) # step 1-2
    # update the parameters based on the loss
    loss = criterion(out, actual)      # step 3
    loss.backward()                    # step 4 (compute the updates for each parameter)
    optimizer.step()                   # step 4 (make the updates for each parameter)
    optimizer.zero_grad()              # a clean up step for PyTorch

# computing the error and accuracy on the training set
error = 0
for (image, label) in mnist_train:
    prob = torch.sigmoid(pigeon(img_to_tensor(image)))
    if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
        error += 1
print("Adam Training Error Rate:", error/len(mnist_train))
print("Adam Training Accuracy:", 1 - error/len(mnist_train))


# computing the error and accuracy on a test set
error = 0
for (image, label) in mnist_val:
    prob = torch.sigmoid(pigeon(img_to_tensor(image)))
    if (prob < 0.5 and label < 3) or (prob >= 0.5 and label >= 3):
        error += 1
print("Adam Test Error Rate:", error/len(mnist_val))
print("Adam Test Accuracy:", 1 - error/len(mnist_val))