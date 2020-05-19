## Part 1

# part a)
def sum_of_cubes(n):
    """Return the sum (1^3 + 2^3 + 3^3 + ... + n^3)
    
    Precondition: n > 0, type(n) == int
    
    >>> sum_of_cubes(3)
    36
    >>> sum_of_cubes(1)
    1
    """

    # Validate user input
    if not isinstance(n, int) or not (n > 0):
        print("Invalid input")
        return -1

    # Compute sum  
    sum = 0
    for i in range(1,n+1):
        sum += i**3

    return sum

# Test cases for sum_of_cubes()
assert(sum_of_cubes(3) == 36)
assert(sum_of_cubes(1) == 1)
#assert(sum_of_cubes("Mit") == -1)
#assert(sum_of_cubes(-9) == -1)

# part b)
def word_lengths(sentence):
    """Return a list containing the length of each word in
    sentence.
    
    >>> word_lengths("welcome to APS360!")
    [7, 2, 7]
    >>> word_lengths("machine learning is so cool")
    [7, 8, 2, 2, 4]
    """

    # Split the input string into an array
    word_list = sentence.split(" ")

    # Iterate over the list of words, and replace each index with the length of the corresponding word
    for i in range(len(word_list)):
        word_list[i] = len(word_list[i])

    return word_list      

# Test cases
assert(word_lengths("welcome to APS360!") == [7, 2, 7])
assert(word_lengths("machine learning is so cool") == [7, 8, 2, 2, 4])

# part c)
def all_same_length(sentence):
    """Return True if every word in sentence has the same
    length, and False otherwise.
    
    >>> all_same_length("all same length")
    False
    >>> all_same_length("hello world")
    True
    """

    # Get list with the letter counts of each word in the sentence
    word_count = word_lengths(sentence)

    # Create a list of all the unique elements in the word_count list
    unique_ele = list(set(word_count))

    if len(unique_ele) == 1:
        return True
    else:
        return False

# Test cases
assert(all_same_length("all same length") == False)
assert(all_same_length("hello world") == True)

## Part 2
import numpy as np

# part b)
matrix = np.array([[1., 2., 3., 0.5],
                   [4., 5., 0., 0.],
                   [-1., -2., 1., 1.]])

vector = np.array([2., 0., 1., -2.])

def matrix_multiplication(m, v):
    '''
    param: m <np.array>: Matrix (nxm)
    param: v <np.array>: Vector (mx1)
    return: <np.array> : Matrix (nx1)
    NOTE: Assuming that the vector will be a nx1 vector for this excercise.
    Matrix multiplication requires doing the dot product. 
    '''

    result = list()

    def dot_product(row, col):
        '''
        param: row <arr> - Given a row represented as an arry of size n
        param: col <arr> - Given a col represented as an array of size n
        return: sum <int> - Dot product of two n sized vectors
        '''
        sum = 0

        for i in range(len(row)):
            sum += row[i]*col[i]

        return sum
    

    for i in range(len(m)):
        result.append(dot_product(m[i], v))

    return np.array(result)

output = matrix_multiplication(matrix, vector)

# Test cases
assert(output.shape == (3,)) # Output should be 3x1
assert(output.size == 3) # Output should be a single column with 3 rows

## Part 3

import matplotlib.pyplot as plt

img = plt.imread("https://drive.google.com/uc?export=view&id=1oaLVR2hr1_qzpKQ47i9rVUIklwbDcews")

# part c
img_add = img + 0.25 # Add constant to all the elements
img_add = np.clip(img_add, 0, 1) # Clip all the elements

# part d
img_cropped = img_add[:151, 65:131, :3] # Crop and discard the alpha channel
#plt.imshow(img_cropped)
#plt.show()

## Part 5

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt # for plotting
import torch.optim as optim

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
