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