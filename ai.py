import numpy as np
import time
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import math
import matplotlib.pyplot as plt
import scipy.io

#The Node class is used in the KDTree data structure
class Node:
    def __init__(self, row, column, value, image):
        #Label is the classification of the node; square or circle - 0 or 1
        #Only leaf nodes have a classification
        self.label = None
        #Row and column refers to the pixel location for the given depth of the tree
        #i.e. the dimension
        self.row = row
        self.column = column
        #Value is the median value of the pixel among the images
        self.value = value
        #Left and right are the branches for images with pixels less than and greater than the value
        #Leaf nodes do not have branches
        self.left = None
        self.right = None
        #Attach the median image to the node for reference when calculating the Euclidean distance
        self.image = image

class DecisionTree:
    def __init__(self):
        #Classification is square or circle - 0 or 1
        #Only leaf nodes have a classification
        self.classification = None
        #Attribute is the dimension the data is split on
        #For our images, the pixels are the dimensions
        #The attribute represents which pixel is considered when splitting the data
        self.attribute = None
        #The value indicates the threshold of the split of the data
        #Images with pixel values less than (or equal to) the value will be passed to the left branch
        #Images with pixel values greater than the value will be passed to the right branch
        self.value = None
        #Leaf nodes do not have branches
        self.left = None
        self.right = None

#Builds a KD Tree based on a set of training data
def buildKDTree(images, labels, depth):
    #If there are no more images to sort there cannot be a new node
    if len(images) == 0:
        return None
    
    else:
        #Our images are 24x24 pixels
        #Therefore, there are 576 different pixels to compare per image
        #Each pixel is a dimension that we can split data around
        #The depth of the tree determines which pixel (dimension) to split on
        #depth mod dimensions
        testImage = images[0]
        numberRows = len(testImage)
        numberColumns = len(testImage[0])
        row = int(depth / numberColumns)
        column = depth % numberColumns
        #Sort the images, according to the selected pixel
        imagesSorted = sorted(images, key = lambda  pixel: pixel[row, column])
        median = int((len(imagesSorted) - 1) / 2)
        #The median value is the selected pixel in the median image of the list of sorted images
        medianImage = imagesSorted[median]
        medianValue = medianImage[row, column]
        #Create a new node
        node = Node(row, column, medianValue, medianImage)

        #Split the data around the selected pixels
        #Some images & labels are passed to the left branch, while the others go right
        leftImages = []
        leftLabels = []
        rightImages = []
        rightLabels = []

        index = 0;
        pixels = numberRows * numberColumns
        #Run through images so that we can get the index of the labels going left/right
        for image in images:
            #We don't pass on the median image or label
            #So we have to check that the current image isn't the median image
            sameImage = True
            #If all the pixels are the same as in the median image, it's the same image
            for pix in range(pixels):
                rowIndex = int(pix / numberColumns)
                columnIndex = pix % numberColumns
                if image[rowIndex, columnIndex] != medianImage[rowIndex, columnIndex]:
                    sameImage = False

            #If it's the median image we just label the node
            if sameImage == True:
                node.label = labels[index]

            #If the value on current dimension is less than or equal to the median value, the image and label get passed left
            elif image[row, column] <= medianValue:
                leftImages.append(image)
                leftLabels.append(labels[index])

            #If the value on current dimension is greater than the median value, the image and label get passed right
            elif image[row, column] > medianValue:
                rightImages.append(image)
                rightLabels.append(labels[index]) 

            index += 1


        #Build the left and right branches of the node
        node.left = buildKDTree(leftImages, leftLabels, depth + 1)
        node.right = buildKDTree(rightImages, rightLabels, depth + 1)
        return node

#This function drops an image down the KD tree until it reaches a leaf node
#It returns a list of the nodes visited on the image's descent
def drop(image, kDTree):
    nextNode = kDTree
    #The parent nodes visited during the image's descent are stored
    #So that the we can go back up the tree when we look for the image's nearest neighbours
    parentNodes = []
    #Go down the tree until a leaf node is reaches
    while nextNode != None:
        #If the image's value on the given dimension is smaller than the node's (splitting) value
        #Go left
        if image[nextNode.row, nextNode.column] <= nextNode.value:
            parentNodes.insert(0, nextNode)
            nextNode = nextNode.left

        #Go right
        elif image[nextNode.row, nextNode.column] > nextNode.value:
            parentNodes.insert(0, nextNode)
            nextNode = nextNode.right

    return parentNodes

#Classifies a given image as either a square or a circle
def kDClassifyImage(k, image, kDTree):
    #Keep track of the K nearest neaighbours
    nearestNeighbours = []
    #Keep track of the K best distances
    bestDistances = []
    #Set the best distances to infinity
    for neighbours in range(k):
        nearestNeighbours.append(None)
        bestDistances.append(math.inf)

    #Get the path down the tree
    parentNodes = drop(image, kDTree)
    numberRows = len(image)
    numberColumns = len(image[0])
    pixels = numberRows * numberColumns
    #For each parent node
    for node in parentNodes:
        #Get the distance to the parent
        nodeDistance = 0
        #Euclidean distance between the input image and the parent node's image
        #Euclidean distance for each pixel
        for pix in range(pixels):
            rowIndex = int(pix / numberColumns)
            columnIndex = pix % numberColumns
            nodeDistance += (node.image[rowIndex, columnIndex] - image[rowIndex, columnIndex]) ** 2

        nodeDistance = nodeDistance ** 0.5
        #If the distance is shorter than our worst best distance
        #We can't be sure that there won't be closer nodes on the other side of the parent
        if nodeDistance < max(bestDistances):
            kIndex = bestDistances.index(max(bestDistances))
            #Replace the worst nearest neighbour with the parent
            bestDistances[kIndex] = nodeDistance
            nearestNeighbours[kIndex] = node
            #If we went left at the parent
            #We descend the right branch
            if image[node.row, node.column] <= node.value:
                #If a right branch exists
                if node.right != None:
                    descend = kDClassifyImage(k, image, node.right)
                    descendedDistances = descend[1]
                    descendedNeighbours = descend[0]
                    #Descending the opposte branch just returned us a list of best distances and nearest neighbours found in that branch
                    #Now we have to check if any of those nearest neighbours are better than our current nearest neighbours
                    while min(descendedDistances) != math.inf:
                        #Look at the nearest neighbour we found
                        descendedIndex = descendedDistances.index(min(descendedDistances))
                        #If it's nearer than our current nearest neighbour
                        if min(descendedDistances) < max(bestDistances):
                            #Replace our worst nearest neighbour with the new neighbour
                            bestIndex = bestDistances.index(max(bestDistances))
                            bestDistances[bestIndex] = descendedDistances[descendedIndex]
                            nearestNeighbours[bestIndex] = descendedNeighbours[descendedIndex]

                        #Set the best nearest neighbour from the parent branch to infinity, so we know not to look at it again
                        descendedDistances[descendedIndex] = math.inf

            #If we went right at the parent branch
            #We go left
            elif image[node.row, node.column] > node.value:
                #If a left branch exists
                if node.left != None:
                    descend = kDClassifyImage(k, image, node.left)
                    descendedDistances = descend[1]
                    descendedNeighbours = descend[0]
                    #Descending the opposte branch just returned us a list of best distances and nearest neighbours found in that branch
                    #Now we have to check if any of those nearest neighbours are better than our current nearest neighbours
                    while min(descendedDistances) != math.inf:
                        #Look at the nearest neighbour we found
                        descendedIndex = descendedDistances.index(min(descendedDistances))
                        #If it's nearer than our current nearest neighbour
                        if min(descendedDistances) < max(bestDistances):
                            #Replace our worst nearest neighbour with the new neighbour
                            bestIndex = bestDistances.index(max(bestDistances))
                            bestDistances[bestIndex] = descendedDistances[descendedIndex]
                            nearestNeighbours[bestIndex] = descendedNeighbours[descendedIndex]

                        #Set the best nearest neighbour from the parent branch to infinity, so we know not to look at it again
                        descendedDistances[descendedIndex] = math.inf

    #Return the nearest neighbours we found
    results = [nearestNeighbours, bestDistances]
    return results

#Classifies a set of images using a given KD Tree
#retursn the accuracy of the KD Tree
def kNNClassifier(k, images, labels, kDTree):
    #A list of the classifcations predicted by the k nearest neighbours algorithm
    classifications = []
    for image in images:
        #Classify each image
        results = kDClassifyImage(k, image, kDTree)
        nearestNeighbours = results[0]
        neighbourDistances = results[1]
        #Keep track of the number of neighbours that we found that were squares/circles
        squareCount = 0
        #In the event that we found an equal amount of squares and circles
        #The classification will be determined by which has the smallest combined distance
        squareDistance = 0
        circleCount = 0
        circleDistance = 0
        kIndex = 0
        #For each neighbour we found
        for neighbour in nearestNeighbours:
            #If it was a square
            if neighbour.label == [0]:
                squareCount += 1
                squareDistance = squareDistance + neighbourDistances[kIndex]

            #If it was a circle
            elif neighbour.label == [1]:
                circleCount += 1
                circleDistance = circleDistance + neighbourDistances[kIndex]

            kIndex += 1

        #If we found more square neighbours than circle neighbours
        #Classify the image as a square
        if squareCount > circleCount:
            classifications.append(0)

        #If we found more circle neighbours than square neighbours
        #Classify the image as a circle
        elif squareCount < circleCount:
            classifications.append(1)

        #If we found the same number of square and circle neighbours
        elif squareCount == circleCount:
            #Classify the image as whatever had the smallest total distance from the image
            if squareDistance < circleDistance:
                classifications.append(0)

            elif squareDistance > circleDistance:
                classifications.append(1)

    labelCount = 0
    successes = 0
    #Count our successes
    for label in labels:
        #If the label matches our classification
        if label[0] == classifications[labelCount]:
            successes += 1

        labelCount += 1

    accuracy = successes / len(images)
    return accuracy

#Plot the accuracy results of the KNN classification
def plotKDAccuracy(x_train, y_train, x_test, y_test, kDTree):
    trainAccuracy = []
    testAccuracy = []
    xAxis = []
    #Get the accuracies for each K
    for k in range(1, 11):
        trainAccuracy.append(kNNClassifier(k, x_train, y_train, kDTree))
        testAccuracy.append(kNNClassifier(k, x_test, y_test, kDTree))
        #Generate the X Axis to use on the plot
        xAxis.append(k)
  
    #Format the plot
    plt.plot(xAxis, trainAccuracy)
    plt.plot(xAxis, testAccuracy)
    plt.title('KNN Accuracy')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend(['training set', 'testing set'], loc='upper right')
    #The optimal generalisation point is the point where the test data was classified most accurately
    testMax = max(testAccuracy)
    testMaxX = testAccuracy.index(max(testAccuracy))
    testMaxX = xAxis[testMaxX]
    plt.annotate('Optimal generalisation', xy = (testMaxX, testMax), xytext = (testMaxX, testMax), arrowprops = dict(facecolor = 'black', shrink = 0.05))
    plt.show()

#Get the probabilities of an image in the set being a square or a circle
def getProbability(labels):
    probabilities = []
    #If there are no labels, there can be no classifications
    #Hence 0 chance of being a square or a circle
    if len(labels) == 0:
        probabilities.append(0)
        probabilities.append(0)
        return probabilities

    squareCount = 0
    circleCount = 0
    for label in labels:
        if label == [0]:
            squareCount += 1

        elif label == [1]:
            circleCount += 1

    probabilitySquare = squareCount / len(labels)
    probabilityCircle = circleCount / len(labels)
    probabilities.append(probabilitySquare)
    probabilities.append(probabilityCircle)
    return probabilities

#Choose which attribute to split the data on
#Each pixel in an image is an attribute
#Returns the bets attribute and the best value to split the data on
def chooseAttribute(images, labels, attributes, informationRoot):
    gain = -1
    bestAttribute = -1
    bestValue = -1
    testImage = images[0]
    numberRows = len(testImage)
    numberColumns = len(testImage[0])
    #For each attribute
    for attribute in attributes:
        row = int(attribute / numberColumns)
        column = attribute % numberColumns
        value = 0
        #Need to determine a value that will split the data on the selected attribute
        #This is the pixel value of the attribute
        #i.e. attribute <= value goes to the left branch, > goes right
        #Found through trial and error testing
        #Start at 0 and increase to 1
        #Increments of 0.1
        while value < 1.1:
            #Sort the images into left and right branches
            leftImages = []
            leftLabels = []
            rightImages = []
            rightLabels = []
            labelCount = 0
            for image in images:
                #If the attribute is less than or equal to the splitting value
                if image[row, column] <= value:
                    #The image gets passed to the left branch
                    leftImages.append(image)
                    leftLabels.append(labels[labelCount])

                #If the attribute is greater than the splitting value
                elif image[row, column] > value:
                    #The image gets passed to teh right branch
                    rightImages.append(image)
                    rightLabels.append(labels[labelCount])

                labelCount += 1

            #The probability of an image getting passed to the left branch
            probabilityLeft = len(leftImages) / len(images)
            probabilities = getProbability(leftLabels)
            probabilitySquare = probabilities[0]
            probabilityCircle = probabilities[1]
            informationLeft = 0
            #Calculate the information in the left branch
            #If the probability of an image being a square isn't 0
            #If the probability of an image being a square is 0 it adds 0 information to the node
            if probabilitySquare != 0:
                informationLeft += - probabilitySquare * math.log(probabilitySquare, 2)
            
            #If the probability of an image being a circle isn't 0
            #If the probability of an image being a circle is 0 it adds 0 information to the node
            if probabilityCircle != 0:
                informationLeft += - probabilityCircle * math.log(probabilityCircle, 2)

            #Calculate the remainder of the attribute split
            remainder = probabilityLeft * informationLeft
            #The probability of an image getting passed to the right branch
            probabilityRight = len(rightImages) / len(images)
            probabilities = getProbability(rightLabels)
            probabilitySquare = probabilities[0]
            probabilityCircle = probabilities[1]
            informationRight = 0
            #Calculate the information in the right branch
            #If the probability of an image being a square isn't 0
            #If the probability of an image being a square is 0 it adds 0 information to the node
            if probabilitySquare != 0:
                informationRight += - probabilitySquare * math.log(probabilitySquare, 2)

            #If the probability of an image being a circle isn't 0
            #If the probability of an image being a circle is 0 it adds 0 information to the node
            if probabilityCircle != 0:
                informationRight += - probabilityCircle * math.log(probabilityCircle, 2)

            remainder += probabilityRight * informationRight
            #Calculate the gain of the attribute split
            tempGain = informationRoot - remainder
            #If the gain is greater than our current gain, we keep it
            if tempGain > gain:
                gain = tempGain
                bestAttribute = attribute
                bestValue = value

            value += 0.1

    results =[]
    results.append(bestAttribute)
    results.append(bestValue)
    return results

#Build the decision tree based on a set of training data
#Attributes is a list of how many pixels there are in the images
#PrePruning tells us when to stop splitting data at a node
#If the information at the node is less than the prepruning paramter, we stop splitting
#Default is a default classification value to return if we get stuck
#Default is the majority classification of the parent node
#We start with a default of 0
def buildDecisionTree(images, labels, attributes, prePruning, default):
    #If there are no more images, return the default value
    if len(images) == 0:
        tree = DecisionTree()
        tree.classification = default
        return tree

    probabilities = getProbability(labels)
    probabilitySquare = probabilities[0]
    probabilityCircle = probabilities[1]

    #If all of our images are squares
    if probabilitySquare != 0 and probabilityCircle == 0:
        #Stop splitting and set the classification to square
        tree = DecisionTree()
        tree.classification = 0
        return tree

    #If all of our images are circle
    elif probabilitySquare == 0 and probabilityCircle != 0:
        #Stop splitting and set the classification to circle
        tree = DecisionTree()
        tree.classification = 1
        return tree

    #If we're out of attributes to split on
    elif len(attributes) == 0:
        #If the majority of images are squares
        if probabilitySquare > probabilityCircle:
            tree = DecisionTree()
            tree.classification = 0
            return tree

        #If the majority of images are circles
        elif probabilitySquare < probabilityCircle:
            tree = DecisionTree()
            tree.classification = 1
            return tree

        #If there are an equal number of squares and circles
        elif probabilitySquare == probabilityCircle:
            #Return the default
            tree = DecisionTree()
            tree.classification = default
            return tree

    else:
        #Calculate the information at the root
        informationRoot = - probabilitySquare * math.log(probabilitySquare, 2)
        informationRoot += - probabilityCircle * math.log(probabilityCircle, 2)
        #If the information value is less than the prepruning parameter
        if informationRoot < prePruning:
            #We stop splitting
            tree = DecisionTree()
            if probabilitySquare > probabilityCircle:
                tree.classification = 0

            elif probabilitySquare < probabilityCircle:
                tree.classification = 1

            elif probabilitySquare == probabilityCircle:
                tree.classification = default

            return tree

        #Get the best attribute and value
        attributeAndValue = chooseAttribute(images, labels, attributes, informationRoot)
        bestAttribute = attributeAndValue[0]
        bestValue = attributeAndValue[1]
        testImage = images[0]
        numberColumns = len(testImage[0])
        row = int(bestAttribute / numberColumns)
        column = bestAttribute % numberColumns
        tree = DecisionTree()
        tree.attribute = bestAttribute
        tree.value = bestValue
        leftImages = []
        leftLabels = []
        rightImages = []
        rightLabels = []
        labelCount = 0
        #Split the data to the left and right branches
        for image in images:
            if image[row, column] <= tree.value:
                leftImages.append(image)
                leftLabels.append(labels[labelCount])

            elif image[row, column] > tree.value:
                rightImages.append(image)
                rightLabels.append(labels[labelCount])

            labelCount += 1

        #We cannot split on the best attribute again, so we remove it from the list of attributes
        attributes.remove(bestAttribute)
        #Get the new deafult to passed on
        #If there are more squares than circles at the current root
        if probabilitySquare > probabilityCircle:
            #Set the default to square
            default = 0

        #If there are more circles than squares at the current root
        elif probabilitySquare < probabilityCircle:
            #Set the default to circle
            default = 1
        
        #If there are the same number of squares and circles at the current root
        #The default stays the same
        #Build the left branch
        subtree = buildDecisionTree(leftImages, leftLabels, attributes, prePruning, default)
        tree.left = subtree
        #Build the right branch
        subtree = buildDecisionTree(rightImages, rightLabels, attributes, prePruning, default)
        tree.right = subtree
        return tree

#Classify an image by using the decision tree
#Returns a classification
def decisionTreeClassifyImage(image, decisionTree):
    #If we're at a leaf node
    if decisionTree.classification != None:
        #Return the classification
        return decisionTree.classification

    else:
        #Drop the image further down the decision tree
        #According to its value of the current attribute
        attribute = decisionTree.attribute
        numberColumns = len(image[0])
        row = int(attribute / numberColumns)
        column = attribute % numberColumns
        if image[row, column] <= decisionTree.value:
            return decisionTreeClassifyImage(image, decisionTree.left)

        elif image[row, column] > decisionTree.value:
            return decisionTreeClassifyImage(image, decisionTree.right)

#Classifies a set of test images with a given decision tree
#Returns the accuracy of the decision tree
def decisionTreeClassifier(images, labels, decisionTree):
    classifications = []
    #Get the classification for each image
    for image in images:
        classifications.append(decisionTreeClassifyImage(image, decisionTree))

    labelCount = 0
    successes = 0
    #Get a count of successful classifications
    for label in labels:
        if label[0] == classifications[labelCount]:
            successes += 1

        labelCount += 1

    accuracy = successes / len(images)
    return accuracy

#Plot the accuracy of the decision tree algorithm
def plotDTAccuracy(x_train, y_train, x_test, y_test):
    attributes = []
    testImage = x_train[0]
    numberRows = len(testImage)
    numberColumns = len(testImage[0])
    pixels = numberRows * numberColumns
    #Generate a list of attributes (pixel locations)
    for i in range(pixels):
        attributes.append(i)

    trainAccuracy = []
    testAccuracy = []
    prePruning = 0
    xAxis = []
    #Get the accuracies of the decision tree from prepruning values 0 through to 0.5
    while prePruning <= 0.5:
        #Have to build a new decision tree for each prepruning value
        decisionTree = buildDecisionTree(x_train, y_train, attributes, prePruning, 0)
        #Get the accuracies with the training and testing data sets
        trainAccuracy.append(decisionTreeClassifier(x_train, y_train, decisionTree))
        testAccuracy.append(decisionTreeClassifier(x_test, y_test, decisionTree))
        #Generate the X axis to use in the plot
        xAxis.append(prePruning)
        prePruning += 0.05
        
    #Generate the plot of accuracies
    plt.plot(xAxis, trainAccuracy)
    plt.plot(xAxis, testAccuracy)
    plt.title('Decision Tree Accuracy')
    plt.xlabel('PrePruning Bits')
    plt.ylabel('Accuracy')
    plt.legend(['training set', 'testing set'], loc = 'upper right')
    #The point of optimal generalisation is the where the testing accuracy is best
    testMax = max(testAccuracy)
    testMaxX = testAccuracy.index(testMax)
    testMaxX = xAxis[testMaxX]
    plt.annotate('Optimal generalisation', xy = (testMaxX, testMax), xytext = (testMaxX, testMax), arrowprops = dict(facecolor = 'black', shrink = 0.05))
    plt.show()

def cNNClassifier(x_train, y_train, x_test, y_test):
    # batch size for gradient descent
    batch_size = 32
    # number of MNIST classes
    num_classes = 2
    # number of epochs (1 epoch = amount of iterations that covers the whole training set)
    epochs =  200
    # input image dimensions
    nmb_samples, img_rows, img_cols = x_train.shape[0], x_train.shape[1], x_train.shape[2]
    nmb_test_samples = x_test.shape[0]

    # adjust training image format
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)

    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # type casting and dimensionality transformations
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # convert class vectors to binary class matrices (one hot vectors)
    y_train = keras.utils.to_categorical(np.squeeze(y_train), num_classes)
    y_test = keras.utils.to_categorical(np.squeeze(y_test), num_classes)

    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    testMax = max(history.history['val_acc'])
    testMaxX = history.history['val_acc'].index(testMax)
    plt.annotate('Optimal generalisation', xy = (testMaxX, testMax), xytext = (testMaxX, testMax), arrowprops = dict(facecolor = 'black', shrink = 0.05))
    plt.title('CNN Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
       
def main():
    # load dataset
    dataset = scipy.io.loadmat('dataset.mat')

    # get training and testing sets
    x_train = dataset['train_image']
    y_train = dataset['train_label']
    x_test = dataset['test_image']
    y_test = dataset['test_label']

    kDTree = buildKDTree(x_train, y_train, 0)
    plotKDAccuracy(x_train, y_train, x_test, y_test, kDTree)
    plotDTAccuracy(x_train, y_train, x_test, y_test)
    cNNClassifier(x_train, y_train, x_test, y_test)

main()