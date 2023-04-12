import random
import numpy as np
import redditScraper
import bow
import model

# subreddit paths
conservative = ['conservative',
                'republican',
                'conservatives',
                'conservativesonly',
                'libertarian']
liberal = ['democraticSocialism',
           'progressive',
           'democrats',
           'liberal',
           'neoliberal']

# subreddit classifications
classes = [conservative,
           liberal]

n_samples = 5000
recreate_db = False

# pulls training data
data = redditScraper.retrieve_db(classes, n_samples, recreate_db)

# creates a blank BoW object ready to be fit to data
bag = bow.Bag()

# creates object layer and layer objects. First layer 1 is created later so that
# data input length can be passed as an argument 
activation1 = model.Activation_ReLU()
dense2 = model.Layer_Dense(10, 2) # (num L1 neurons, num L2 neuorons/outputs)
loss_activation = model.Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = model.Optimizer_Adam(learning_rate = .05, decay = 5e-7)
    
# creates one list of all subreddit comments per possible class
def merge_subR_data(data, class_index):
    # finds current classification 
    class_data = data[class_index]
    all_comments = []

    # creates a list of all comments contained in individual subR lists
    for comment_list in class_data:
        for comment in comment_list:
            all_comments.append(comment)
    
    return all_comments

def ordered_shuffle(data, classes):
    # temp list allows both lists to be manipulated identically
    temp = list(zip(data, classes))
    random.shuffle(temp)
    shuff_data, shuff_classes = zip(*temp)

    # must be converted to np arrays for split()
    return np.array(shuff_data), np.array(shuff_classes)

# multiplies len(data) by ratio's 1st element to get an index to split at
def split(data, ratio):
    #assert ratio's sum = 1 for a more accurate split
    assert(np.sum(ratio) == 1.0)
    train_ratio = ratio[0]
    indices = [int(len(data) * train_ratio)]
    train, test = np.split(data, indices)
    return train, test

def test_model(test_data, test_values):
    correct = 0

    # iterates through the testing sample counting correct guesses
    for i, data in enumerate(test_data):
        input = classify(data)
        if input[0][0] > input[0][1] and test_values[i][0] == 1:
            correct += 1
        elif input[0][0] < input[0][1] and test_values[i][0] == 0: 
            correct += 1

    print("Testing Accuracy: ", correct/len(test_data))

# creates an infinite loop to repeatedly classify an input string
def guess_inputs():
    while True:
        comment = input("Enter input: ")
        new_data = bag.build_index(comment, 0)
        answer = classify(new_data)
        print(answer)

        if answer[0][0] > answer[0][1]:
            print("conservative")
        elif answer[0][0] < answer[0][1]:
            print("liberal")
        else:
            print("not sure")

# runs a forward pass to use the model
def classify(data):
    dense1.forward(data)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    return dense2.output

# merges and blends subreddit training data
con_data = merge_subR_data(data, 0)
lib_data = merge_subR_data(data, 1)
random.shuffle(con_data)
random.shuffle(lib_data)

# fits BoW to data
bow_data = bag.fit_build(con_data, lib_data)

# gathers training data generated while BoW is being fit
bow_classes = np.array(bag.testing)

#shuffles data while preserving relative order
inputs_unsplit, classes_unsplit = ordered_shuffle(bow_data, bow_classes)

#splits testing/training sets
training_inputs, testing_inputs = split(inputs_unsplit, [0.8, 0.2])# training input
training_classes, testing_classes = split(classes_unsplit, [0.8, 0.2])# model validation

# creates an object for the first layer
dense1 = model.Layer_Dense(len(training_inputs[0]), 10) # (num inputs, num, L1 neurons)

# trains the model
for epoch in range(25):
    # forward passes and calculating loss, predictions and accuracy
    dense1.forward(training_inputs)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, training_classes)
    predictions = np.argmax(loss_activation.output, axis = 1)
    accuracy = np.mean(predictions == training_classes)
    
    # fits training_classes for backward pass 
    if len(training_classes.shape) == 2:
        training_classes = np.argmax(training_classes, axis = 1)

    print('epoch: ', epoch, ' acc: ', accuracy, ' loss: ', loss, ' learning rate: ', optimizer.current_learning_rate)
    
    # backward passes
    loss_activation.backward(loss_activation.output, training_classes)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # tweaks the network
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

test_model(testing_inputs, testing_classes)

# uses the model on user input
guess_inputs()