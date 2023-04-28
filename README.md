# Binary-Sentiment-Analysis-with-Machine-Learning
This is my first machine learning project. The application is designed to determine the polical affiliation of commentors on reddit. It works by gathering comments from political focused subreddits using the Reddit API, and transforms inputed data into one hot encoded bags of words for processing by a neural network.

I avoided using any machine learning packages so that I would have to work directly with the math that makes these networks possible, although I did use some code found in the book "Neural Networks from Scratch" in order to actually generate a model, so that I wouldn't need to write out all of the verbose formulas myself. 

The model uses one hot encoding to determine whether or not a word appears in a given comment during training, and the positivity or negativity of the encoded variable indicates its political affiliation to the model. I use the Rectified Linear Unit formula as the activation formula for the neurons in the hidden layer, and the softmax function for the output layer. The ADAM optimizer was used during training, as it was just as effective as the standard Stochastic Gradient Descent, except it was much faster to build the model.

The application consists of four modules, the purposes of which are described below:

1. redditScraper gathers comments from subreddits.

2. The bow module turns comments into bags of words.

3. The model module creates a neural network and uses bags of words to train it.

4. The Main file pulls all of these elements together, makes some minor adjustments to prepare the data, and tests the model for accuracy and loss.
