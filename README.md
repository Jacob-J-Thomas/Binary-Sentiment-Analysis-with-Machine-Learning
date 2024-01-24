# Binary-Sentiment-Analysis-with-Machine-Learning
### Project Sumarry
This application is designed to determine the polical affiliation of commentors on reddit. It works by gathering comments from political focused subreddits using the Reddit API, and transforms this data into a bags of words, and uses sentiment analyis to determine a binary classification. 

The choice to use polical subreddits ended up not being the best, as much of the language was actually rather similar once I drilled into the extracted text, but it was a great learning process for this reason as well, and taught me a lot about what constitutes a good machine learning problem, and what types of approaches should be used to accomplish different tasks.

For this project, I also wanted to avoid using any existing machine learning packages so that I would have to work directly with the math that makes these networks possible. Although I did use some code found in the book "Neural Networks from Scratch" since otherwise I would have just been rewriting equations at a ceratin point. Implementing the code this way allowed me to gain a much deeper understanding of the math that makes these programs possible.

The model uses one hot encoding to determine whether or not a word appears in a given comment during training, and uses this to guess the political affiliation of the commentor. I use the Rectified Linear Unit formula as the activation formula for the neurons in the hidden layer, and the softmax function for the output layer. The ADAM optimizer was used during training, as it provided the best balance of performance to training time.

### Application Overview
The application consists of four modules, the purposes of which are described below:

1. "redditScraper.py" gathers comments from subreddits.

2. The "bow.py" module turns comments into bags of words.

3. The "model.py" module creates a neural network and uses bags of words to train it.

4. The "main.py" file pulls all of these elements together, makes some minor adjustments to prepare the data, and tests the model for accuracy and loss.
