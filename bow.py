import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#nltk.download('stopwords')#only need to run download once
from nltk.corpus import stopwords

class Bag:
    def __init__(self):
        self.testing = []
        self.bag_sum = []

        self.all_no_repeats = set()
        self.word_to_index = {}
        self.index_to_word = {}

    def fit_data(self, con_data, lib_data):
        con_bag = self.bagify(con_data)#can we move these or remove these bagify calls?
        lib_bag = self.bagify(lib_data)

        all_words = con_bag + lib_bag
        self.all_no_repeats = set(all_words)

        for index, word in enumerate(self.all_no_repeats):
            self.word_to_index[word] = index
            self.index_to_word[index] = word

    def bagify(self, data):
        bag = []
        data_size = np.array(data).size
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))#set of irrelevant words

        # refactor this if heather has a solution
        if data_size > 1:
            for string in data:
                string = word_tokenize(string)#seperates comments into tokens
                for word in string:
                    word = lemmatizer.lemmatize(word)#converts words to their roots
                    if (word not in stop_words):
                        bag.append(word)
        
        #parses individual comments
        else:
            data = word_tokenize(data)
            for word in data:
                word = lemmatizer.lemmatize(word)
                if (word not in stop_words):
                    bag.append(word)

        return bag

    def build_data(self, con_data, lib_data):
        frequency_matrix = np.empty((len(con_data) + len(lib_data), len(self.word_to_index)))
        total_data = np.concatenate([con_data, lib_data])
        con_size = np.array(con_data).size
        polarity = 0

        for i, sample in enumerate(total_data):
            if i < con_size:
                polarity = -1
                self.testing.append([1,0])
            else:
                polarity = 1
                self.testing.append([0,1])

            frequency_matrix[i] = self.build_index(sample, polarity)

        self.bag_sum = sum(frequency_matrix)
        return frequency_matrix
    
    def build_index(self, comment, polarity):
        frequency_list = np.zeros(len(self.all_no_repeats))

        if any(isinstance(value, str) for value in comment):
            comment_bag = self.bagify(comment)
        else:
            comment_bag = comment

        for word in comment_bag:
            if word in self.all_no_repeats:#checks for removed stop words
                index = self.word_to_index[word]
                if polarity == 0:# 0 is passed to build a single index for post training testing/classification
                    frequency_list[index] = self.bag_sum[index]
                else:
                    frequency_list[index] += polarity

        return frequency_list

    def fit_build(self, con_data, lib_data):
        self.fit_data(con_data, lib_data)
        return self.build_data(con_data, lib_data)