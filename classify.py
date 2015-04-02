"""
Assignment 4. Implement a Naive Bayes classifier for spam filtering.
You'll only have to implement 3 methods below:
train: compute the word probabilities and class priors given a list of documents labeled as spam or ham.
classify: compute the predicted class label for a list of documents
evaluate: compute the accuracy of the predicted class labels.
"""

import glob
from collections import defaultdict

class Document(object):
    """ A Document. DO NOT MODIFY.
    The instance variables are:
    filename....The path of the file for this document.
    label.......The true class label ('spam' or 'ham'), determined by whether the filename contains the string 'spmsg'
    tokens......A list of token strings.
    """

    def __init__(self, filename):
        self.filename = filename
        self.label = 'spam' if 'spmsg' in filename else 'ham'
        self.tokenize()

    def tokenize(self):
        self.tokens = ' '.join(open(self.filename).readlines()).split()


class NaiveBayes(object):
    
		
    def train(self, documents):
        """
        TODO: COMPLETE THIS METHOD.
        Given a list of labeled Document objects, compute the class priors and
        word conditional probabilities, following Figure 13.2 of your book.
        """
        # class prior = P(y)
	    # word conditional probability P(x|y) 
        # Ti = no(of T in class i)
        # ET = no(of tokens in class i)
	
        epsilon = 1
        no_of_spam_doc = 0
        no_of_ham_doc  = 0
	
        spam_tokens    = 0
        ham_tokens     = 0
	
        spamdoc_tf  = defaultdict(int)
        hamdoc_tf   = defaultdict(int)
        self.total_words = defaultdict(int) 
	
        spam_word_prob   = defaultdict(int)
        ham_word_prob    = defaultdict(int)

        for doc in documents:
            if doc.label == "spam":
                no_pf_spam_doc += 1
                spam_tokens += len(doc.tokens)
                for word in doc.tokens:
                    spamdoc_words[word] += 1
                    total_words[word] += 1
            if doc.label == "ham":
                no_of_ham += 1
                ham_tokens += len(doc.tokens)
                for word in doc.tokens:
                    hamdoc_words[word] += 1
                    total_words[word] += 1
	
        for w in total_words:
            self.spam_word_prob[w] = spamdoc_tf[w] + epsilon / spam_tokens + (epsilon * len(total_words))  
            self.ham_word_prob[w] = hamdoc_tf[w] + epsilon / ham_tokens + (epsilon * len(total_words)) 
	
    def classify(self, documents):
        """
        TODO: COMPLETE THIS METHOD.
        Return a list of strings, either 'spam' or 'ham', for each document.
        documents....A list of Document objects to be classified.
        """
        spam_prob = 0
        ham_prob  = 0 
        
        for w in total_words:
            spam_prob +=  

def evaluate(predictions, documents):
    """
    TODO: COMPLETE THIS METHOD.
    Evaluate the accuracy of a set of predictions.
    Print the following:
    accuracy=xxx, yyy false spam, zzz missed spam
    where
    xxx = percent of documents classified correctly
    yyy = number of ham documents incorrectly classified as spam
    zzz = number of spam documents incorrectly classified as ham
    See the provided log file for the expected output.
    predictions....list of document labels predicted by a classifier.
    documents......list of Document objects, with known labels.
    """


def main():
    """ DO NOT MODIFY. """
    train_docs = [Document(f) for f in glob.glob("train/*.txt")]
    print 'read', len(train_docs), 'training documents.'
    nb = NaiveBayes()
    nb.train(train_docs)
    test_docs = [Document(f) for f in glob.glob("test/*.txt")]
    print 'read', len(test_docs), 'testing documents.'
    predictions = nb.classify(test_docs)
    evaluate(predictions, test_docs)

if __name__ == '__main__':
    main()
