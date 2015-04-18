"""
Assignment 5: K-Means. See the instructions to complete the methods below.
"""

from collections import Counter
import io
import math
from collections import defaultdict
import numpy as np


class KMeans(object):

    def __init__(self, k=2):
        """ Initialize a k-means clusterer. Should not have to change this."""
        self.k = k
        #self.distance_from_mean = defaultdict(list)     # dictionary mapping a doc to its distances from each mean 
        self.list_of_means = []
        

    def cluster(self, documents, iters=10,k=2):
        """
        Cluster a list of unlabeled documents, using iters iterations of k-means.
        Initialize the k mean vectors to be the first k documents provided.
        Each iteration consists of calls to compute_means and compute_clusters.
        After each iteration, print:
        - the number of documents in each cluster
        - the error rate (the total Euclidean distance between each document and its assigned mean vector)
        See Log.txt for expected output. 
        """    
        self.list_of_means = documents[:10]
        self.documents = documents
         
        #self.compute_clusters(documents)
                
        for i in range(iters):
            #print "calling compute_clusters"
            self.compute_clusters(documents)
            #print "calling compute_means"
            self.compute_means(documents)
            print "[%d," % (len(self.mean_to_doc[0])),
            for cluster_id in range(1,self.k - 1):
                print "%d," % len(self.mean_to_doc[cluster_id]),
            print "%d]" % len(self.mean_to_doc[self.k - 1])

            print self.error(documents)
            #for cluster_id in range(self.k):
            #    print self.errors[cluster_id],
            #print 
        
    def compute_means(self, documents):
        """ Compute the mean vectors for each cluster (storing the results in an
        instance variable)."""
        temp_means = list()
        total = 0
        # self.mean_to_doc 
        # for each cluster
        for mean_id in range(self.k):
            mean = Counter()
            # for  all the documents of cluster
            for doc_id in self.mean_to_doc[mean_id]:
                # for all the words of a document 
                for word in documents[doc_id]:
                    mean[word] += documents[doc_id][word]
            
            for word in mean:
                mean[word] =  float(mean[word]) / float(len(self.mean_to_doc[mean_id]))
            
            temp_means.append(mean)
        self.list_of_means = temp_means 

    def compute_clusters(self, documents):
        """ Assign each document to a cluster. (Results stored in an instance
        variable). """
        # dictionary mean-doc = mean_id(cluster_id) -> list of doc_id in that cluster
        #self.mean_to_doc = defaultdict(list)
        self.mean_to_doc = {}
        
        for index in range(self.k):
            self.mean_to_doc[index] = []
            
        mean_mag = []
        for mean in self.list_of_means:
            mean_mag.append(self.calc_mean_norm(mean))
        
        # for each document in documents 
        for doc_id, doc in enumerate(documents):
            min_index = 0
            # for each mean in global list of means
            doc_mag = self.calc_mean_norm(doc)
            min_dist = self.distance(doc, self.list_of_means[0], doc_mag + mean_mag[0])
            
            for mean_id,mean in enumerate(self.list_of_means):
                # calculate distances of doc to mean global means
                if mean_id == 0:
                    continue 
                #mean_mag = self.calc_mean_norm(mean)
                mean_norm = mean_mag[mean_id] + doc_mag 
                dist = self.distance(doc,mean,mean_norm)
                if dist < min_dist:
                    min_dist = dist
                    min_index = mean_id
            self.mean_to_doc[min_index].append(doc_id) 
        
        return self.mean_to_doc 
        
    def calc_mean_norm(self, doc):
        # calculates mean_norm for a given doc and mean(||x||^2 + ||u||^2)
        doc_mag = 0.0 
        #print doc
        for term in doc:
            #print term
            doc_mag += doc[term] ** 2
        return doc_mag 
        
    def distance(self, doc, mean, mean_norm):
        """ Return the Euclidean distance between a document and a mean vector.
        See here for a more efficient way to compute:
        http://en.wikipedia.org/wiki/Cosine_similarity#Properties"""
        
        dot_product = 0.0
        for word in doc:
            #print word
            dot_product += doc[word] * mean[word]
        
        return mean_norm - 2*(dot_product) 
        

    def error(self, documents):
        """ Return the error of the current clustering, defined as the sum of the
        Euclidean distances between each document and its assigned mean vector."""
        '''
        self.errors = defaultdict(int)
        '''
        self.error_per_doc = defaultdict(int)
        error = 0.0
        for mean_index in self.mean_to_doc:
            mean_mag = self.calc_mean_norm(self.list_of_means[mean_index])
            for doc_index in self.mean_to_doc[mean_index]:
                doc_mag = self.calc_mean_norm(documents[doc_index])
                mean_norm = mean_mag + doc_mag 
                #dist = 4.0
                dist = self.distance(documents[doc_index],self.list_of_means[mean_index],mean_norm)
                error_from_a_doc = math.sqrt(dist)
                error += error_from_a_doc
                self.error_per_doc[doc_index] = error_from_a_doc
        self.error_val = error
        return self.error_val
        
    def print_top_docs(self, n=10):
        """ Print the top n documents from each cluster, sorted by distance to the mean vector of each cluster.
        Since we store each document as a Counter object, just print the keys
        for each Counter (which will be out of order from the original
        document).
        Note: To make the output more interesting, only print documents with more than 3 distinct terms.
        See Log.txt for an example."""
        
        for mean_index in self.mean_to_doc:
            print "CLUSTER %d" % mean_index
            
            top_docs = sorted([x for x in self.mean_to_doc[mean_index] if len(self.documents[x]) > 3], key=lambda x: self.error_per_doc[x])[:n]
            for doc in top_docs:
                for word in self.documents[doc].keys():
                    print word.encode("utf-8"),
                print
        
def prune_terms(docs, min_df=3):
    """ Remove terms that don't occur in at least min_df different
    documents. Return a list of Counters. Omit documents that are empty after
    pruning words.
    >>> prune_terms([{'a': 1, 'b': 10}, {'a': 1}, {'c': 1}], min_df=2)
    [Counter({'a': 1}), Counter({'a': 1})]
    """
    
    #first, determine document frequencies of each term
    doc_freq = Counter()
    result = []
   
    for doc in docs:
        for word in doc:
            doc_freq[word] += 1
    
    for doc in docs:
        result_doc = Counter()
        for term in doc:
            if doc_freq[term] >= min_df:
                result_doc[term] = doc[term]
        if len(result_doc) > 0:
            result.append(result_doc)
    return result 
    
    
def read_profiles(filename):
    """ Read profiles into a list of Counter objects.
    DO NOT MODIFY"""
    profiles = []
    with io.open(filename, mode='rt', encoding='utf8') as infile:
        for line in infile:
            profiles.append(Counter(line.split()))
    return profiles


def main():
    """ DO NOT MODIFY. """
    profiles = read_profiles('profiles.txt')
    print 'read', len(profiles), 'profiles.'
    profiles = prune_terms(profiles, min_df=2)
    km = KMeans(k=10)
    km.cluster(profiles, iters=20)
    km.print_top_docs()

if __name__ == '__main__':
    main()