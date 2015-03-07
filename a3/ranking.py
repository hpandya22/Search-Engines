""" Assignment 3
You will modify Assignment 1 to support cosine similarity queries.
The documents are read from documents.txt.
The index will store tf-idf values using the formulae from class.
The search method will sort documents by the cosine similarity between the
query and the document (normalized only by the document length, not the query
length, as in the examples in class).
The search method also supports a use_champion parameter, which will use a
champion list (with threshold 10) to perform the search.
"""
from collections import defaultdict
import codecs
import math
import re
from operator import itemgetter 

class Index(object):

    def __init__(self, filename="TIME.ALL" , champion_threshold=10):
        """ DO NOT MODIFY.
        Create a new index by parsing the given file containing documents,
        one per line. You should not modify this. """
        if filename:  # filename may be None for testing purposes.
            self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
            self.documents = self.read_doc(filename)
            print "num docs=", len(self.documents)
            #print self.documents
            stemmed_docs = [self.stem(self.tokenize(d)) for d in self.documents]
            self.words_count = self.word_counts(self.documents)
    
            
            self.doc_freqs = self.count_doc_frequencies(stemmed_docs)
            self.index = self.create_tfidf_index(stemmed_docs, self.doc_freqs)
            
            self.doc_lengths = self.compute_doc_lengths(self.index)
            self.champion_index = self.create_champion_index(self.index, champion_threshold)
            
            self.doc_lenths_for_BM = self.doc_lenths(stemmed_docs)
            
            #dictionary with doc_id and their respective doc_dict dictionary which contains the individual words in the document and its term frequency
            self.TF_values = self.map_doc_by_tf_value(stemmed_docs)
            self.document_frequencies = self.count_doc_frequencies(stemmed_docs)
            
            
    def read_lines(self, filename):
        """ DO NOT MODIFY.
        Read a file to a list of strings. You should not need to modify
        this. """
        return [l.strip() for l in codecs.open(filename, 'r', 'utf-8').readlines()]

    def read_doc (self, filename):
    # reads documents from the TIME.ALL file
        docs = []
        doc_lines = []
        for r in self.read_lines(filename):
            if not (r.startswith("*TEXT") or r.startswith("*STOP")):
                doc_lines.append(r)
            else:
                docs.append(" ".join(doc_lines))
                doc_lines = []
        docs.remove("")
        return docs
   
    def read_query(self,query_filename):
        docs = []
        doc_lines = []
        for r in self.read_lines(query_filename):
            if not (r.startswith("*FIND") or r.startswith("*STOP")):
                doc_lines.append(r)
            else:
                docs.append(" ".join(doc_lines))
                doc_lines = []
                                       
        docs.remove("")
        return docs

    def read_relevant_docs(self,rel_doc_name):
        docs = defaultdict(list)
        for r in self.read_lines(rel_doc_name):
            docs_lines = r.split()
            if len(docs_lines) > 0:
                docs[int(docs_lines[0]) - 1] = [int(x) - 1 for x in docs_lines[1:]]
            
            #docs.remove("")
        return docs


    def compute_doc_lengths(self, index):
        """
        Return a dict mapping doc_id to length, computed as sqrt(sum(w_i**2)),
        where w_i is the tf-idf weight for each term in the document.
        E.g., in the sample index below, document 0 has two terms 'a' (with
        tf-idf weight 3) and 'b' (with tf-idf weight 4). It's length is
        therefore 5 = sqrt(9 + 16).
        >>> lengths = Index().compute_doc_lengths({'a': [[0, 3]], 'b': [[0, 4]]})
        >>> lengths[0]
        5.0
        """
        result = defaultdict(int)
        for i in index.values():
            for j in i:
                result[j[0]] += j[1]**2
        
        for i in result.keys():
            result[i] = math.sqrt(result[i])
        return result             
    
    def create_champion_index(self, index, threshold=10):
        """
        Create an index mapping each term to its champion list, defined as the
        documents with the K highest tf-idf values for that term (the
        threshold parameter determines K).
        In the example below, the champion list for term 'a' contains
        documents 1 and 2; the champion list for term 'b' contains documents 0
        and 1.
        >>> champs = Index().create_champion_index({'a': [[0, 10], [1, 20], [2,15]], 'b': [[0, 20], [1, 15], [2, 10]]}, 2)
        >>> champs['a']
        [[1, 20], [2, 15]]
        >>> champs['b']
        [[0, 20], [1, 15]]
        """
        result = defaultdict(list)
        for item in index.keys():
            temp_list = index[item]
            temp_list = sorted(temp_list, key=lambda x:x[1],reverse=True)
            if len(index[item]) > threshold:
                result[item] = temp_list[:threshold]
            else:
                result[item] = temp_list
        return result

    def create_tfidf_index(self, docs, doc_freqs):
        """
        Create an index in which each postings list contains a list of
        [doc_id, tf-idf weight] pairs. For example:
        {'a': [[0, .5], [10, 0.2]],
         'b': [[5, .1]]}
        This entry means that the term 'a' appears in document 0 (with tf-idf
        weight .5) and in document 10 (with tf-idf weight 0.2). The term 'b'
        appears in document 5 (with tf-idf weight .1).
        Parameters:
        docs........list of lists, where each sublist contains the tokens for one document.
        doc_freqs...dict from term to document frequency (see count_doc_frequencies).
        Use math.log10 (log base 10).
        >>> index = Index().create_tfidf_index([['a', 'b', 'a'], ['a']], {'a': 2., 'b': 1., 'c': 1.})
        >>> sorted(index.keys())
        ['a', 'b']
        >>> index['a']
        [[0, 0.0], [1, 0.0]]0
        
        >>> index['b']  # doctest:+ELLIPSIS
        [[0, 0.301...]]
        """
        
        result_dict = defaultdict(list)
        for id, doc in enumerate(docs):
            doc_dict = defaultdict(int)      # a dictionary mapping all the words of a doc to their frequency in doc 
            for item in doc:
                doc_dict[item] += 1
            #at this point, doc_dict has term freq. for this doc.
            for term in doc_dict:
                calc = (1 + math.log10(doc_dict[term])) * (math.log10(float(len(docs)) / float(doc_freqs[term])))
                result_dict[term].append([id, calc])
        return result_dict
        
    def count_doc_frequencies(self, docs):
        """ Return a dict mapping terms to document frequency.
        >>> res = Index().count_doc_frequencies([['a', 'b', 'a'], ['a', 'b', 'c'], ['a']])
     ds   >>> res['a']
        3
        >>> res['b']
        2
        >>> res['c']
        1
        """
        result = defaultdict(int)
        for lists in docs:
            unique_set = set(lists)
            for j in unique_set:
                result[j] += 1
        return result 
        
        '''# dictionary with the terms in the documents and their document frequncies
        doc_frequencies = defaultdict(int)
    
        for id,doc in enumerate(docs):
            for terms in doc:
                doc_term_freq = defaultdict(int)
                doc_term_freq[terms] += 1      # a counter for no_of_terms in
            for term in doc_term_freq:
                doc_frequencies[term] += 1
        
        return doc_frequencies
        '''
        
    def query_to_vector(self, query_terms):
        """ Convert a list of query terms into a dict mapping term to inverse document frequency.
        Parameters:
        query_terms....list of terms    
        """
        dict = defaultdict(list)
        
       
        for i in query_terms:
            if i in self.doc_freqs:
                #i = self.correct(i)
                dict[i] = math.log10(float(len(self.documents))/float(self.doc_freqs[i]))
            else:
                dict[i] = 0
        return dict 
    
    # make changes here..!!!! errors in this code currently!!!!!!!!!
    
    def compute_RSV(self,query_vector):
        temp = defaultdict(int)
        
        for word in query_vector:
            if word in self.index:
                for matching_word in self.index[word]:
                    temp[matching_word[0]] += query_vector[word]  #query_vector is the list with all the logN/df values      
        RSV_result = sorted([(doc, temp[doc]) for doc in temp], key=lambda x: x[1], reverse=True)
        return RSV_result
       
    def doc_lenths(self,docs):
        doc_length = defaultdict(int)
        '''
        for doc in documents:
            for word in self.tokenize(doc):
                doc_length[doc] += 1 
        return doc_length
        '''
        total_length = 0
        for docid, doc in enumerate(docs):
            doc_length[docid] = len(doc)
            total_length += doc_length[docid]
        self.avg_length_of_docs = total_length / len(docs)
        return doc_length 
        
    def map_doc_by_tf_value(self, docs):
        #maps all the document_ids with their dictionary which maps the document terms with its term frequencies. 
        tf_values = defaultdict(int)
        for doc_id, doc in enumerate(docs):
            doc_dict = defaultdict(int)      # a dictionary mapping all the words of a doc to their frequency in doc 
            for item in doc:
                doc_dict[item] += 1
            tf_values[doc_id] = doc_dict
        return tf_values
    
    def compute_BM25(self,index,query_vector,k,b):
        
        BM_values = defaultdict(int)
        '''
        for doc in self.documents:*
            avg_doc_length += self.doc_lenths_for_BM[doc]
        avg_doc_length /= len(self.doc_lenths_for_BM)
        '''
        for q in query_vector:
            if q in self.index:
                for word in self.index[q]:
                    temp = k * ((1-b) + b * (self.doc_lenths_for_BM[word[0]]/self.avg_length_of_docs)) + self.TF_values[word[0]][q]
                    BM_values[word[0]] = query_vector[q]*(k+1)*self.TF_values[word[0]][q]/temp        
        BM_result = sorted([(doc, BM_values[doc]) for doc in BM_values], key=lambda x: x[1], reverse=True)
        return BM_result
            
    def search_by_cosine(self, query_vector, index, doc_lengths):
        """
        Return a sorted list of doc_id, score pairs, where the score is the
        cosine similarity between the query_vector and the document. The
        document length should be used in the denominator, but not the query
        length (as discussed in class). You can use the built-in sorted method
        (rather than a priority queue) to sort the results.
        The parameters are:
        query_vector.....dict from term to weight from the query
        index............dict from term to list of doc_id, weight pairs
        doc_lengths......dict from doc_id to length (output of compute_doc_lengths)
        In the example below, the query is the term 'a' with weight
        1. Document 1 has cosine similarity of 2, while document 0 has
        similarity of 1.
        >>> Index().search_by_cosine({'a': 1}, {'a': [[0, 1], [1, 2]]}, {0: 1, 1: 1})
        [(1, 2), (0, 1)]
        """
        doc = defaultdict(int)
        for query in query_vector.keys():
            #for list in range(len(index[query])):
            #   doc[index[query]][list][0] += query_vector[query]* index[query][list][1]
            for l in index[query]:
                doc[l[0]] += query_vector[query] * l[1]
        for doc_id in doc:
            doc[doc_id] /= doc_lengths[doc_id]
        
        return sorted([(doc_id,doc[doc_id]) for doc_id in doc], key=lambda x:x[1],reverse=True)
 
    def search(self, query, use_champions=False):
        """ Return the document ids for documents matching the query. Assume that
        query is a single string, possible containing multiple words. Assume
        queries with multiple words are phrase queries. The steps are to:
        1. Tokenize the query (calling self.tokenize)
        2. Stem the query tokens (calling self.stem)
        3. Convert the query into an idf vector (calling self.query_to_vector)
        4. Compute cosine similarity between query vector and each document (calling search_by_cosine).
        Parameters:
        query...........raw query string, possibly containing multiple terms (though boolean operators do not need to be supported)
        use_champions...If True, Step 4 above will use only the champion index to perform the search.
        """
        #l = list()
        #temp_list= self.stem(self.tokenize(query))
        
        
        dict = self.query_to_vector(self.stem(self.tokenize(query)))
        
        list_of_query_terms = self.stem(self.tokenize(query))
        
        
        self.cosine_values = self.search_by_cosine(dict,self.index,self.doc_lengths)
        self.RSV_values = self.compute_RSV(dict)
        self.BM_values_k1_b1 = self.compute_BM25(self.index,dict,1,1)        
        
        self.BM_values_k1_b5 = self.compute_BM25(self.index,dict,1,.5)        
        
        self.BM_values_k2_b1 = self.compute_BM25(self.index,dict,2,1)        
        
        self.BM_values_k2_b5 = self.compute_BM25(self.index,dict,2,.5)        

        '''
        
        if use_champions:
            return self.search_by_cosine(dict,self.champion_index,self.doc_lengths)
        else:
            return self.cosine_values
        '''
        return [self.cosine_values, self.RSV_values, self.BM_values_k1_b1, self.BM_values_k1_b5, self.BM_values_k2_b1, self.BM_values_k2_b5]
         
   
    def tokenize(self, document):
        """ DO NOT MODIFY.
        Convert a string representing one document into a list of
        words. Retain hyphens and apostrophes inside words. Remove all other
        punctuation and convert to lowercase.
        >>> Index().tokenize("Hi there. What's going on? first-class")
        ['hi', 'there', "what's", 'going', 'on', 'first-class']
        """
        return [t.lower() for t in re.findall(r"\w+(?:[-']\w+)*", document)]

    def stem(self, tokens):
        """ DO NOT MODIFY.
        Given a list of tokens, collapse 'did' and 'does' into the term 'do'.
        >>> Index().stem(['did', 'does', 'do', "doesn't", 'splendid'])
        ['do', 'do', 'do', "doesn't", 'splendid']
        """
        return [re.sub('^(did|does)$', 'do', t) for t in tokens]
    
    def word_counts(self,documents):
        result = defaultdict(lambda: 1)
        for doc in documents:
            for word in self.tokenize(doc):
                result[word] += 1
        return result

    def known(self,words):
        return set(w for w in words if w in self.words_count)
    
    def correct(self,word):
        candidates = self.known([word]) or self.known(self.edits(word)) or [word] # 'or' returns whichever is the first non-empty value
        return max(candidates, key=self.words_count.get)

    def edits(self, word):
        splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes    = [a + b[1:] for a, b in splits if b]                       # cat-> ca
        transposes = [a+ b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]  # cat -> act
        replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b] # cat -> car
        inserts    = [a + c + b     for a, b in splits for c in self.alphabet]      # cat -> cats
        return set(deletes + transposes + replaces + inserts)                  # union all edits


def main():
    """ DO NOT MODIFY.
    Main method. Constructs an Index object and runs a sample query. """
    indexer = Index("TIME.ALL")
    queries = indexer.read_query("TIME.QUE")
    relevent_docs = indexer.read_relevant_docs("TIME.REL")
    
    precision_sum_cosine = 0
    precision_sum_rsv = 0
    precision_sum_bm1 = 0
    precision_sum_bm2 = 0
    precision_sum_bm3 = 0
    precision_sum_bm4 = 0
    recall_sum_cosine = 0
    recall_sum_rsv = 0
    recall_sum_bm1 = 0
    recall_sum_bm2 = 0
    recall_sum_bm3 = 0
    recall_sum_bm4 = 0
    f1_sum_cosine = 0
    f1_sum_rsv = 0
    f1_sum_bm1 = 0
    f1_sum_bm2 = 0
    f1_sum_bm3 = 0
    f1_sum_bm4 = 0
    
    print "calculating"
    for q_num, query in enumerate(queries):
        
        #print '\n\nQUERY=', query
        #print "cosine"
        #print "precision:"
        tempp = precision(indexer.search(query)[0][:20], relevent_docs[q_num])
        precision_sum_cosine += tempp
        tempr = recall(indexer.search(query)[0][:20], relevent_docs[q_num])
        recall_sum_cosine += tempr
        if tempp + tempr > 0:
            f1_sum_cosine += (2 * tempp * tempr) / (tempp + tempr)
        
        #print "rsv"
        #print "precision:"
        tempp = precision(indexer.search(query)[1][:20], relevent_docs[q_num])
        precision_sum_rsv += tempp
        tempr = recall(indexer.search(query)[1][:20], relevent_docs[q_num])
        recall_sum_rsv += tempr
        if tempp + tempr > 0:
            f1_sum_rsv += (2 * tempp * tempr) / (tempp + tempr)
        
        
        #print "bm25_1"
        #print "precision:"
        tempp = precision(indexer.search(query)[2][:20], relevent_docs[q_num])
        precision_sum_bm1 += tempp
        tempr = recall(indexer.search(query)[2][:20], relevent_docs[q_num])
        recall_sum_bm1 += tempr
        if tempp + tempr > 0:
            f1_sum_bm1 += (2 * tempp * tempr) / (tempp + tempr)
        
        #print "bm25_2"
        #print "precision:"
        tempp = precision(indexer.search(query)[3][:20], relevent_docs[q_num])
        precision_sum_bm2 += tempp
        tempr = recall(indexer.search(query)[3][:20], relevent_docs[q_num])
        recall_sum_bm2 += tempr
        if tempp + tempr > 0:
            f1_sum_bm2 += (2 * tempp * tempr) / (tempp + tempr)
        
        #print "bm25_3"
        #print "precision:"
        tempp = precision(indexer.search(query)[4][:20], relevent_docs[q_num])
        precision_sum_bm3 += tempp
        tempr = recall(indexer.search(query)[4][:20], relevent_docs[q_num])
        recall_sum_bm3 += tempr
        if tempp + tempr > 0:
            f1_sum_bm3 += (2 * tempp * tempr) / (tempp + tempr)
        
        #print "bm25_4"
        #print "precision:"
        precision_sum_bm4 += precision(indexer.search(query)[5][:20], relevent_docs[q_num])
        recall_sum_bm4 += recall(indexer.search(query)[5][:20], relevent_docs[q_num])

        tempp = precision(indexer.search(query)[5][:20], relevent_docs[q_num])
        precision_sum_bm4 += tempp
        tempr = recall(indexer.search(query)[5][:20], relevent_docs[q_num])
        recall_sum_bm4 += tempr
        if tempp + tempr > 0:
            f1_sum_bm4 += (2 * tempp * tempr) / (tempp + tempr)
        
    print "cosine"
    print "avg precision:",
    print precision_sum_cosine / float(len(queries))
    print "avg recall:",
    print recall_sum_cosine / float(len(queries))
    print "avg f1:",
    print f1_sum_cosine / float(len(queries))
        
    print "rsv"
    print "avg precision:",
    print precision_sum_rsv / float(len(queries))
    print "avg recall:",
    print recall_sum_rsv / float(len(queries))
    print "avg f1:",
    print f1_sum_rsv / float(len(queries))
        
    print "bm25_1"
    print "avg precision:",
    print precision_sum_bm1 / float(len(queries))
    print "avg recall:",
    print recall_sum_bm1 / float(len(queries))
    print "avg f1:",
    print f1_sum_bm1 / float(len(queries))
        
    print "bm25_2"
    print "avg precision:",
    print precision_sum_bm2 / float(len(queries))
    print "avg recall:",
    print recall_sum_bm2 / float(len(queries))
    print "avg f1:",
    print f1_sum_bm2 / float(len(queries))
        
    print "bm25_3"
    print "avg precision:",
    print precision_sum_bm3 / float(len(queries))
    print "avg recall:",
    print recall_sum_bm3 / float(len(queries))
    print "avg f1:",
    print f1_sum_bm3 / float(len(queries))
        
    print "bm25_4"
    print "avg precision:",
    print precision_sum_bm4 / float(len(queries))
    print "avg recall:",
    print recall_sum_bm4 / float(len(queries))
    print "avg f1:",
    print f1_sum_bm4 / float(len(queries))
    


def precision(search_results, relevant_docs):
    tp = 0
    for result in search_results:
        if result[0] in relevant_docs:
            tp += 1
    return float(tp) / float(len(search_results)) 

def recall(search_results,relevant_docs):
    temp = [doc_id[0] for doc_id in search_results] 
    found = 0
    for r in relevant_docs:
        if r in temp:
            found += 1
    return float(found)/float(len(relevant_docs)) 
    
    
if __name__ == '__main__':
    main()









