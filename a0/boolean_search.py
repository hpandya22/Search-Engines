""" Assignment 0

You will implement a simple in-memory boolean search engine over the jokes
from http://web.hawkesnest.net/~jthens/laffytaffy/.

The documents are read from documents.txt.
The queries to be processed are read from queries.txt.

Your search engine will only need to support AND queries. A multi-word query
is assumed to be an AND of the words. E.g., the query "why because" should be
processed as "why AND because."
"""

# Some imports you may want to use.
from collections import defaultdict
import re


def read_lines(filename):
    """ Read a file to a list of strings. You should not need to modify
    this. """
    return [l.strip() for l in open(filename, 'rt').readlines()]


def tokenize(document):
    """ Convert a string representing one document into a list of
    words. Remove all punctuation and split on white space.
    >>> tokenize("Hi there. What's going on?")
    ['hi', 'there', 'what', 's', 'going', 'on']
    """
    return re.findall('\w+', document.lower())


def create_index(tokens):
    """
    Create an inverted index given a list of document tokens. The index maps
    each unique word to a list of document ids, sorted in increasing order.
    >>> index = create_index([['a', 'b'], ['a', 'c']])
    >>> sorted(index.keys())
    ['a', 'b', 'c']
    >>> index['a']
    [0, 1]
    >>> index['b']
    [0]
    >>> index['c']
    [1]
    """
    d = dict()                                           # the dictionary to index the documents
    for i in range (len(tokens)):                                    # given list of document tokens
        for w in tokens[i]:
            if d.has_key(w):                              # if the word already exists and matches with the doc token t
                if i not in d[w]:
				    d[w].append(i)               			# if the word already exists in some doc , then append the list with the new doc's index
            else:
                d.update({w:[i]})                           # if the doc is found the first time then create a new list for that word with this new found index
    return d
    
def intersect(list1, list2):
    """ Return the intersection of two posting lists. Use the optimize
    algorithm of Figure 1.6 of the MRS text.
    >>> intersect([1, 3, 5], [3, 4, 5, 10])
    [3, 5]
    >>> intersect([1, 2], [3, 4])
    []
    """
    l = []
    p1 = 0
    p2 = 0
    while p1 < len(list1) and p2 < len(list2):
        if list1[p1] == list2[p2]:
			l.append(list1[p1])
			p2 = p2 + 1
			p1 = p1 + 1 
        elif list1[p1] > list2[p2]:
			p2 = p2 + 1
        else:
			p1 = p1 + 1
    return l
    
def sort_by_num_postings(words, index):
    """
    Sort the words in increasing order of the length of their postings list in
    index.
    >>> sort_by_num_postings(['a', 'b', 'c'], {'a': [0, 1], 'b': [1, 2, 3], 'c': [4]})
    ['c', 'a', 'b']
    """
    words = sorted(words,key= lambda i : len(index[i]))
    return words
   
    
def search(index, query):
    """ Return the document ids for documents matching the query. Assume that query is a single string, possible containing multiple words. The steps are to:
    1. tokenize the query
    2. Sort the query words by the length of their postings list
    3. Intersect the postings list of each word in the query.
    E.g., below we search for documents containing 'a' and 'b':
    >>> search({'a': [0, 1], 'b': [1, 2, 3], 'c': [4]}, 'a b')
    [1]
    """
    query_words_in_order = sort_by_num_postings(tokenize(query),index)
    final_list = index[query_words_in_order[0]] 
    for i in range (len(query_words_in_order)):
        final_list = intersect(final_list,index[query_words_in_order[i]])
    return final_list

def main():
    """ Main method. You should not modify this. """
    documents = read_lines('documents.txt')
    tokens = [tokenize(d) for d in documents]
    index = create_index(tokens)
    queries = read_lines('queries.txt')
    
    ind = create_index([['a', 'b'], ['a', 'c']])
    print ind 
    print type(ind['c'])
    for query in queries:
        results = search(index, query)
        print '\n\nQUERY:', query, '\nRESULTS:\n', '\n'.join(documents[r] for r in results)
	
    #print create_index(tokenize("Hi there. What's going on?"))
    #print intersect([1, 3, 5], [3, 4, 5, 10])
    #print sort_by_num_postings(['a', 'b', 'c'], {'a': [0, 1], 'b': [1, 2, 3], 'c': [4]})
    #print search({'a': [0, 1], 'b': [1, 2, 3], 'c': [4]}, 'a b')

if __name__ == '__main__':
    main()
