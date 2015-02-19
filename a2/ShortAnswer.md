Edit this file and push to your private repository to provide answers to the following questions.

1. In `searcher.py`, why do we keep an inverted index instead of simply a list
of document vectors (e.g., dicts)? What is the difference in time and space
complexity between the two approaches?

    **Insert your answer here.**
    A document vector gives list of all the unique words it has but an inverted index maps all the unique terms to the document they occur in. 
    So that saves the time of looking through all the documents to find the query terms. By using inverted index we look for only those documents 
    that contains the query words. 
    
    In case of searching through document vectors the time complexity depends on the number of document vectors(total no of docs, N) times the unique terms in each vector (length of individual vector, d) = N*d . 
    note that d here is going to repeat the unique words visited across list of documents. 
   
    On the contrary in case of inverted index , the search is limited to the a given word which is constant time in a dictionary (1) and then its just matter of going through 
    the list of documents relevant to that word (length of each list).
    
    space complexity for document vector would be total no of documents (N) times the no of unique words in each document (n) = N*n
    in case of inverted index space complexit would be total no of unique words (w) times the length of the list of documents they occur assuming it being (l) the total space required would be = w(l1+l2+l3+...ln) 
    here the length of each list for a word varies greatly. it depends on how common or rare a word is. for the worst case scenario a word may occur in all the documents in a given list. which would then equal the space to = w*N (N= total no of documents)
    
    
2. Consider the query `chinese` with and without using champion lists.  Why is
the top result without champion lists absent from the list that uses champion
lists? How can you alter the algorithm to fix this?

    **Insert your answer here.**
   
    In case of champion list we narrow down the documents with highest tf-idf for a specific word , at that point the length of the document is not taken into consideration.
    In case of regular index , the list of document is finalized after finding the cosine similarity of the document to the query which takes the document's length in consideration.
    hence documents with significantly short size tend to have higher cosine similarity or lower cosine similarity in case of very long documents which affects their likeliness in 
    the relevant list. Hence in order to solve it, we can divide the documents' tf-idf values with their document lengths when computing champion list, Which gives us the accurate set of documents that are relevant.
    
3. Describe in detail the data structures you would use to implement the
Cluster Pruning approach, as well as how you would use them at query time.

    **Insert your answer here.**
    
    In order to implement Cluster Pruning approach we could either use a list of lists or a dictionary 
    dictionary: in case of dictionary the leader of the cluster can be mapped to the list of nodes of that cluster..
    List : a list contains sublists, where each sublist stands for each cluster in the system. each sublist's first element represents the leader of that cluster 
            and the rest of the elements would be the nodes in the cluster... 