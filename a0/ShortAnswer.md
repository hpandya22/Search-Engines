-Edit this file in your private repository to provide answers to the following questions (from MRS).

1. Extend the postings merge algorithm to arbitrary Boolean query formulas. What is
its time complexity? For instance, consider:

  `(Brutus OR Caesar) AND NOT (Antony OR Cleopatra)`

  Can we always merge in linear time? Linear in what? Can we do better than this?

  **Insert your answer here.**
    As We do boolean operations we perform the merge algorithm on the given two lists, hence the time complexity would be multiplied by number of boolean operations
    performed on it. Although by multiplying it stays the same linear form. So mainly the time complexity depends on the amount of documents and the number of operations/comparisons. 
    And it is almost impossible to minimize that. Hence it will always be linear. 


    
    2. If the query is:

  `friends AND romans AND (NOT countrymen)`

  How could we use the frequency of countrymen in evaluating the best query evaluation order? In particular, propose a way of handling negation in determining the order of query processing.
  
  **Insert your answer here.**
  In case of NOT joins, the higher frequency of countrymen decreases the pool of data to compare decreases and hence it creates a smaller set of data to compare to friends 
  hence the higher frequency of countrymen the smaller is the overall amount of comparisons performed for the other AND operations. So overall it makes it optimal approach. 


  
  
  
  3. For a conjunctive query, is processing postings lists in order of size guaranteed to be
optimal? Explain why it is, or give an example where it isn’t.

  **Insert your answer here.**
    In case of conjunctive query the by processing in order will narrow down the sample of data that is highly likely to be in the pool of result than 
    the other data present in a larger files. The result is always going to be involve either equal or less than the data present in the smallest list. 
    Hence this way the lesser amount of data decreases the number of comparisons that are going to be non productive towards achieving the resulting list 
    and will result in the optimal path to process the lists. 