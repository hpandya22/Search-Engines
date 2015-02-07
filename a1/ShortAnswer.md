Edit this file in your private repository to provide answers to the following questions.

1. Consider merging the following two lists, where the first list has skip pointers.
![skips](skips.png)
  1. How often is a skip pointer followed (i.e., p1 is advanced to skip(p1))?

    **Insert your answer here.**
    In the given example only the second skip pointer is followed, 
    as the other skips are either too early to take or they skip to a much higher value than desired.
  
  2. How many postings comparisons will be made by this algorithm while intersecting the two lists?

    **Insert your answer here.**
    cmp (3,3) + cmp (5,5)
    cmp (89,9) + cmp (89,15)
    cmp (89,24) + [skippping from 24 -> 75]
     + [don't take the skip 75 -> 92]
    cmp(89,81) 
    cmp (89,84) + cmp(89,89)
    cmp(95,92) + [don't take the skip])
    cmp (95,96) + cmp(97,96) + cmp(97,97)
    cmp(99,100) + cmp (100,100)
    cmp (101,115) 
    = 18 
      




    
  3. How many postings comparisons would be made if the postings lists are intersected without the use of skip pointers?

    **Insert your answer here.**
    
    cmp (3,3) + cmp (5,5)
    cmp (89,9) + cmp (89,15)
    cmp (89,24) + cmp(89,39)
    cmp(89,60) + cmp(89,75)
    cmp(89,81) 
    cmp (89,84) + cmp(89,89)
    cmp(95,92) + 
    cmp (95,96) + cmp(97,96) + cmp(97,97)
    cmp(99,100) + cmp (100,100)
    cmp (101,115) 
       
    
2. Compute the Levenshtein edit distance between *paris* and *alice*. Fill in the 5 × 5 table below of
distances between all preﬁxes as computed by the algorithm in Figure 3.5 in [MRS](http://nlp.stanford.edu/IR-book/pdf/03dict.pdf). Cell (*i*, *j*) should store the minimum edit distance between the first *i* characters of *alice* and the first *j* characters of *paris* (as in the bottom right number of each cell in Figure 3.6).

  |       |   | p | a | r | i | s |
  |-------|---|---|---|---|---|---|
  |       | 0 | 1 | 2 | 3 | 4 | 5 |
  | **a** | 1 | 1 | 1 | 2 | 3 | 4 |
  | **l** | 2 | 2 | 2 | 2 | 3 | 4 |
  | **i** | 3 | 3 | 3 | 3 | 2 | 3 |
  | **c** | 4 | 4 | 4 | 4 | 3 | 3 |
  | **e** | 5 | 5 | 5 | 5 | 4 | 4 |
  
  

3. (Inspired by [H Schütze](http://www.cis.uni-muenchen.de/~hs/teach/13s/ir/).)We define a *hapax legomenon* as a term that occurs exactly once in a collection. We want to estimate the number of hapax legomena using Heaps’ law and Zipf’s law.
    1. How many unique terms does a web collection of 400,000,000 web pages containing 400 tokens on average have? Use the Heaps parameters k = 100 and b = 0.5.
    2. Use Zipf’s law to estimate the proportion of the term vocabulary of the collection that consists of hapax legomena. You may want to use the approximation 1/1 + 1/2 + ... + 1/*n* = ln *n*
    3. Do you think that the estimate you get is correct? Why or why not?

  **Insert your answer here.**
    1. V = k(T)^b 
         = 100 (400 * 400,000,000)^0.5
         = 100 * 4 * 100,000 
         = 40,000,000
    2.  f(i) = 1/2, + 1 / 3 + 1 / 4 + 1/ 5
             
    
    3.   It is not a good estimate. The least frequent words which may not have a good credibility and 
    