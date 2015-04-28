""" Assignment 6: PageRank. """
from collections import defaultdict
import glob
from BeautifulSoup import BeautifulSoup


def parse(folder, inlinks, outlinks):
    """
    Read all .html files in the specified folder. Populate the two
    dictionaries inlinks and outlinks. inlinks maps a url to its set of
    backlinks. outlinks maps a url to its set of forward links.
    """
    pathnames = glob.glob(folder + '/*.html')
    for path in pathnames:
        '''print "Inside first for loop" '''
        soup = BeautifulSoup(open(path))
        path = path.replace("\\", "/")
        temp_inlinks = list()
        temp_outlinks = list()
        for link in soup.findAll('a'):
            temp_outlinks.append(folder + "/" + link.get('href'))
        #print "PATH " + path 
        outlinks[path] = set(temp_outlinks)
        for x in outlinks[path]: 
            if path not in inlinks[x]:
                inlinks[x].add(path)
    
def compute_pagerank(urls, inlinks, outlinks, b=.85, iters=20):
    """ Return a dictionary mapping each url to its PageRank.
    The formula is R(u) = 1-b + b * (sum_{w in B_u} R(w) / (|F_w|)
    Initialize all scores to 1.0
    """
    pagerank_results = defaultdict(lambda: 1)
    for i in range(iters):
        for link in urls:
            rank = 0
            sum_of_backlinks = 0
            for i in inlinks[link]:
                sum_of_backlinks += pagerank_results[i]/ len(outlinks[i])
            rank = (1 - b) + (b * sum_of_backlinks)   
            pagerank_results[link] = rank
    return pagerank_results
        
def run(folder, b):
    """ Do not modify this function. """
    inlinks = defaultdict(lambda: set())
    outlinks = defaultdict(lambda: set())
    parse(folder, inlinks, outlinks)
    urls = sorted(set(inlinks) | set(outlinks))
    ranks = compute_pagerank(urls, inlinks, outlinks, b=b)
    print 'Result for', folder, '\n', '\n'.join('%s\t%.3f' % (url, ranks[url]) for url in sorted(ranks))


def main():
    """ Do not modify this function. """
    run('set1', b=.5)
    run('set2', b=.85)
    
if __name__ == '__main__':
    main()