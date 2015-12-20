#
# Utility functions for analysis used in the following two articles.
#
# [Reference]
# Jiepu Jiang and James Allan. Correlation between system and user metrics in a session.
# In Proceedings of the first ACM SIGIR Conference on Human Information Interaction and Retrieval (CHIIR '16),
# Chapel Hill, North Carolina, USA, 2016.
# http://people.cs.umass.edu/~jpjiang/papers/chiir16_metrics.pdf
#
# Jiepu Jiang and James Allan. Adaptive effort for search evaluation metrics.
# In Proceedings of the 38th European Conference on Information Retrieval (ECIR '16), 2016
# http://people.cs.umass.edu/~jpjiang/papers/ecir16_metrics.pdf


import random
import scipy.stats as stats


#
# Compute Pearson's r and Spearman's rho of umetric and smetric on a few sessions.
#
# sratings      sessions' user ratings (ground truth)
# sresults      sessions' search results
# sqrels        sessions' qrels
# umetric       the user experience metric, either 'performance' or 'difficulty' in this dataset
# smetric       the system-oriented metric
# k             the top k results of each query to be evaluated by smetric
def correlation(sratings, sresults, sqrels, umetric, smetric, k):
    ratings = []
    sevals = []
    for sessid in sresults.keys():
        ratings.append(sratings[sessid][umetric])
        sevals.append(smetric.evaluate(sqrels[sessid], sresults[sessid], k))
    pearson, p_pearson = stats.pearsonr(ratings, sevals)
    spearman, p_spearman = stats.spearmanr(ratings, sevals)
    return pearson, p_pearson, spearman, p_spearman


#
# Regress umetric using smetric on a few sessions.
#
# sratings      sessions' user ratings (ground truth)
# sresults      sessions' search results
# sqrels        sessions' qrels
# umetric       the user experience metric, either 'performance' or 'difficulty' in this dataset
# smetric       the system-oriented metric
# k             the top k results of each query to be evaluated by smetric
# norm          the maximum possible user rating difference, i.e., max(rating) - min(rating)
# numfolds      the number of folds x to perform x-fold cross validation, where (x-1) folds are used for training
# numsamples    the number of random partitions of the dataset to be generated; each partition will be evalauted using
#               x-fold cross validation
# seed          the seed used for generating random numbers
def regress(sratings, sresults, sqrels, umetric, smetric, k, norm, numfolds, numsamples, seed=0):
    nrmse = []
    random.seed(seed)
    for i in xrange(0, numsamples):
        sessionlist = [sessid for sessid in sresults.keys()]
        random.shuffle(sessionlist)
        for foldid in xrange(0, numfolds):
            train, test = [], []
            for ix in xrange(0, len(sessionlist)):
                if ix % numfolds == foldid:
                    test.append(sessionlist[ix])
                else:
                    train.append(sessionlist[ix])
            nrmse.append(regress_fold(sratings, sresults, sqrels, umetric, smetric, k, norm, train, test))
    return nrmse


#
# Regress umetric using smetric on the specified train & test sessions.
#
# sratings      sessions' user ratings (ground truth)
# sresults      sessions' search results
# sqrels        sessions' qrels
# umetric       the user experience metric, either 'performance' or 'difficulty' in this dataset
# smetric       the system-oriented metric
# k             the top k results of each query to be evaluated by smetric
# norm          the maximum possible user rating difference, i.e., max(rating) - min(rating)
# train         a list of training sessions' sessids
# test          a list of testing sessions' sessids
def regress_fold(sratings, sresults, sqrels, umetric, smetric, k, norm, train, test):
    ratings = []
    sevals = []
    for sessid in train:
        ratings.append(sratings[sessid][umetric])
        sevals.append(smetric.evaluate(sqrels[sessid], sresults[sessid], k))
    slope, intercept, _, _, _ = stats.linregress(sevals, ratings)
    sum_se = 0
    for sessid in test:
        rating = sratings[sessid][umetric]
        seval = smetric.evaluate(sqrels[sessid], sresults[sessid], k)
        rating_pred = slope * seval + intercept
        sum_se += (rating - rating_pred) ** 2
    return (sum_se / len(test)) ** 0.5 / norm


#
# Get stars for the provided p value.
# *, **, and *** indicate 0.05, 0.01, and 0.001 levels of significance, respectively.
def star(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ''


#
# Get the first element of the array.
def first(array):
    return array[0]


#
# Get the last element of the array.
def last(array):
    return array[len(array) - 1]
