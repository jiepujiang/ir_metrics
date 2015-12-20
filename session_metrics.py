#
# Metrics for evaluating a search session (multiple queries)'s quality.
#
# [Reference]
# Jiepu Jiang and James Allan. Correlation between system and user metrics in a session.
# In Proceedings of the first ACM SIGIR Conference on Human Information Interaction and Retrieval (CHIIR '16),
# Chapel Hill, North Carolina, USA, 2016.
#
# http://people.cs.umass.edu/~jpjiang/papers/chiir16_metrics.pdf
#

import math
import random


#
# sDCG.
#
# [Reference]
# Kalervo Jarvelin, Susan L. Price, Lois M. L. Delcambre, and Marianne Lykke Nielsen. 2008.
# Discounted cumulated gain based evaluation of multiple-query IR sessions.
# In Proceedings of the IR research, 30th European conference on Advances in information retrieval (ECIR'08),
# Craig Macdonald, Iadh Ounis, Vassilis Plachouras, Ian Ruthven, and Ryen W. White (Eds.).
# Springer-Verlag, Berlin, Heidelberg, 4-15.
class SDCG:
    #
    # b             the rank discount parameter
    # bq            the query discount parameter
    # discountq     whether or not to apply query discount
    def __init__(self, b, bq, discountq):
        self.b = b
        self.bq = bq
        self.discountq = discountq

    def evaluate(self, qrels, sresults, k):
        sdcg = 0
        for qix in xrange(0, len(sresults)):
            qdiscount = math.log(self.bq, qix + self.bq)
            sum_gain, rank = 0.0, 1
            for doc in sresults[qix]:
                rel = qrels.get(doc, 0)
                gain = 2 ** rel - 1.0
                discount = math.log(self.b, rank + self.b - 1)
                sum_gain += gain * discount
                rank += 1
                if rank > k:
                    break
            if self.discountq:
                sdcg += qdiscount * sum_gain
            else:
                sdcg += sum_gain
        return sdcg


#
# Normalized sDCG.
#
# [Reference]
# Evangelos Kanoulas, Ben Carterette, Paul D. Clough, and Mark Sanderson. 2011. Evaluating multi-query sessions.
# In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval
# (SIGIR '11). ACM, New York, NY, USA, 1053-1062. DOI=http://dx.doi.org/10.1145/2009916.2010056
class NSDCG:
    #
    # b             the rank discount parameter
    # bq            the query discount parameter
    # discountq     whether or not to apply query discount
    def __init__(self, b, bq, discountq):
        self.b = b
        self.bq = bq
        self.discountq = discountq

    def evaluate(self, qrels, sresults, k):
        ideal_list = sorted(qrels, key=lambda key: qrels[key], reverse=True)
        ideal_session = [ideal_list for i in xrange(0, len(sresults))]
        sdcg = SDCG(self.b, self.bq, self.discountq)
        return sdcg.evaluate(qrels, sresults, k) / sdcg.evaluate(qrels, ideal_session, k)


#
# sDCG/q: a metric that normalizes sDCG by simply the number of queries in a session.
class SDCGQ:
    #
    # b             the rank discount parameter
    # bq            the query discount parameter
    # discountq     whether or not to apply query discount
    def __init__(self, b, bq, discountq):
        self.b = b
        self.bq = bq
        self.discountq = discountq

    def evaluate(self, qrels, sresults, k):
        sdcg = SDCG(self.b, self.bq, self.discountq)
        return sdcg.evaluate(qrels, sresults, k) / len(sresults)


#
# Estimated session nDCG.
#
# [Reference]
# Evangelos Kanoulas, Ben Carterette, Paul D. Clough, and Mark Sanderson. 2011. Evaluating multi-query sessions.
# In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval
# (SIGIR '11). ACM, New York, NY, USA, 1053-1062. DOI=http://dx.doi.org/10.1145/2009916.2010056
class ESNDCG:
    #
    # pref              the probability to reformulate to the next query after examining a query's SERP
    # pdown             the probability to examine the next result in a ranked list
    # path_discount     whether to discount lower ranked results in a scan path
    # N                 the number of sampling iteration
    def __init__(self, pref, pdown, path_discount, N=1000):
        self.pref = pref
        self.pdown = pdown
        self.normScanPath = path_discount
        self.N = N

    #
    # compute dcg of a ranked list until some cutoff k
    def dcg(self, qrels, results, k):
        dcg, rank = 0.0, 1
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = 2 ** rel - 1.0
            discount = math.log(2, rank + 1)
            if self.normScanPath:
                dcg += gain * discount
            else:
                dcg += gain
            rank += 1
            if rank > k:
                break
        return dcg

    #
    # sample a scan path by pref and pdown
    def sample(self, sresults, k):
        scanpath = []
        for results in sresults:
            rank = 1
            for doc in results:
                scanpath.append(doc)
                if random.random() >= self.pdown:
                    break
                rank += 1
                if rank > k:
                    break
            if random.random() >= self.pref:
                break
        return scanpath

    #
    # estimate esnDCG by sampling
    def evaluate(self, qrels, sresults, k):
        ideal_list = sorted(qrels, key=lambda key: qrels[key], reverse=True)
        sum_sample = 0
        for i in xrange(0, self.N):
            scanpath = self.sample(sresults, k)
            dcg = self.dcg(qrels, scanpath, len(scanpath))
            idcg = self.dcg(qrels, ideal_list, len(scanpath))
            sum_sample += dcg / idcg
        return sum_sample / self.N


#
# SQMetric aggregates individual queries' scores to evaluate a session.
class SQMetric:
    #
    # qmetric       the metric used to evaluate each individual query
    # aggfunc       the aggregation function used to derive session score from a list of query scores, e.g., np.mean
    def __init__(self, qmetric, aggfunc):
        self.qmetric = qmetric
        self.aggfunc = aggfunc

    def evaluate(self, qrels, sresults, k):
        qscores = []
        for results in sresults:
            qscores.append(self.qmetric.evaluate(qrels, results, k))
        return self.aggfunc(qscores)
