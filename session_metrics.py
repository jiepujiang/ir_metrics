import math
import random


# sDCG.
class SDCG:
    # b             rank discount parameter
    # bq            query discount parameter
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


# nsDCG.
class NSDCG:
    # b             rank discount parameter
    # bq            query discount parameter
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


# sDCG/q.
class SDCGQ:
    # b             rank discount parameter
    # bq            query discount parameter
    # discountq     whether or not to apply query discount
    def __init__(self, b, bq, discountq):
        self.b = b
        self.bq = bq
        self.discountq = discountq

    def evaluate(self, qrels, sresults, k):
        sdcg = SDCG(self.b, self.bq, self.discountq)
        return sdcg.evaluate(qrels, sresults, k) / len(sresults)


# nsDCG.
class ESNDCG:
    # b             rank discount parameter
    # bq            query discount parameter
    # discountq     whether or not to apply query discount
    def __init__(self, pref, pdown, normScanPath, N=10000):
        self.pref = pref
        self.pdown = pdown
        self.normScanPath = normScanPath
        self.N = N

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

    def evaluate(self, qrels, sresults, k):
        ideal_list = sorted(qrels, key=lambda key: qrels[key], reverse=True)
        sum_sample = 0
        for i in xrange(0, self.N):
            scanpath = self.sample(sresults, k)
            dcg = self.dcg(qrels, scanpath, len(scanpath))
            idcg = self.dcg(qrels, ideal_list, len(scanpath))
            sum_sample += dcg / idcg
        return sum_sample / self.N


# statistics of individual queries' scores
class SQMetric:
    # qmetric       the query level metric
    # aggfunc       aggregation function used to derive session score from query scores, e.g., np.mean
    def __init__(self, qmetric, aggfunc):
        self.qmetric = qmetric
        self.aggfunc = aggfunc

    def evaluate(self, qrels, sresults, k):
        qscores = []
        for results in sresults:
            qscores.append(self.qmetric.evaluate(qrels, results, k))
        return self.aggfunc(qscores)
