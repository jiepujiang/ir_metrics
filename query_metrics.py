#
# Metrics and their adaptive effort variant for evaluating a SERP (a ranked list of results)'s quality.
#
# [Reference]
# Jiepu Jiang and James Allan. Adaptive effort for search evaluation metrics.
# In Proceedings of the 38th European Conference on Information Retrieval (ECIR '16), 2016
#
# http://people.cs.umass.edu/~jpjiang/papers/ecir16_metrics.pdf
#

import math


#
# P@k.
class Prec:
    #
    # evec      the effort vector
    def __init__(self, evec):
        self.evec = evec

    def evaluate(self, qrels, results, k):
        sum_gain, sum_effort, rank = 0.0, 0.0, 1
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = rel > 0
            effort = self.evec[rel]
            sum_gain += gain
            sum_effort += effort
            rank += 1
            if rank > k:
                break
        if sum_gain == 0:
            return 0
        return sum_gain / sum_effort


#
# Graded relevance P@k, where grade relevance is handled as the same as in graded average precision (GAP).
class GradPrec:
    #
    # evec      the effort vector
    # gs        the probability that users will consider results with each relevance grade as relevant.
    #           for example, gs = [0, 0.4, 0.6] means that users have:
    #               0 probability to consider r>=0 as relevant
    #               0.4 probability to consider r>=1 as relevant
    #               0.6 probability to consider r>=2 as relevant
    def __init__(self, evec, gs):
        self.evec = evec
        self.gs = gs

    def evaluate(self, qrels, results, k):
        sum_gain, sum_effort, rank = 0.0, 0.0, 1
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = sum(self.gs[r] for r in range(0, rel + 1))
            effort = self.evec[rel]
            sum_gain += gain
            sum_effort += effort
            rank += 1
            if rank > k:
                break
        if sum_gain == 0:
            return 0
        return sum_gain / sum_effort


#
# DCG@k (the exponential gain version).
#
# [reference]
# Kalervo Jarvelin and Jaana Kekalainen. 2000. IR evaluation methods for retrieving highly relevant documents.
# In Proceedings of the 23rd annual international ACM SIGIR conference on Research and development in
# information retrieval (SIGIR '00). ACM, New York, NY, USA, 41-48. DOI=http://dx.doi.org/10.1145/345508.345545
class DCG:
    #
    # evac      the effort vector
    def __init__(self, evec):
        self.evec = evec

    def evaluate(self, qrels, results, k):
        sum_gain, sum_effort, rank = 0.0, 0.0, 1
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = 2 ** rel - 1.0
            effort = self.evec[rel]
            discount = math.log(2, rank + 1)
            sum_gain += gain * discount
            sum_effort += effort * discount
            rank += 1
            if rank > k:
                break
        if sum_gain == 0:
            return 0
        return sum_gain / sum_effort


#
# nDCG@k (the exponential gain version).
#
# [reference]
# Kalervo Jarvelin and Jaana Kekalainen. 2002. Cumulated gain-based evaluation of IR techniques.
# ACM Trans. Inf. Syst. 20, 4 (October 2002), 422-446. DOI=http://dx.doi.org/10.1145/582415.582418
class NDCG:
    #
    # evac      the effort vector
    def __init__(self, evec):
        self.evec = evec

    def evaluate(self, qrels, results, k):
        dcg = DCG(self.evec)
        ideal_list = sorted(qrels, key=lambda key: qrels[key], reverse=True)
        dcg_results = dcg.evaluate(qrels, results, k)
        dcg_ideal = dcg.evaluate(qrels, ideal_list, k)
        if dcg_results == 0:
            return 0
        return dcg_results / dcg_ideal


#
# RBP.
#
# [reference]
# Alistair Moffat and Justin Zobel. 2008. Rank-biased precision for measurement of retrieval effectiveness.
# ACM Trans. Inf. Syst. 27, 1, Article 2 (December 2008), 27 pages. DOI=http://dx.doi.org/10.1145/1416950.1416952
class RBP:
    #
    # evac      the effort vector
    # pdown     the probability to examine the next result
    def __init__(self, evec, pdown):
        self.evec = evec
        self.pdown = pdown

    def evaluate(self, qrels, results, k):
        sum_gain, sum_effort, rank, pexam = 0.0, 0.0, 1, 1.0
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = rel > 0
            effort = self.evec[rel]
            sum_gain += gain * pexam
            sum_effort += effort * pexam
            rank += 1
            pexam *= self.pdown
            if rank > k:
                break
        if sum_gain == 0:
            return 0
        return sum_gain / sum_effort


#
# A graded relevance variant for RBP. Graded relevance is handled in the same way as in graded average precision (GAP).
class GRBP:
    #
    # evac      the effort vector
    # pdown     the probability to examine the next result
    # gs        the probability that users will consider results with each relevance grade as relevant.
    #           for example, gs = [0, 0.4, 0.6] means that users have:
    #               0 probability to consider r>=0 as relevant
    #               0.4 probability to consider r>=1 as relevant
    #               0.6 probability to consider r>=2 as relevant
    def __init__(self, evec, pdown, gs):
        self.evec = evec
        self.pdown = pdown
        self.gs = gs

    def evaluate(self, qrels, results, k):
        sum_gain, sum_effort, rank, pexam = 0.0, 0.0, 1, 1.0
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = sum(self.gs[r] for r in range(0, rel + 1))
            effort = self.evec[rel]
            sum_gain += gain * pexam
            sum_effort += effort * pexam
            rank += 1
            pexam *= self.pdown
            if rank > k:
                break
        if sum_gain == 0:
            return 0
        return sum_gain / sum_effort


#
# Average precision.
class AvgPrec:
    #
    # evac      the effort vector
    def __init__(self, evec):
        self.evec = evec

    def evaluate(self, qrels, results, k):
        sum_prec, sum_gain, sum_effort, rank = 0.0, 0.0, 0.0, 1
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = rel > 0
            effort = self.evec[rel]
            sum_gain += gain
            sum_effort += effort
            if rel > 0:
                sum_prec += sum_gain / sum_effort
            rank += 1
            if rank > k:
                break
        if sum_prec == 0:
            return 0
        numrel = sum(rel > 0 for rel in qrels.itervalues())
        return sum_prec / numrel


#
# Graded average precision.
#
# [reference]
# Stephen E. Robertson, Evangelos Kanoulas, and Emine Yilmaz. 2010.
# Extending average precision to graded relevance judgments.
# In Proceedings of the 33rd international ACM SIGIR conference on Research and development in
# information retrieval (SIGIR '10). ACM, New York, NY, USA, 603-610. DOI=http://dx.doi.org/10.1145/1835449.1835550
class GradAvgPrec:
    #
    # evac      the effort vector
    # gs        relevance threshold probability
    #           for example, [0, 0.4, 0.6] means that users have:
    #               0 probability to consider r>=0 as relevant
    #               0.4 probability to consider r>=1 as relevant
    #               0.6 probability to consider r>=2 as relevant
    def __init__(self, evec, gs):
        self.evec = evec
        self.gs = gs

    def evaluate(self, qrels, results, k):
        sum_prec, sum_gain, sum_effort, rank = 0.0, 0.0, 0.0, 1
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = sum(self.gs[r] for r in range(0, rel + 1))
            effort = self.evec[rel]
            sum_gain += gain
            sum_effort += effort
            if rel > 0:
                sum_prec += sum_gain / sum_effort
            rank += 1
            if rank > k:
                break
        if sum_prec == 0:
            return 0
        enumrel = 0.0
        for rel in qrels.values():
            erel = 0.0
            for r in range(0, rel + 1):
                erel += self.gs[r]
            enumrel += erel
        enumrel = sum(sum(self.gs[r] for r in range(0, rel + 1)) for rel in qrels.itervalues())
        return sum_prec / enumrel


#
# Reciprocal rank.
class RR:
    #
    # evac      the effort vector
    def __init__(self, evec):
        self.evec = evec

    def evaluate(self, qrels, results, k):
        sum_gain, sum_effort, rank = 0.0, 0.0, 1
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = rel > 0
            effort = self.evec[rel]
            sum_gain += gain
            sum_effort += effort
            if rel > 0:
                break
            rank += 1
            if rank > k:
                break
        if sum_gain == 0:
            return 0
        return sum_gain / sum_effort


#
# ERR.
#
# [reference]
# Olivier Chapelle, Donald Metlzer, Ya Zhang, and Pierre Grinspan.
# 2009. Expected reciprocal rank for graded relevance.
# In Proceedings of the 18th ACM conference on Information and knowledge management (CIKM '09).
# ACM, New York, NY, USA, 621-630. DOI=http://dx.doi.org/10.1145/1645953.1646033
class ERR:
    #
    # evac      the effort vector
    # rmax      the maximum possible relevance grade
    def __init__(self, evec, rmax):
        self.evec = evec
        self.rmax = rmax

    def evaluate(self, qrels, results, k):
        sum_utility, sum_effort, pexamine, rank = 0.0, 0.0, 1.0, 1
        for doc in results:
            rel = qrels.get(doc, 0)
            pstop = (2 ** rel - 1.0) / (2 ** self.rmax)
            effort = self.evec[rel]
            sum_effort += effort
            if pstop > 0:
                sum_utility += pexamine * pstop * 1.0 / sum_effort
            rank += 1
            pexamine *= 1 - pstop
            if rank > k:
                break
        return sum_utility


#
# A variant of time-biased gain using result relevance (instead of length) to estimate time.
#
# [reference]
# Mark D. Smucker and Charles L.A. Clarke. 2012. Time-based calibration of effectiveness measures.
# In Proceedings of the 35th international ACM SIGIR conference on Research and development in
# information retrieval (SIGIR '12). ACM, New York, NY, USA, 95-104. DOI=http://dx.doi.org/10.1145/2348283.2348300
class TBG:
    #
    # time      the expected time spent on results with each relevance grade
    # pclick    the probability to click on results with each relevance grade
    # psave     the probability to save results with each relevance grade after clicking
    # h         the parameter h
    def __init__(self, time, pclick, psave, h):
        self.time = time
        self.pclick = pclick
        self.psave = psave
        self.h = h

    def evaluate(self, qrels, results, k):
        tbg, arrive_time, rank = 0.0, 0.0, 1
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = self.pclick[rel] * self.psave[rel]
            discount = math.exp(-arrive_time * math.log(2, math.e) / self.h)
            tbg += gain * discount
            arrive_time += self.time[rel]
            rank += 1
            if rank > k:
                break
        return tbg


#
# A variant of U-measure based on time spent (instead of the number of examined characters).
#
# [reference]
# Tetsuya Sakai and Zhicheng Dou. 2013. Summaries, ranked retrieval and sessions:
# a unified framework for information access evaluation. In Proceedings of the 36th international ACM
# SIGIR conference on Research and development in information retrieval (SIGIR '13).
# ACM, New York, NY, USA, 473-482. DOI=http://dx.doi.org/10.1145/2484028.2484031
class UMeasure:
    #
    # time      the expected time spent on results with each relevance grade
    # gain      the gain value for results with each relevance grade
    # T
    def __init__(self, rmax, time, T):
        self.rmax = rmax
        self.time = time
        self.T = T

    def evaluate(self, qrels, results, k):
        sum_gain, arrive_time, rank = 0.0, 0.0, 1
        for doc in results:
            rel = qrels.get(doc, 0)
            gain = (2 ** rel - 1.0) / 2 ** self.rmax
            discount = 1 - arrive_time / self.T
            if discount < 0:
                discount = 0
            sum_gain += gain * discount
            arrive_time += self.time[rel]
            rank += 1
            if rank > k:
                break
        return sum_gain
