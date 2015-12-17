import math


# P@k.
class Prec:
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
            rank += 1
            if rank > k:
                break
        if sum_gain == 0:
            return 0
        return sum_gain / sum_effort


# GP@k.
class GradPrec:
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


# DCG@k
class DCG:
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


# nDCG@k
class NDCG:
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


# RBP
class RBP:
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


# GRBP
class GRBP:
    # evac      the effort vector
    # pdown     the probability to examine the next result
    # gs        relevance threshold probability
    #           for example, [0, 0.4, 0.6] means that users have:
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


# AP.
class AvgPrec:
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


# GAP.
class GradAvgPrec:
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


# RR.
class RR:
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


# ERR.
class ERR:
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


# TBG.
class TBG:
    # time      the expected time spent on results with each relevance grade
    # pclick    the probability to click on results with each relevance grade
    # psave     the probability to save results with each relevance grade after clicking
    # h         parameter h
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
