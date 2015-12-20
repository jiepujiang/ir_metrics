import numpy as np
import scipy.stats as stats

from utils import *
from query_metrics import *
from session_metrics import *

session_ratings = load_ratings('data/session')
session_results = load_results('data/results')
session_qrels = load_qrels('data/qrels')

examine_time = [9.8, 23.0, 37.6]

# best_para = [0, 0, 0]
# best_r = 0.0
# for r1 in xrange(0, 10):
#     for r2 in xrange(1, 10):
#         gain = [0.0, r1, r2]
#         for T in xrange(50, 500, 10):
#             metric = SQMetric(UMeasure(examine_time, gain, T), np.mean)
#             ratings = []
#             sevals = []
#             for sessid in sorted(session_results.keys()):
#                 rating, qrels = session_ratings[sessid]['performance'], session_qrels[sessid]
#                 sevals.append(metric.evaluate(qrels, session_results[sessid], 9))
#                 ratings.append(rating)
#             r = stats.pearsonr(ratings, sevals)[0]
#             if r > best_r:
#                 best_r = r
#                 best_para = [r1, r2, T]
#                 print '%d  %d  %d  %.4f' % (r1, r2, T, stats.pearsonr(ratings, sevals)[0])

examine_time = [9.8, 23.0, 37.6]

best_para = [0, 0, 0]
best_r = 0.0
for p1 in xrange(0, 20, 1):
    for p2 in xrange(1, 20, 1):
        for T in xrange(50, 500, 10):
            p1, p2 = p1 / 20.0, p2 / 20.0
            metric = SQMetric(UMeasure2(examine_time, [0.26, 0.50, 0.55], [0, p1, p2], T), np.mean)
            ratings = []
            sevals = []
            for sessid in sorted(session_results.keys()):
                rating, qrels = session_ratings[sessid]['performance'], session_qrels[sessid]
                sevals.append(metric.evaluate(qrels, session_results[sessid], 9))
                ratings.append(rating)
            r = stats.pearsonr(ratings, sevals)[0]
            if r > best_r:
                best_r = r
                best_para = [p1, p2, T]
                print '%.1f  %.1f  %d  %.4f' % (p1, p2, T, stats.pearsonr(ratings, sevals)[0])
