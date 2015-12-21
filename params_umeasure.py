#
# tune U-measure parameters by a brute force scan
#

import numpy as np
import scipy.stats as stats

from dataset import *
from query_metrics import *
from session_metrics import *

session_ratings = load_ratings('data/session')
session_results = load_results('data/results')
session_qrels = load_qrels('data/qrels')

examine_time = [9.8, 23.0, 37.6]

best_para = 0
best_r = 0.0
for T in xrange(40, 1000, 1):
    metric = SQMetric(UMeasure(2, examine_time, T), np.mean)
    ratings = []
    sevals = []
    for sessid in sorted(session_results.keys()):
        rating, qrels = session_ratings[sessid]['performance'], session_qrels[sessid]
        sevals.append(metric.evaluate(qrels, session_results[sessid], 9))
        ratings.append(rating)
    r = stats.pearsonr(ratings, sevals)[0]
    if r > best_r:
        best_r = r
        best_para = T
        print '%d  %.4f' % (T, stats.pearsonr(ratings, sevals)[0])
