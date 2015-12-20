#
# tune TBG parameters by a brute force scan
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
pclick = [0.26, 0.50, 0.55]

best_r = 0.0

for ps1 in xrange(0, 10):
    for ps2 in xrange(1, 10):
        psave = [0, ps1 / 10.0, ps2 / 10.0]
        for h in xrange(1, 500, 1):
            metric = SQMetric(TBG(examine_time, pclick, psave, h), np.mean)
            ratings = []
            sevals = []
            for sessid in sorted(session_results.keys()):
                rating, qrels = session_ratings[sessid]['performance'], session_qrels[sessid]
                sevals.append(metric.evaluate(qrels, session_results[sessid], 9))
                ratings.append(rating)
            r = stats.pearsonr(ratings, sevals)[0]
            if r >= best_r:
                best_r = r
                print '%.1f  %.1f  %d  %.4f' % (ps1 / 10.0, ps2 / 10.0, h, stats.pearsonr(ratings, sevals)[0])
