#
# tune esnDCG parameters by a brute force scan
#

import scipy.stats as stats

from dataset import *
from session_metrics import *

session_ratings = load_ratings('data/session')
session_results = load_results('data/results')
session_qrels = load_qrels('data/qrels')

evec_static = [1.0, 1.0, 1.0]
umetric = 'performance'

for pref in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for pdown in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        smetric = ESNDCG(pref, pdown, True)
        ratings = []
        sevals = []
        for sessid in sorted(session_results.keys()):
            rating, qrels = session_ratings[sessid][umetric], session_qrels[sessid]
            sevals.append(smetric.evaluate(qrels, session_results[sessid], 9))
            ratings.append(rating)
        print 'esNDCG %.1f %.1f        %.3f' % (pref, pdown, stats.pearsonr(ratings, sevals)[0])

for pref in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    for pdown in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        smetric = ESNDCG(pref, pdown, False)
        ratings = []
        sevals = []
        for sessid in sorted(session_results.keys()):
            rating, qrels = session_ratings[sessid][umetric], session_qrels[sessid]
            sevals.append(smetric.evaluate(qrels, session_results[sessid], 9))
            ratings.append(rating)
        print 'esNCG %.1f %.1f        %.3f' % (pref, pdown, stats.pearsonr(ratings, sevals)[0])
