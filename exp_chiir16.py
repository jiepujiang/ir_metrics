#
# Experiment script for comparing different evaluation metrics for a search session.
#
# [Reference]
# Jiepu Jiang and James Allan. Correlation between system and user metrics in a session.
# In Proceedings of the first ACM SIGIR Conference on Human Information Interaction and Retrieval (CHIIR '16),
# Chapel Hill, North Carolina, USA, 2016.
#
# http://people.cs.umass.edu/~jpjiang/papers/chiir16_metrics.pdf
#

import numpy as np

from utils import *
from dataset import *
from query_metrics import *
from session_metrics import *

# turn this on to print the latex table
latex = False

# load dataset
session_ratings = load_ratings('data/session')
session_results = load_results('data/results')
session_qrels = load_qrels('data/qrels')

# evaluate the top 9 results for each query (because the dataset only provides 9 results per SERP)
k = 9

# metrics to be compared with
metrics = [
    ['sDCG', SDCG(2, 4, True)],
    ['nsDCG', NSDCG(2, 4, True)],
    ['sDCG/q', SDCGQ(2, 4, True)],
    ['esNDCG (0.9, 0.7)', ESNDCG(0.9, 0.7, True)],
    ['esNCG (0.8, 0.7)', ESNDCG(0.8, 0.7, False)],
    ['sDCG (no query discount)', SDCG(2, 4, False)],
    ['nsDCG (no query discount)', NSDCG(2, 4, False)],
    ['sDCG/q (no query discount)', SDCGQ(2, 4, False)],
    ['sum nDCG', SQMetric(NDCG([1.0, 1.0, 1.0]), np.sum)],
    ['mean nDCG', SQMetric(NDCG([1.0, 1.0, 1.0]), np.mean)],
    ['max nDCG', SQMetric(NDCG([1.0, 1.0, 1.0]), np.max)],
    ['min nDCG', SQMetric(NDCG([1.0, 1.0, 1.0]), np.min)],
    ['first nDCG', SQMetric(NDCG([1.0, 1.0, 1.0]), first)],
    ['last nDCG', SQMetric(NDCG([1.0, 1.0, 1.0]), last)],
]

# compute: performance, difficulty, #queries
if not latex:
    print(
        '%-30s  %20s  %20s  %20s  %20s'
        %
        (
            'Metric',
            'Pearson: performance',
            'Spearman: performance',
            'Pearson: difficulty',
            'Spearman: difficulty',
        )
    )

vals_performance, vals_difficulty, vals_numq = [], [], []
for sessid in session_results.keys():
    vals_performance.append(session_ratings[sessid]['performance'])
    vals_difficulty.append(session_ratings[sessid]['difficulty'])
    vals_numq.append(len(session_results[sessid]))

r_pd, pr_pd = stats.pearsonr(vals_performance, vals_difficulty)
r_pq, pr_pq = stats.pearsonr(vals_performance, vals_numq)
r_dq, pr_dq = stats.pearsonr(vals_numq, vals_difficulty)

rho_pd, prho_pd = stats.spearmanr(vals_performance, vals_difficulty)
rho_pq, prho_pq = stats.spearmanr(vals_performance, vals_numq)
rho_dq, prho_dq = stats.spearmanr(vals_numq, vals_difficulty)

if latex:
    print(
        ' & %-30s & %16s & %-3s & %16s & %-3s & $%.3f$ & %-3s & $%.3f$ & %-3s \\\\'
        %
        (
            'Performance',
            '-', '',
            '-', '',
            r_pd, star(pr_pd),
            rho_pd, star(prho_pd),
        )
    )
else:
    print(
        '%-30s  %16s %-3s  %16s %-3s  %16.3f %-3s  %16.3f %-3s'
        %
        (
            'Performance',
            '-', '',
            '-', '',
            r_pd, star(pr_pd),
            rho_pd, star(prho_pd),
        )
    )

if latex:
    print(
        ' & %-30s & $%.3f$ & %-3s & $%.3f$ & %-3s & %16s & %-3s & %16s & %-3s \\\\'
        %
        (
            'Difficulty',
            r_pd, star(pr_pd),
            rho_pd, star(prho_pd),
            '-', '',
            '-', '',
        )
    )
else:
    print(
        '%-30s  %16.3f %-3s  %16.3f %-3s  %16s %-3s  %16s %-3s'
        %
        (
            'Difficulty',
            r_pd, star(pr_pd),
            rho_pd, star(prho_pd),
            '-', '',
            '-', '',
        )
    )

if latex:
    print(
        ' & %-30s & $%.3f$ & %-3s & $%.3f$ & %-3s & $%.3f$ & %-3s & $%.3f$ & %-3s \\\\'
        %
        (
            '#queries',
            r_pq, star(pr_pq),
            rho_pq, star(prho_pq),
            r_dq, star(pr_dq),
            rho_dq, star(prho_dq),
        )
    )
else:
    print(
        '%-30s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s'
        %
        (
            '#queries',
            r_pq, star(pr_pq),
            rho_pq, star(prho_pq),
            r_dq, star(pr_dq),
            rho_dq, star(prho_dq),
        )
    )

# compute correlation for other metrics
for [name, metric] in metrics:
    r1, pr1, rho1, prho1 = correlation(session_ratings, session_results, session_qrels, 'performance', metric, k)
    r2, pr2, rho2, prho2 = correlation(session_ratings, session_results, session_qrels, 'difficulty', metric, k)
    if latex:
        print(
            ' & %-30s & $%.3f$ & %-3s & $%.3f$ & %-3s & $%.3f$ & %-3s & $%.3f$ & %-3s \\\\'
            %
            (
                name,
                r1, star(pr1),
                rho1, star(prho1),
                r2, star(pr2),
                rho2, star(prho2),
            )
        )
    else:
        print(
            '%-30s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s'
            %
            (
                name,
                r1, star(pr1),
                rho1, star(prho1),
                r2, star(pr2),
                rho2, star(prho2),
            )
        )
