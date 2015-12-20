import numpy as np
import scipy.stats as stats

from utils import *
from query_metrics import *
from session_metrics import *


def correlation(sratings, sresults, sqrels, umetric, smetric, k):
    ratings = []
    sevals = []
    for sessid in sresults.keys():
        ratings.append(sratings[sessid][umetric])
        sevals.append(smetric.evaluate(sqrels[sessid], sresults[sessid], k))
    pearson, p_pearson = stats.pearsonr(ratings, sevals)
    spearman, p_spearman = stats.spearmanr(ratings, sevals)
    return pearson, p_pearson, spearman, p_spearman


def star(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ''


def first(array):
    return array[0]


def last(array):
    return array[len(array) - 1]


latex = True

session_ratings = load_ratings('data/session')
session_results = load_results('data/results')
session_qrels = load_qrels('data/qrels')

k = 9

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

if not latex:
    print(
        '%-20s  %20s  %20s  %20s  %20s'
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
        ' & %-20s & %16s & %-3s & %16s & %-3s & $%.3f$ & %-3s & $%.3f$ & %-3s \\\\'
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
        '%-20s  %16s %-3s  %16s %-3s  %16.3f %-3s  %16.3f %-3s'
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
        ' & %-20s & $%.3f$ & %-3s & $%.3f$ & %-3s & %16s & %-3s & %16s & %-3s \\\\'
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
        '%-20s  %16.3f %-3s  %16.3f %-3s  %16s %-3s  %16s %-3s'
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
        ' & %-20s & $%.3f$ & %-3s & $%.3f$ & %-3s & $%.3f$ & %-3s & $%.3f$ & %-3s \\\\'
        %
        (
            'Performance',
            r_pq, star(pr_pq),
            rho_pq, star(prho_pq),
            r_dq, star(pr_dq),
            rho_dq, star(prho_dq),
        )
    )
else:
    print(
        '%-20s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s'
        %
        (
            'Performance',
            r_pq, star(pr_pq),
            rho_pq, star(prho_pq),
            r_dq, star(pr_dq),
            rho_dq, star(prho_dq),
        )
    )

for [name, metric] in metrics:
    r1, pr1, rho1, prho1 = correlation(session_ratings, session_results, session_qrels, 'performance', metric, k)
    r2, pr2, rho2, prho2 = correlation(session_ratings, session_results, session_qrels, 'difficulty', metric, k)
    if latex:
        print(
            ' & %-20s & $%.3f$ & %-3s & $%.3f$ & %-3s & $%.3f$ & %-3s & $%.3f$ & %-3s \\\\'
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
            '%-20s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s  %16.3f %-3s'
            %
            (
                name,
                r1, star(pr1),
                rho1, star(prho1),
                r2, star(pr2),
                rho2, star(prho2),
            )
        )
