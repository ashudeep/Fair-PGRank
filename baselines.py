import torch
import numpy as np

from itertools import permutations
from sklearn import linear_model

from YahooDataReader import YahooDataReader
from models import NNModel, LinearModel
from evaluation import evaluate_model

import scipy

vvector = lambda N: 1. / np.log2(2 + np.arange(N))


def get_best_rankmatrix(true_rel_vector):
    N = len(true_rel_vector)
    bestranking = np.zeros((N, N))
    bestr = np.argsort(true_rel_vector)[::-1]
    for i in range(N):
        bestranking[bestr[i], i] = 1
    return bestranking


#returns DCG value
def get_DCG(ranking, relevances, vvector):
    N = len(relevances)
    return np.matmul(np.matmul(relevances, ranking), vvector.transpose())


def get_ndcg(ranking, relevances, vvector):
    if np.all(relevances == 0):
        return 1.0
    bestr = get_best_rankmatrix(relevances)
    return get_DCG(ranking, relevances, vvector) / get_DCG(
        bestr, relevances, vvector)


def get_fairness_loss(ranking, relevances, vvector, groups):
    #     print(ranking, relevances, vvector, groups)
    if np.all(groups == 0) or np.all(groups == 1):
        return 0.0
    avg_rels = [np.mean(relevances[groups == i]) for i in range(2)]
    if avg_rels[0] == 0 or avg_rels[1] == 0:
        return 0.0
    sign = +1 if avg_rels[0] >= avg_rels[1] else -1
    exposures = np.matmul(ranking, vvector)
    group_avg_exposures = [
        np.mean(exposures[groups == 0]),
        np.mean(exposures[groups == 1])
    ]
    #print(avg_rels, sign, exposures, group_avg_exposures)
    loss = max([
        0.0, sign * (group_avg_exposures[0] / avg_rels[0] -
                     group_avg_exposures[1] / avg_rels[1])
    ])
    return loss


def get_avg_fairness_loss(dr, predicted_rels, vvector, lmbda, args):
    feats, rel = dr.data
    test_losses = []
    for i in range(len(rel)):
        N = len(rel[i])
        pred_rels = predicted_rels[i]
        groups = np.array(feats[i][:, args.group_feat_id], dtype=np.int)
        P, _, _ = fair_rank(pred_rels, groups, lmbda)
        test_loss = get_fairness_loss(P, rel[i], vvector[:N], groups)
        test_losses.append(test_loss)
    return np.mean(test_losses)


def get_avg_ndcg_unfairness(dr, predicted_rels, vvector, lmbda,
                            group_feature_id):
    feats, rel = dr.data
    test_losses = []
    test_ndcgs = []
    for i in range(len(rel)):
        N = len(rel[i])
        pred_rels = predicted_rels[i]
        groups = np.array(feats[i][:, group_feature_id], dtype=np.int)
        P, _, _ = fair_rank(pred_rels, groups, lmbda)
        test_ndcg = get_ndcg(P, rel[i], vvector[:N])
        test_ndcgs.append(test_ndcg)
        test_loss = get_fairness_loss(P, rel[i], vvector[:N], groups)
        test_losses.append(test_loss)
    return np.mean(test_ndcgs), np.mean(test_losses)


def assign_groups(groups):
    G = [[], []]
    for i in range(len(groups)):
        G[groups[i]].append(i)
    return G


def fair_rank(relevances, groups, lmda=1):
    n = len(relevances)
    pos_bias = vvector(n)
    G = assign_groups(groups)
    n_g, n_i = 0, 0
    n_g += (len(G) - 1) * len(G)
    n_c = n**2 + n_g

    c = np.ones(n_c)
    c[:n**2] *= -1
    c[n**2:] *= lmda
    A_eq = []
    #For each Row
    for i in range(n):
        A_temp = np.zeros(n_c)
        A_temp[i * n:(i + 1) * n] = 1
        assert (sum(A_temp) == n)
        A_eq.append(A_temp)
        c[i * n:(i + 1) * n] *= relevances[i]

    #For each coloumn
    for i in range(n):
        A_temp = np.zeros(n_c)
        A_temp[i:n**2:n] = 1
        assert (sum(A_temp) == n)
        A_eq.append(A_temp)
        #Optimization
        c[i:n**2:n] *= pos_bias[i]
    b_eq = np.ones(n * 2)
    A_eq = np.asarray(A_eq)
    bounds = [(0, 1) for _ in range(n**2)] + [(0, None)
                                              for _ in range(n_g + n_i)]

    A_ub = []
    b_ub = np.zeros(n_g)
    sum_rels = []
    for group in G:
        #Avoid devision by zero
        sum_rel = np.max([np.sum(np.asarray(relevances)[group]), 0.01])
        sum_rels.append(sum_rel)
    comparisons = list(permutations(np.arange(len(G)), 2))
    j = 0
    for a, b in comparisons:
        f = np.zeros(n_c)
        if len(G[a]) > 0 and len(G[b]) > 0 and sum_rels[a] / len(
                G[a]) >= sum_rels[b] / len(G[b]):
            for i in range(n):
                tmp1 = len(G[a]) / sum_rels[a] if i in G[a] else 0
                tmp2 = len(G[b]) / sum_rels[b] if i in G[b] else 0
                #f[i*n:(i+1)*n] *= max(0, sign*(tmp1 - tmp2))
                f[i * n:(i + 1) * n] = (tmp1 - tmp2)
            for i in range(n):
                f[i:n**2:n] *= pos_bias[i]
            f[n**2 + j] = -1
        j += 1
        A_ub.append(f)

    res = scipy.optimize.linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="interior-point")  #, options=dict(tol=1e-12),)
    if res.success is False:
        print("Constraint not satisfied!!")
    probabilistic_ranking = np.reshape(res.x[:n**2], (n, n))
    return probabilistic_ranking, res, res.fun


def learn_and_predict(dr, vdr, intercept=True):
    # Linear regression
    print("Training linear regression on data with {} queries".format(
        len(dr.data[1])))
    model = linear_model.LinearRegression(
        fit_intercept=intercept, normalize=False)
    feats, rel = dr.data
    feats = np.array([item for sublist in feats for item in sublist])
    rel = np.array([item for sublist in rel for item in sublist])
    model.fit(feats, rel)
    # predictions on validation
    feats, rel = vdr.data
    se_sum = 0
    length = 0
    predicted_rels = []
    for i, query in enumerate(feats):
        rel_pred = model.predict(query[:, :])
        predicted_rels.append(rel_pred)
        se_sum += np.sum((rel_pred - rel[i])**2)
        length += len(rel[i])
    print("MSE : {}".format(se_sum / length))
    return predicted_rels, model


def eval_params(w, bias, dr, D, det=False, args=None, intercept=True):
    # Given the model weights, this function evaluates the model
    model = LinearModel(D=D)
    model.w.weight.data = torch.FloatTensor([w])
    if intercept:
        model.w.bias.data = torch.FloatTensor([bias])
    return evaluate_model(
        model,
        dr,
        deterministic=det,
        group_fairness_evaluation=True,
        args=args,
        fairness_evaluation=True)
