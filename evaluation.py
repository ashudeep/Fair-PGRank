import numpy as np
import math, sys

from progressbar import progressbar
from utils import torchify
import torch
from models import convert_vars_to_gpu

from fairness_loss import get_exposures
from scipy.optimize import minimize
from scipy.stats import linregress


def sample_ranking(probs, output_propensities=True):
    probs_ = np.array(probs, copy=True)
    ranking = []
    propensity = 1.0
    candidate_set_size = probs_.shape[0]
    #probs_ = np.clip(probs_, 1e-5, 1.0)
    probs_ = probs_ / probs_.sum()
    try:
        ranking = np.random.choice(
            candidate_set_size,
            size=candidate_set_size,
            p=probs_,
            replace=False)
    except ValueError:
        print(probs_)
        sys.exit(1)
    if output_propensities:
        for i in range(candidate_set_size):
            propensity *= probs_[ranking[i]]
            probs_[ranking[i]] = 0.0
            probs_ = probs_ / probs_.sum()
        return ranking, propensity
    else:
        return ranking


# def sample_ranking(scores):
#     ranking = []
#     candidate_set_size = scores.shape[0]


def compute_average_rank(ranking,
                         relevance_vector,
                         relevance_threshold=0,
                         get_dcg=False):
    """
    get_dcg: True computes dcg the way Agrawal et al. calculates it
    Works well only when we are dealing with binary relevance case
    """
    rank_rels = []
    for i, doc in enumerate(ranking):
        if relevance_vector[doc] > relevance_threshold:
            if get_dcg:
                rank_rels.append(1.0 / math.log(2 + i))
            else:
                rank_rels.append(i)
    return rank_rels


def compute_dcg(ranking, relevance_vector, k=0):
    """
    returns the array. actual dcg is the sum or average of this array
    """
    dcgmax = 0.0
    sorted_relevances = -np.sort(-relevance_vector)
    N = len(relevance_vector)
    if k == 0:
        k = N
    for i, relevance in enumerate(sorted_relevances[:min((k, N))]):
        dcgmax += float(2.0**relevance - 1) / math.log2(2 + i)
    dcg = 0.0
    for i, doc in enumerate(ranking[:min((k, N))]):
        dcg += float(2**relevance_vector[doc] - 1) / math.log2(2 + i)
    if dcgmax == 0:
        # print(relevance_vector, ranking)
        return 1.0, 1.0
    else:
        return dcg / dcgmax, dcg


def get_relative_gain(relevance):
    return (2.0**relevance - 1) / 16


def compute_err(ranking, relevance_vector):
    """
    Defined in Chapelle 11a (Section 5.1.1)
    """
    err = 0.0
    for i, doc in enumerate(ranking):
        not_found_probability = 1.0
        for j in range(i):
            not_found_probability *= 1 - get_relative_gain(
                relevance_vector[ranking[j]])
        err += get_relative_gain(
            relevance_vector[doc]) * not_found_probability / (1 + i)
    return err


def pairwise_mse(exposures, relevances):
    mse = 0.0
    e_by_r = exposures / relevances
    N = len(relevances)
    for i in range(N):
        for j in range(i, N):
            mse += (e_by_r[i] - e_by_r[j])**2
    return mse / (N * N)


def scale_invariant_mse(exposures, relevances):
    """
    https://arxiv.org/pdf/1406.2283v1.pdf Equation 1, 2, 3
    """
    # sqrt(Eq. 3)
    assert (np.all(
        np.isfinite(exposures) & np.isfinite(relevances) & (exposures > 0) &
        (relevances > 0)))
    log_diff = np.log(exposures) - np.log(relevances)
    num_pixels = float(log_diff.size)

    if num_pixels == 0:
        return np.nan
    else:
        return np.sqrt(
            np.sum(np.square(log_diff)) / num_pixels -
            np.square(np.sum(log_diff)) / np.square(num_pixels))


def asymmetric_disparity(exposures, relevances):
    disparities = []
    for i in range(len(exposures)):
        for j in range(len(exposures)):
            if relevances[i] >= relevances[j]:
                if relevances[j] > 0.0:
                    disparities.append(
                        max([
                            0, exposures[i] / relevances[i] -
                            exposures[j] / relevances[j]
                        ]))
                else:
                    disparities.append(0)
    if np.isnan(np.mean(disparities)):
        print("NAN occured at", exposures, relevances, disparities)
    return np.mean(disparities)


def evaluate_model(model,
                   validation_data_reader,
                   num_sample_per_query=10,
                   deterministic=False,
                   gpu_id=None,
                   fairness_evaluation=False,
                   position_bias_vector=None,
                   group_fairness_evaluation=False,
                   writer=None,
                   epoch_num=None,
                   args=None):
    # validation
    # val_files = os.listdir(validation_data_reader.directory)
    ndcg_list = []
    dcg_list = []
    err_list = []
    relevant_rank_list = []
    if (fairness_evaluation
            or group_fairness_evaluation) and position_bias_vector is None:
        position_bias_vector = 1. / np.log2(2 + np.arange(200))
    if fairness_evaluation:
        l1_dists = []
        rsq_dists = []
        residuals = []
        scale_invariant_mses = []
        asymmetric_disparities = []
    if group_fairness_evaluation:
        group_exposure_disparities = []
        group_asym_disparities = []
    val_feats, val_rel = validation_data_reader.data
    len_val_set = len(val_feats)

    all_exposures = []
    all_rels = []
    iterator = progressbar(range(
        len_val_set)) if args is not None and args.progressbar else range(
            len_val_set)
    for i in iterator:  # for each query
        feats, rel = val_feats[i], val_rel[i]
        if gpu_id is not None:
            feats, rel = convert_vars_to_gpu([feats, rel], gpu_id)
        scores = model(torchify(feats))
        scores = scores * args.eval_temperature
        probs = torch.nn.Softmax(dim=0)(scores).data.numpy().flatten()
        #print(feats, scores, probs, rel)
        if deterministic:
            num_sample_per_query = 1

        if fairness_evaluation or group_fairness_evaluation:
            exposures = np.zeros(len(feats))
            one_hot_rel = np.array(rel, dtype=float)
            if group_fairness_evaluation:
                group_identities = feats[:, args.group_feat_id]
                # print(group_identities, args.group_feat_id)
                # print(feats)

        curr_dcg_list = []
        curr_ndcg_list = []
        curr_err_list = []

        for j in range(num_sample_per_query):
            if deterministic:
                ranking = np.argsort(probs)[::-1]
            else:
                ranking = sample_ranking(probs, False)
            ndcg, dcg = compute_dcg(ranking, rel, args.evalk)
            av_ranks = compute_average_rank(ranking, rel)
            err = compute_err(ranking, rel)
            curr_ndcg_list.append(ndcg)
            curr_dcg_list.append(dcg)
            relevant_rank_list.extend(av_ranks)
            curr_err_list.append(err)
            if fairness_evaluation or group_fairness_evaluation:
                curr_exposure = get_exposures(ranking, position_bias_vector)
                exposures += curr_exposure
        dcg_list.append(np.mean(curr_dcg_list))
        ndcg_list.append(np.mean(curr_ndcg_list))
        err_list.append(np.mean(curr_err_list))
        if group_fairness_evaluation or fairness_evaluation:
            exposures = exposures / num_sample_per_query

        if group_fairness_evaluation:
            rel_mean_g0 = np.mean(rel[group_identities == 0])
            rel_mean_g1 = np.mean(rel[group_identities == 1])
            # skip for candidate sets when there is no diversity
            if (np.sum(group_identities == 0) == 0
                    or np.sum(group_identities == 1) == 0
                ) or rel_mean_g0 == 0 or rel_mean_g1 == 0:
                # print(group_identities, rel)
                group_exposure_disparities.append(0.0)
                group_asym_disparities.append(0.0)
                # if there is only one group
            else:
                exposure_mean_g0 = np.mean(exposures[group_identities == 0])
                exposure_mean_g1 = np.mean(exposures[group_identities == 1])
                # print(exposure_mean_g0, exposure_mean_g1)
                disparity = exposure_mean_g0 / rel_mean_g0 - exposure_mean_g1 / rel_mean_g1
                group_exposure_disparity = disparity**2
                sign = +1 if rel_mean_g0 >= rel_mean_g1 else -1
                one_sided_group_disparity = max([0, sign * disparity])

                # print(group_exposure_disparity, exposure_mean_g0,
                # exposure_mean_g1, rel, group_identities)
                group_exposure_disparities.append(group_exposure_disparity)
                group_asym_disparities.append(one_sided_group_disparity)

        if fairness_evaluation:
            all_exposures.extend(exposures)
            all_rels.extend(one_hot_rel)
            # print(ratios, one_hot_rel, exposures)

            non_zero_indices = one_hot_rel != 0
            if sum(non_zero_indices) == 0:
                continue
            scale_invariant_mses.append(
                scale_invariant_mse(exposures[non_zero_indices], one_hot_rel[
                    non_zero_indices]))
            asymmetric_disparities.append(
                asymmetric_disparity(exposures, one_hot_rel))
            # MSE is always calculated for non_zero_indices
            if args.skip_zero_relevance:
                exposures, one_hot_rel = exposures[
                    non_zero_indices], one_hot_rel[non_zero_indices]
            try:
                res = minimize(
                    lambda k: np.sum(np.abs(k * one_hot_rel - exposures)),
                    1.0,
                    method='Nelder-Mead')
            except:
                print("l1 distance error", exposures, one_hot_rel)
            l1_dist = res.fun
            l1_dists.append(l1_dist)

            if len(one_hot_rel) == 1:
                rsq_dists.append(1.0)
            else:
                # one_hot_rel = add_tiny_noise(one_hot_rel)
                _, _, rval, _, _ = linregress(exposures, one_hot_rel)
                rsq_dists.append(rval**2)
            try:
                residual = minimize(
                    lambda k: np.sum(np.square(exposures - k * one_hot_rel)),
                    1.0,
                    method='Nelder-Mead')
            except:
                print("residual error", exposures, one_hot_rel)
            residuals.append(residual.fun)

            # ratios = one_hot_rel / exposures
            # ratios /= np.sum(ratios)
            # hentropy = entropy(ratios)
            # exposures /= np.sum(exposures)
            # one_hot_rel /= np.sum(one_hot_rel)
            # # kl_div = entropy(one_hot_rel, exposures)
            # entropies.append(hentropy)
            # kl_divs.append(kl_div)

            # assuming group identities are only 0 or 1

        # if args.macro_avg:
        #     ndcg_list.extend(curr_ndcg_list)
        #     dcg_list.extend(curr_dcg_list)
        # else:
        #     ndcg_list.append(np.mean(curr_ndcg_list))
        #     dcg_list.append(np.mean(curr_dcg_list))

    avg_ndcg = np.mean(ndcg_list)
    avg_dcg = np.mean(dcg_list)
    average_rank = np.mean(relevant_rank_list)
    avg_err = np.mean(err_list)

    if writer is not None:
        writer.add_embedding(
            np.vstack((all_exposures, all_rels)).transpose(),
            global_step=epoch_num)
    # if plot_exposure_vs_rel:

    results = {
        "ndcg": avg_ndcg,
        "dcg": avg_dcg,
        "avg_rank": average_rank,
        "err": avg_err
    }
    if fairness_evaluation:
        # avg_kl_div = np.mean(kl_div)
        # avg_entropies = np.mean(entropies)
        avg_l1_dists = np.mean(l1_dists)
        avg_rsq = np.mean(rsq_dists)
        avg_residuals = np.mean(residuals)
        avg_sc_inv_mse = np.mean(scale_invariant_mses)
        avg_asym_disparity = np.mean(asymmetric_disparities)
        results.update({
            # "avg_kl_div": avg_kl_div,
            # "avg_entropies": avg_entropies,
            "avg_residuals": avg_residuals,
            "avg_rsq": avg_rsq,
            "avg_l1_dists": avg_l1_dists,
            # "exposures": all_exposures,
            # "rels": all_rels,
            "scale_inv_mse": avg_sc_inv_mse,
            "asymmetric_disparity": avg_asym_disparity
        })
    if group_fairness_evaluation:
        avg_group_exposure_disparity = np.mean(group_exposure_disparities)
        avg_group_asym_disparity = np.mean(group_asym_disparities)
        results.update({
            "avg_group_disparity": avg_group_exposure_disparity,
            "avg_group_asym_disparity": avg_group_asym_disparity
        })
    return results


def add_tiny_noise(one_hot_rel):
    """
    used to add tiny noise to avoid warnings in linregress
    """
    if one_hot_rel.min() == one_hot_rel.max():
        one_hot_rel = one_hot_rel + np.random.random(len(one_hot_rel)) * 1e-20
    return one_hot_rel


def optimal_exposure(num_relevant, num_docs, position_bias_function):
    """
    returns the optimal exposure that a randomized policy can give for
    the given number of relevant documents
    """
    top_k_exposure = np.mean(
        [position_bias_function(i) for i in range(num_relevant)])
    remaining_exposure = np.mean(
        [position_bias_function(i) for i in range(num_relevant, num_docs)])
    optimal_exposure = [top_k_exposure
                        ] * num_relevant + [remaining_exposure] * (
                            num_docs - num_relevant)
    return optimal_exposure


def get_random_ranking_baseline(validation_data_reader,
                                num_sample_per_query=100,
                                fairness_evaluation=True,
                                group_fairness_evaluation=False,
                                position_bias_vector=None,
                                args=None):
    # validation
    # val_files = os.listdir(validation_data_reader.directory)
    ndcg_list = []
    dcg_list = []
    err_list = []
    relevant_rank_list = []
    if (fairness_evaluation
            or group_fairness_evaluation) and position_bias_vector is None:
        position_bias_vector = 1. / np.log2(2 + np.arange(200))
    if fairness_evaluation:
        l1_dists = []
        rsq_dists = []
        residuals = []
        scale_invariant_mses = []
        asymmetric_disparities = []
    if group_fairness_evaluation:
        group_exposure_disparities = []
        group_asym_disparities = []
    val_feats, val_rel = validation_data_reader.data
    len_val_set = len(val_feats)

    all_exposures = []
    all_rels = []
    iterator = progressbar(range(
        len_val_set)) if args is not None and args.progressbar else range(
            len_val_set)
    for i in iterator:  # for each query
        feats, rel = val_feats[i], val_rel[i]

        if fairness_evaluation or group_fairness_evaluation:
            exposures = np.zeros(len(feats))
            one_hot_rel = np.array(rel, dtype=float)
            if group_fairness_evaluation:
                group_identities = feats[:, args.group_feat_id]
                # print(group_identities, args.group_feat_id)
                # print(feats)

        curr_dcg_list = []
        curr_ndcg_list = []
        curr_err_list = []

        for j in range(num_sample_per_query):
            ranking = np.random.permutation(len(rel))
            ndcg, dcg = compute_dcg(ranking, rel, args.evalk)
            av_ranks = compute_average_rank(ranking, rel)
            err = compute_err(ranking, rel)
            curr_ndcg_list.append(ndcg)
            curr_dcg_list.append(dcg)
            relevant_rank_list.extend(av_ranks)
            curr_err_list.append(err)
            if fairness_evaluation or group_fairness_evaluation:
                curr_exposure = get_exposures(ranking, position_bias_vector)
                exposures += curr_exposure
        dcg_list.append(np.mean(curr_dcg_list))
        ndcg_list.append(np.mean(curr_ndcg_list))
        err_list.append(np.mean(curr_err_list))
        if group_fairness_evaluation or fairness_evaluation:
            exposures = exposures / num_sample_per_query

        if group_fairness_evaluation:
            rel_mean_g0 = np.mean(rel[group_identities == 0])
            rel_mean_g1 = np.mean(rel[group_identities == 1])
            # skip for candidate sets when there is no diversity
            if (np.sum(group_identities == 0) == 0
                    or np.sum(group_identities == 1) == 0
                ) or rel_mean_g0 == 0 or rel_mean_g1 == 0:
                # print(group_identities, rel)
                group_exposure_disparities.append(0.0)
                group_asym_disparities.append(0.0)
                # if there is only one group
            else:
                exposure_mean_g0 = np.mean(exposures[group_identities == 0])
                exposure_mean_g1 = np.mean(exposures[group_identities == 1])
                # print(exposure_mean_g0, exposure_mean_g1)
                disparity = exposure_mean_g0 / rel_mean_g0 - exposure_mean_g1 / rel_mean_g1
                group_exposure_disparity = disparity**2
                sign = +1 if rel_mean_g0 > rel_mean_g1 else -1
                one_sided_group_disparity = max([0, sign * disparity])

                # print(group_exposure_disparity, exposure_mean_g0,
                # exposure_mean_g1, rel, group_identities)
                group_exposure_disparities.append(group_exposure_disparity)
                group_asym_disparities.append(one_sided_group_disparity)

        if fairness_evaluation:
            all_exposures.extend(exposures)
            all_rels.extend(one_hot_rel)
            # print(ratios, one_hot_rel, exposures)

            non_zero_indices = one_hot_rel != 0
            if sum(non_zero_indices) == 0:
                continue
            scale_invariant_mses.append(
                scale_invariant_mse(exposures[non_zero_indices], one_hot_rel[
                    non_zero_indices]))
            asymmetric_disparities.append(
                asymmetric_disparity(exposures, one_hot_rel))
            # MSE is always calculated for non_zero_indices
            if args.skip_zero_relevance:
                exposures, one_hot_rel = exposures[
                    non_zero_indices], one_hot_rel[non_zero_indices]
            try:
                res = minimize(
                    lambda k: np.sum(np.abs(k * one_hot_rel - exposures)),
                    1.0,
                    method='Nelder-Mead')
            except:
                print("l1 distance error", exposures, one_hot_rel)
            l1_dist = res.fun
            l1_dists.append(l1_dist)

            if len(one_hot_rel) == 1:
                rsq_dists.append(1.0)
            else:
                # one_hot_rel = add_tiny_noise(one_hot_rel)
                _, _, rval, _, _ = linregress(exposures, one_hot_rel)
                rsq_dists.append(rval**2)
            try:
                residual = minimize(
                    lambda k: np.sum(np.square(exposures - k * one_hot_rel)),
                    1.0,
                    method='Nelder-Mead')
            except:
                print("residual error", exposures, one_hot_rel)
            residuals.append(residual.fun)

            # ratios = one_hot_rel / exposures
            # ratios /= np.sum(ratios)
            # hentropy = entropy(ratios)
            # exposures /= np.sum(exposures)
            # one_hot_rel /= np.sum(one_hot_rel)
            # # kl_div = entropy(one_hot_rel, exposures)
            # entropies.append(hentropy)
            # kl_divs.append(kl_div)

            # assuming group identities are only 0 or 1

        # if args.macro_avg:
        #     ndcg_list.extend(curr_ndcg_list)
        #     dcg_list.extend(curr_dcg_list)
        # else:
        #     ndcg_list.append(np.mean(curr_ndcg_list))
        #     dcg_list.append(np.mean(curr_dcg_list))

    avg_ndcg = np.mean(ndcg_list)
    avg_dcg = np.mean(dcg_list)
    average_rank = np.mean(relevant_rank_list)
    avg_err = np.mean(err_list)

    results = {
        "ndcg": avg_ndcg,
        "dcg": avg_dcg,
        "avg_rank": average_rank,
        "err": avg_err
    }
    if fairness_evaluation:
        # avg_kl_div = np.mean(kl_div)
        # avg_entropies = np.mean(entropies)
        avg_l1_dists = np.mean(l1_dists)
        avg_rsq = np.mean(rsq_dists)
        avg_residuals = np.mean(residuals)
        avg_sc_inv_mse = np.mean(scale_invariant_mses)
        avg_asym_disparity = np.mean(asymmetric_disparities)
        results.update({
            # "avg_kl_div": avg_kl_div,
            # "avg_entropies": avg_entropies,
            "avg_residuals": avg_residuals,
            "avg_rsq": avg_rsq,
            "avg_l1_dists": avg_l1_dists,
            # "exposures": all_exposures,
            # "rels": all_rels,
            "scale_inv_mse": avg_sc_inv_mse,
            "asymmetric_disparity": avg_asym_disparity
        })
    if group_fairness_evaluation:
        avg_group_exposure_disparity = np.mean(group_exposure_disparities)
        avg_group_asym_disparity = np.mean(group_asym_disparities)
        results.update({
            "avg_group_disparity": avg_group_exposure_disparity,
            "avg_group_asym_disparity": avg_group_asym_disparity
        })
    return results
