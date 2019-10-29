import numpy as np
from scipy.optimize import minimize


def relevant_indices_to_onehot(rel, num_docs):
    onehot = np.zeros(num_docs)
    for relevant_doc in rel:
        onehot[relevant_doc] = 1
    return onehot


def get_exposures(ranking, position_bias_vector):
    num_docs = len(ranking)
    exposure = np.zeros(num_docs)
    exposure[ranking] = position_bias_vector[:num_docs]
    return exposure


def get_expected_exposure(rankings, position_bias_vector):
    exp_exposure = np.zeros(len(rankings[0]))
    for ranking in rankings:
        exp_exposure += get_exposures(ranking, position_bias_vector)
    exp_exposure = exp_exposure / len(rankings)
    return exp_exposure


def minimize_for_k(rel, exposure, skip_zero=False):
    if skip_zero:
        inds = rel != 0
        rel, exposure = rel[inds], exposure[inds]
    res = minimize(
        lambda k: np.sum(np.square((exposure - k * rel))),
        1.0,
        method='Nelder-Mead')
    return res.x  # res.x is the value of k for which the minimum occurs


class IndividualFairnessLoss(object):
    @staticmethod
    def compute_disparities(rankings,
                            one_hot_rel,
                            position_biases,
                            k,
                            skip_zero=False):
        """
        returns a (num_rankings, num_docs) matrix of
        disparities, disparity = relevance - z * positionbias
        """
        disparitiy_matrix = np.zeros((len(rankings), len(rankings[0])))
        for i, ranking in enumerate(rankings):
            disparitiy_matrix[i, ranking] = (
                position_biases[:len(ranking)] * k - one_hot_rel[ranking])**2
            if skip_zero:
                disparitiy_matrix[i, one_hot_rel == 0] = 0.0
        return disparitiy_matrix

    @staticmethod
    def get_scale_invariant_mse_coeffs(rankings,
                                       rels,
                                       position_biases,
                                       skip_zero=True):
        """
        given the rankings, gives a vector of coeffients that is then multiplied with
        the log \pi(r) to compute gradient over. See derivation in paper/appendix

        skip_zero always has to be True
        """
        n = len(rankings)
        coeffs = np.zeros(n)
        num_docs = len(rankings[0])
        exposures = np.zeros((n, num_docs))
        for i in range(n):
            ranking = rankings[i]
            exposures[i, ranking] = position_biases[:num_docs]
        mean_exposures = np.mean(exposures, axis=0)
        diffs = np.log(mean_exposures) - np.log(rels)
        if skip_zero:
            zero_inds = rels == 0
            diffs[zero_inds] = 0
        mean_diff = np.mean(diffs)
        for i in range(n):
            weighted_diffs = (diffs - mean_diff) * (
                exposures[i, :] / mean_exposures)
            if skip_zero:
                weighted_diffs[zero_inds] = 0.0
            coeffs[i] = 2 * np.mean(weighted_diffs)
        return coeffs

    @staticmethod
    def compute_marginal_disparity(disparitiy_matrix):
        """
        disparity matrix is of size (num_rankings, num_docs)
        rankings (num_rankings, num_docs)
        returns the marginal_disparity i.e averaged over the columns
        """
        return np.mean(disparitiy_matrix, axis=0)

    @staticmethod
    def compute_sq_individual_fairness_loss_coeff(ranking, disparity_vector,
                                                  marginal_disparity, k):
        inner_sum = np.sum(
            2 * k * marginal_disparity[ranking] * disparity_vector[ranking])
        return float(inner_sum)

    @staticmethod
    def compute_cross_entropy_fairness_loss(
            ranking, one_hot_rel, expected_exposures, position_biases):
        # print(ranking, position_biases[:len(ranking)])
        exposures = np.zeros(len(ranking))
        exposures[ranking] = position_biases[:len(ranking)]
        numerators = expected_exposures - one_hot_rel
        denominators = expected_exposures * (1 - expected_exposures)
        inner_sum = np.sum(exposures * (numerators / denominators))
        return float(inner_sum)

    @staticmethod
    def compute_pairwise_disparity_matrix(rankings, relevance_vector,
                                          position_bias_vector):

        num_rankings = len(rankings)
        N = len(rankings[0])
        matrix = np.zeros((num_rankings, N, N))
        for k, ranking in enumerate(rankings):
            for i in range(N):  # index in ranking 1
                for j in range(N):  # index in ranking 2
                    if relevance_vector[ranking[i]] == 0 or relevance_vector[ranking[j]] == 0:
                        matrix[k, ranking[i], ranking[j]] = 0.0
                    else:
                        matrix[k, ranking[i], ranking[
                            j]] = (position_bias_vector[i] /
                                   relevance_vector[ranking[i]]) - (
                                       position_bias_vector[j] /
                                       relevance_vector[ranking[j]])
        return matrix

    def get_H_matrix(relevance_vector):
        N = len(relevance_vector)
        H_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if (relevance_vector[i] >= relevance_vector[j]):
                    H_mat[i, j] = 1
        return H_mat


class GroupFairnessLoss:
    @staticmethod
    def compute_group_fairness_coeffs_generic(
            rankings, rel_labels, group_identities, position_bias_vector,
            group_fairness_version, skip_zero_relevance):
        if group_fairness_version == "sq_disparity":
            group_fairness_coeffs = GroupFairnessLoss.compute_group_disparity_coeffs(
                rankings, rel_labels, group_identities, position_bias_vector,
                skip_zero_relevance)
        elif group_fairness_version == "asym_disparity":
            group_fairness_coeffs = GroupFairnessLoss.compute_asym_group_disparity_coeffs(
                rankings, rel_labels, group_identities, position_bias_vector,
                skip_zero_relevance)
        return group_fairness_coeffs

    @staticmethod
    def compute_group_disparity(ranking,
                                rel,
                                group_identities,
                                position_biases,
                                skip_zero=False):
        exposures = get_exposures(ranking, position_biases)
        inds_g0 = group_identities == 0
        inds_g1 = group_identities == 1
        if skip_zero:
            inds_g0 = np.logical_and(inds_g0, rel != 0)
            inds_g1 = np.logical_and(inds_g1, rel != 0)
        return np.sum(exposures[inds_g0]) / np.sum(rel[inds_g0]) - np.sum(
            exposures[inds_g1]) / np.sum(rel[inds_g1])

    @staticmethod
    def compute_group_disparity_coeffs(rankings,
                                       rels,
                                       group_identities,
                                       position_biases,
                                       skip_zero=False):
        group_disparities = []
        for j, ranking in enumerate(rankings):
            group_disparities.append(
                GroupFairnessLoss.compute_group_disparity(
                    ranking, rels, group_identities, position_biases,
                    skip_zero))
        return 2 * np.mean(group_disparities) * np.array(group_disparities)

    @staticmethod
    def compute_asym_group_disparity_coeffs(rankings,
                                            rels,
                                            group_identities,
                                            position_biases,
                                            skip_zero=False):
        """
        compute disparity and then compute the gradient coefficients for
        asymmetric group disaprity loss
        """
        # compute average v_i/r_i for each group, then the group which has higher relevance
        group_disparities = []
        sign = +1 if np.mean(rels[group_identities == 0]) >= np.mean(
            rels[group_identities == 1]) else -1
        for j, ranking in enumerate(rankings):
            group_disparities.append(
                GroupFairnessLoss.compute_group_disparity(
                    ranking, rels, group_identities, position_biases,
                    skip_zero))
        indicator = (sign * np.mean(group_disparities)) > 0
        return sign * indicator * np.array(group_disparities)
