import torch
import numpy as np
import sys
import math
import os

from models import NNModel, LinearModel

# from tqdm import tqdm

from tensorboardX import SummaryWriter
from progressbar import progressbar

from utils import logsumexp, parse_my_args_reinforce, shuffle_combined, torchify
from YahooDataReader import YahooDataReader
#from log_likelihood_training import log_likelihood_training

from evaluation import compute_dcg, evaluate_model, sample_ranking, compute_average_rank
from models import convert_vars_to_gpu

from fairness_loss import (get_expected_exposure, minimize_for_k,
                           IndividualFairnessLoss, GroupFairnessLoss)

from utils import exp_lr_scheduler


def log_and_print(model,
                  data_reader,
                  writer,
                  epoch,
                  iteration,
                  epoch_length,
                  name="val",
                  experiment_name=None,
                  gpu_id=None,
                  fairness_evaluation=False,
                  exposure_relevance_plot=False,
                  deterministic=True,
                  group_fairness_evaluation=False,
                  args=None):

    results = evaluate_model(
        model,
        data_reader,
        deterministic=deterministic,
        gpu_id=args.gpu_id,
        fairness_evaluation=fairness_evaluation,
        position_bias_vector=1. / np.log2(2 + np.arange(200)),
        writer=writer if exposure_relevance_plot else None,
        epoch_num=epoch,
        group_fairness_evaluation=group_fairness_evaluation,
        args=args)
    """
    Evaluate
    """
    if fairness_evaluation:
        (avg_l1_dists, avg_rsq, avg_residuals, scale_inv_mse,
         asymmetric_disparity) = (results["avg_l1_dists"], results["avg_rsq"],
                                  results["avg_residuals"],
                                  results["scale_inv_mse"],
                                  results["asymmetric_disparity"])
    if group_fairness_evaluation:
        avg_group_exposure_disparity, avg_group_asym_disparity = results[
            "avg_group_disparity"], results["avg_group_asym_disparity"]
    avg_ndcg, avg_dcg, average_rank, avg_err = results["ndcg"], results[
        "dcg"], results["avg_rank"], results["err"]
    step = epoch * epoch_length + iteration
    """
    Log
    """
    if experiment_name is None:
        experiment_name = "/"
    else:
        experiment_name += "/"
    if writer is not None:
        writer.add_scalars(experiment_name + "ndcg",
                           {name + '_average_ndcg': avg_ndcg}, step)
        writer.add_scalars(experiment_name + "rank",
                           {name + '_average_rank': average_rank}, step)
        writer.add_scalars(experiment_name + "dcg",
                           {name + '_average_dcg': avg_dcg}, step)
        writer.add_scalars(experiment_name + "err",
                           {name + '_average_err': avg_err}, step)
        if fairness_evaluation:

            # writer.add_scalars(experiment_name + "kl_div",
            #                    {name + '_average_kl_divergence':
            #                     avg_kl_div}, step)
            # writer.add_scalars(experiment_name + "entropy",
            #                    {name + '_average_entropy': avg_entropy}, step)
            writer.add_scalars(experiment_name + "l1_dist",
                               {name + '_average_l1_dist': avg_l1_dists}, step)
            writer.add_scalars(experiment_name + "r_sq",
                               {name + '_average_r_sq': avg_rsq}, step)
            writer.add_scalars(experiment_name + "residuals",
                               {name + '_average_residuals':
                                avg_residuals}, step)
            writer.add_scalars(experiment_name + "scale_inv_mse", {
                name + '_average_scale_inv_mse': scale_inv_mse
            }, step)
            writer.add_scalars(experiment_name + "asymmetric_disparity", {
                name + '_asymmetric_disparity':
                asymmetric_disparity
            }, step)
        if group_fairness_evaluation:
            writer.add_scalars(experiment_name + "avg_group_disparity", {
                name + "_average_group_disparity":
                avg_group_exposure_disparity
            }, step)
            writer.add_scalars(experiment_name + "avg_group_asym_disparity", {
                name + "_average_group_asym_disparity":
                avg_group_asym_disparity
            }, step)
    word = "Validation" if name == "val" else "Train"
    """
    Print
    """
    print("Epoch {}, Average {}: NDCG: {}, DCG {}, Average Rank {}, ERR {}".
          format(epoch, word, avg_ndcg, avg_dcg, average_rank, avg_err))
    if fairness_evaluation:
        print("Average {} "
              "L1 distance: {:.6f}, R-squared value: {:.6f}, "
              "Residuals: {:.6f}, Scale invariant MSE: {:.6f}, "
              "Avg Asymmetric Disparity: {:.6f}".format(
                  word, avg_l1_dists, avg_rsq, avg_residuals, scale_inv_mse,
                  asymmetric_disparity))
    if group_fairness_evaluation:
        print(
            "Average {} Group Exposure disparity: {}, Group Asymmetric disparity: {}".
            format(word, avg_group_exposure_disparity,
                   avg_group_asym_disparity, avg_group_asym_disparity))
    """
    Return
    """
    returned = args.lambda_reward * avg_ndcg
    if args.lambda_group_fairness > 0:
        returned -= args.lambda_group_fairness * avg_group_asym_disparity
    if args.lambda_ind_fairness > 0:
        returned -= args.lambda_ind_fairness * asymmetric_disparity
    return returned


def on_policy_training(yahoo_data_reader,
                       validation_data_reader,
                       model,
                       experiment_name=None,
                       writer=None,
                       args=None):
    position_bias_vector = 1. / np.log2(2 + np.arange(200))
    lr = args.lr
    num_epochs = args.epochs
    weight_decay = args.weight_decay
    sample_size = args.sample_size

    print("Starting training with the following config")
    print(
        "Learning rate {}, Weight decay {}, Sample size {}\n"
        "Lambda_reward: {}, lambda_ind_fairness:{}, lambda_group_fairness:{}".
        format(lr, weight_decay, sample_size, args.lambda_reward,
               args.lambda_ind_fairness, args.lambda_group_fairness))
    if writer is None and args.summary_writing:
        writer = SummaryWriter(log_dir='runs')
    from utils import get_optimizer
    optimizer = get_optimizer(model.parameters(), lr, args.optimizer,
                              weight_decay)
    train_feats, train_rel = yahoo_data_reader.data
    len_train_set = len(train_feats)
    fairness_evaluation = True if args.lambda_ind_fairness > 0.0 else False
    group_fairness_evaluation = True if args.lambda_group_fairness > 0.0 else False

    if args.early_stopping:
        time_since_best = 0
        best_metric = 0.0
    for epoch in range(num_epochs):

        # # training
        print("Training....")
        if args.lr_scheduler and epoch >= 1:
            optimizer = exp_lr_scheduler(
                optimizer, epoch, lr, decay_factor=args.lr_decay)
        args.entropy_regularizer = args.entreg_decay * args.entropy_regularizer
        epoch_rewards_list = []
        running_ndcgs_list = []
        running_dcgs_list = []
        fairness_losses = []
        variances = []
        # shuffle(file_list)
        train_feats, train_rel = shuffle_combined(train_feats, train_rel)

        iterator = progressbar(
            range(len_train_set)) if args.progressbar else range(len_train_set)
        for i in iterator:
            if i % args.evaluate_interval == 0:
                if i != 0:
                    print(
                        "\nAverages of last 1000 rewards: {}, ndcgs: {}, dcgs: {}".
                        format(
                            np.mean(epoch_rewards_list[
                                -min([len(epoch_rewards_list), 1000]):]),
                            np.mean(running_ndcgs_list[
                                -min([len(running_dcgs_list), 1000]):]),
                            np.mean(running_dcgs_list[
                                -min([len(running_dcgs_list), 1000]):])))
                    exposure_relevance_plot = False
                else:
                    exposure_relevance_plot = False
                print(
                    "Evaluating on validation set: iteration {}/{} of epoch {}".
                    format(i, len_train_set, epoch))
                curr_metric = log_and_print(
                    model,
                    validation_data_reader,
                    writer,
                    epoch,
                    i,
                    len_train_set,
                    "val",
                    experiment_name,
                    args.gpu_id,
                    fairness_evaluation=fairness_evaluation,
                    exposure_relevance_plot=exposure_relevance_plot,
                    deterministic=args.validation_deterministic,
                    group_fairness_evaluation=group_fairness_evaluation,
                    args=args)
                # """
                # Early stopping
                # """
                if args.early_stopping:
                    if curr_metric >= best_metric:
                        best_metric = curr_metric
                        time_since_best = 0
                    elif curr_metric <= best_metric * 0.99:
                        time_since_best += 1
                    if time_since_best >= 5:
                        print(
                            "Validation set metric hasn't increased in 5 steps. Exiting"
                        )
                        return model

                # print("Evaluating on training set")
                # log_and_print(model, yahoo_data_reader, writer, epoch, i,
                #               len_train_set, "train", experiment_name,
                #               args.gpu_id, True)

                # feats, rel = yahoo_data_reader.readfile(file)
            feats, rel = train_feats[i], train_rel[i]
            if len(feats) == 1:
                continue
            if args.lambda_group_fairness > 0.0:
                group_identities = np.array(
                    feats[:, args.group_feat_id], dtype=np.int)
            if feats is None:
                continue
            if args.gpu_id is not None:
                feats, rel = convert_vars_to_gpu([feats, rel], args.gpu_id)

            scores = model(torchify(feats))
            probs_ = torch.nn.Softmax(dim=0)(scores)
            probs = probs_.data.numpy().flatten()

            rankings, rewards_list, ndcg_list, dcg_list = [], [], [], []
            # propensities = []
            for j in range(sample_size):
                # ranking, propensity = sample_ranking(
                # np.array(probs, copy=True))
                # print([(param.name, param.data)
                #        for param in model.parameters()], probs)
                ranking = sample_ranking(np.array(probs, copy=True), False)
                rankings.append(ranking)
                # propensities.append(propensity)
                ndcg, dcg = compute_dcg(ranking, rel, args.eval_rank_limit)
                if args.reward_type == "ndcg":
                    rewards_list.append(ndcg)
                elif args.reward_type == "dcg":
                    rewards_list.append(dcg)
                elif args.reward_type == "avrank":
                    avrank = -np.mean(compute_average_rank(ranking, rel))
                    rewards_list.append(np.sum(avrank))
                ndcg_list.append(ndcg)
                dcg_list.append(dcg)
            if args.baseline_type == "value":
                baseline = np.mean(rewards_list)
            elif args.baseline_type == "max":
                state = (rel)
                baseline = compute_baseline(
                    state=state, type=args.baseline_type)
            else:
                print("Choose a valid baseline type! Exiting")
                sys.exit(1)

            # FAIRNESS constraints
            if args.lambda_ind_fairness > 0.0:
                num_docs = len(ranking)
                rel_labels = np.array(rel)
                # relevant_indices_to_onehot(rel, num_docs)
                # relevance_variance = np.var(rel_labels)
                if args.fairness_version == "squared_residual":
                    expected_exposures = get_expected_exposure(
                        rankings, position_bias_vector)
                    k = minimize_for_k(rel_labels, expected_exposures,
                                       args.skip_zero_relevance)
                    disparity_matrix = IndividualFairnessLoss(
                    ).compute_disparities(rankings, rel_labels,
                                          position_bias_vector, k,
                                          args.skip_zero_relevance)
                    marginal_disparity = IndividualFairnessLoss(
                    ).compute_marginal_disparity(
                        disparity_matrix)  # should be size of the ranking set
                    assert len(marginal_disparity) == num_docs, \
                        "Marginal disparity is of the wrong dimension"
                    individual_fairness_coeffs = np.zeros(sample_size)
                    for index in range(sample_size):
                        individual_fairness_coeffs[
                            index] = IndividualFairnessLoss.compute_sq_individual_fairness_loss_coeff(
                                rankings[index], disparity_matrix[index],
                                marginal_disparity, k)
                    fairness_baseline = np.mean(individual_fairness_coeffs)
                    fairness_losses.append(fairness_baseline)
                elif args.fairness_version == "scale_inv_mse":
                    individual_fairness_coeffs = IndividualFairnessLoss(
                    ).get_scale_invariant_mse_coeffs(rankings, rel_labels,
                                                     position_bias_vector,
                                                     args.skip_zero_relevance)
                    fairness_baseline = np.mean(individual_fairness_coeffs
                                                ) if args.use_baseline else 0.0
                    fairness_losses.append(fairness_baseline)
                elif args.fairness_version == "asym_disparity":
                    pdiff = IndividualFairnessLoss.compute_pairwise_disparity_matrix(
                        rankings, rel_labels, position_bias_vector)
                    H_mat = IndividualFairnessLoss.get_H_matrix(rel_labels)
                    sum_h_mat = np.sum(
                        H_mat) + 1e-7  # to prevent Nans when dividing
                    # print(rel_labels, H_mat, sum_h_mat)
                    H_mat = np.tile(H_mat, (len(rankings), 1, 1))
                    pdiff_pi = np.mean(pdiff, axis=0)
                    pdiff_indicator = pdiff_pi > 0
                    pdiff_indicator = np.tile(pdiff_indicator, (len(rankings),
                                                                1, 1))

                    individual_fairness_coeffs = pdiff_indicator * H_mat * pdiff
                    individual_fairness_coeffs = np.sum(
                        individual_fairness_coeffs, axis=(1, 2)) / sum_h_mat
                    # print(pdiff_indicator.shape, H_mat.shape, pdiff_pi.shape,
                    #       pdiff.shape)
                    fairness_baseline = np.mean(individual_fairness_coeffs
                                                ) if args.use_baseline else 0.0

                elif args.fairness_version == "pairwise_disparity":
                    pairwise_disparity_matrix, pair_counts = IndividualFairnessLoss.compute_pairwise_disparity_matrix(
                        rankings,
                        rel_labels,
                        position_bias_vector,
                        conditional=False)
                    marginal_pairwise_disparity_matrix = np.mean(
                        pairwise_disparity_matrix, axis=0)

            if args.lambda_group_fairness > 0.0:
                rel_labels = np.array(rel)
                if np.sum(rel_labels[group_identities == 0]) == 0 or np.sum(
                        rel_labels[group_identities == 1]) == 0:
                    skip_this_query = True
                else:
                    skip_this_query = False
                    group_fairness_coeffs = GroupFairnessLoss.compute_group_fairness_coeffs_generic(
                        rankings, rel_labels, group_identities,
                        position_bias_vector, args.group_fairness_version,
                        args.skip_zero_relevance)

                    fairness_baseline = np.mean(np.mean(group_fairness_coeffs))

                # log the reward/dcg variance
            variances.append(np.var(rewards_list))
            epoch_rewards_list.append(np.mean(rewards_list))
            running_ndcgs_list.append(np.mean(ndcg_list))
            running_dcgs_list.append(np.mean(dcg_list))

            if i % 1000 == 0 and i != 0:
                if experiment_name is None:
                    experiment_name = ""
                if writer is not None:
                    writer.add_scalars(experiment_name + "/var_reward",
                                       {"var_reward": np.mean(variances)},
                                       epoch * len_train_set + i)
                    if fairness_evaluation:
                        writer.add_scalars(
                            experiment_name + "/mean_fairness_loss", {
                                "mean_fairness_loss": np.mean(fairness_losses)
                            }, epoch * len_train_set + i)
                variances = []
                fairness_losses = []
            optimizer.zero_grad()
            for j in range(sample_size):
                ranking = rankings[j]
                reward = rewards_list[j]

                log_model_prob = compute_log_model_probability(
                    scores, ranking, args.gpu_id)
                if args.use_baseline:
                    reinforce_loss = float(args.lambda_reward * -(
                        reward - baseline)) * log_model_prob
                else:
                    reinforce_loss = args.lambda_reward * log_model_prob * -reward
                if args.lambda_ind_fairness != 0.0:
                    if (args.fairness_version == "squared_residual") or (
                            args.fairness_version == "scale_inv_mse"):
                        individual_fairness_cost = float(
                            args.lambda_ind_fairness *
                            (individual_fairness_coeffs[j] - fairness_baseline
                             )) * log_model_prob
                    elif args.fairness_version == "cross_entropy":
                        individual_fairness_cost = float(
                            args.lambda_ind_fairness * IndividualFairnessLoss.
                            compute_cross_entropy_fairness_loss(
                                ranking, rel_labels, expected_exposures,
                                position_bias_vector)) * log_model_prob
                    elif args.fairness_version == "asym_disparity":
                        individual_fairness_cost = float(
                            args.lambda_ind_fairness *
                            (individual_fairness_coeffs[j] - fairness_baseline
                             )) * log_model_prob
                    elif args.fairness_version == "pairwise_disparity":
                        individual_fairness_cost = float(
                            args.lambda_ind_fairness *
                            (np.sum(2 * marginal_pairwise_disparity_matrix *
                                    pairwise_disparity_matrix[j]) / pair_counts
                             )) * log_model_prob
                    else:
                        print("Use a valid version of fairness constraints")
                    reinforce_loss += individual_fairness_cost
                if args.lambda_group_fairness != 0.0 and not skip_this_query:
                    group_fairness_cost = float(
                        args.lambda_group_fairness * group_fairness_coeffs[j]
                    ) * log_model_prob
                    reinforce_loss += group_fairness_cost
                # debias the loss because the model gets updated every sampled ranking
                # i.e. log_model_prob is biased
                # if debias_training:
                #     bias_corrections.append(
                #         math.exp(log_model_prob.data) / propensities[j])
                #     reinforce_loss *= bias_corrections[-1]
                # ^ not reqd anymore
                reinforce_loss.backward(retain_graph=True)
            if args.entropy_regularizer > 0.0:
                entropy_loss = args.entropy_regularizer * (
                    -get_entropy(probs_))
                entropy_loss.backward()
            optimizer.step()
        if args.save_checkpoints:
            if epoch == 0 and not os.path.exists(
                    "models/{}".format(experiment_name)):
                os.makedirs("models/{}/".format(experiment_name))
            torch.save(model, "models/{}/epoch{}.ckpt".format(
                experiment_name, epoch))
    log_and_print(
        model,
        validation_data_reader,
        writer,
        epoch,
        i,
        len_train_set,
        "val",
        experiment_name,
        args.gpu_id,
        fairness_evaluation=fairness_evaluation,
        exposure_relevance_plot=exposure_relevance_plot,
        deterministic=args.validation_deterministic,
        group_fairness_evaluation=group_fairness_evaluation,
        args=args)
    return model


# def get_entropy(propensity):
#     return -propensity * math.log(propensity)


def get_entropy(probs):
    return -torch.sum(torch.log(probs) * probs)


def compute_baseline(state, type="max"):
    if type == "max":
        print("Depracated: Doesn't work anymore")
        rel = state
        max_dcg = 0.0
        for i in range(sum(rel)):
            max_dcg += 1.0 / math.log(2 + i)
        return max_dcg
    elif type == "value":
        rankings, rewards_list = state
        # state is sent as a set of rankings sampled using the policy and
        # the set of relevant documents
        return np.mean(rewards_list)
    else:
        print("-----No valid reward type selected-------")


def compute_log_model_probability(scores, ranking, gpu_id=None):
    """
    more stable version
    if rel is provided, use it to calculate probability only till
    all the relevant documents are found in the ranking
    """
    subtracts = torch.zeros_like(scores)
    log_probs = torch.zeros_like(scores)
    if gpu_id is not None:
        subtracts, log_probs = convert_vars_to_gpu([subtracts, log_probs],
                                                   gpu_id)
    for j in range(scores.size()[0]):
        posj = ranking[j]
        log_probs[j] = scores[posj] - logsumexp(scores - subtracts, dim=0)
        subtracts[posj] = scores[posj] + 1e6
    return torch.sum(log_probs)


if __name__ == "__main__":
    args = parse_my_args_reinforce()
    if args.train_dir is None and args.test_dir is None:
        print("Loading data from pickle files: {}, {}".format(
            args.train_pkl, args.test_pkl))
        from YahooDataReader import reader_from_pickle
        dr = reader_from_pickle(args.train_pkl)
        vdr = reader_from_pickle(args.test_pkl)
    else:
        print("Loading data from directory: {}".format(args.train_dir))
        dr = YahooDataReader(args.train_dir)
        vdr = YahooDataReader(args.test_dir)
        dr.pickelize_data("YahooData/train.pkl")
        vdr.pickelize_data("YahooData/test.pkl")
        # use a outpath if you want to save the data in a pickle file
    if args.pretrained_model:
        model = torch.load(args.pretrained_model)
        print("Initializing the model with model at {}".format(
            args.pretrained_model))
    else:
        if args.model_type == "Linear":
            model = LinearModel(D=args.input_dim, clamp=args.clamp)
            print("Linear model initialized")
        else:
            model = NNModel(
                D=args.input_dim,
                hidden_layer=args.hidden_layer,
                dropout=args.dropout,
                pooling=args.pooling,
                clamp=args.clamp)
            print(
                "Model initialized with {} hidden layer size, Dropout={}, using {} pooling".
                format(args.hidden_layer, args.dropout, args.pooling))
        if args.pretrain:
            model = log_likelihood_training(dr, vdr, model, args.lr[0],
                                            args.epochs[0],
                                            args.weight_decay[0], "pretrain")
        # torch.save(model, "pretrained_model.ckpt")

    if args.gpu_id is not None:
        from models import convert_to_gpu
        model = convert_to_gpu(model, args.gpu_id)
    else:
        torch.set_num_threads(args.num_cores)

    i = 0
    writer = SummaryWriter(log_dir='runs')
    args_ = args
    for lr, epochs, l2, sample_size, baseline_type in zip(
            args.lr, args.epochs, args.weight_decay, args.sample_size,
            args.baseline):
        args_.lr = lr
        args_.epochs = epochs
        args_.weight_decay = l2
        args_.sample_size = sample_size
        args_.baseline_type = baseline_type
        i += 1
        if baseline_type == "none":
            args_.use_baseline = False
        else:
            args_.use_baseline = True
        expname = "lr{}_lrdecay_{}_l2_{}_D_{}".format(lr, args.lr_decay, l2,
                                                      args.hidden_layer)
        print(expname)
        model = on_policy_training(
            dr,
            vdr,
            model,
            experiment_name=args.expname if args.expname else expname,
            writer=writer,
            args=args)

        # torch.save(model, "model_{}.ckpt".format(i))
