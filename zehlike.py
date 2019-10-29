import torch
from utils import shuffle_combined
import numpy as np
from YahooDataReader import YahooDataReader
from models import NNModel, LinearModel
from utils import parse_my_args_reinforce, torchify
from evaluation import evaluate_model
from baselines import vvector
from progressbar import progressbar


def demographic_parity_train(model, dr, vdr, vvector, args):
    feat, rel = dr.data
    N = len(rel)
    from utils import get_optimizer
    optimizer = get_optimizer(
        model.parameters(),
        args.lr[0],
        args.optimizer,
        weight_decay=args.weight_decay[0])

    for epoch in range(args.epochs[0]):
        for i in range(N):
            feat, rel = shuffle_combined(feat, rel)
            optimizer.zero_grad()
            curr_feats = feat[i]
            scores = model(torchify(curr_feats)).squeeze()
            probs = torch.nn.Softmax(dim=0)(scores)
            if np.sum(rel[i]) == 0:
                continue
            normalized_rels = rel[i]  # / np.sum(rel[i])
            # np.random.shuffle(normalized_rels)
            ranking_loss = -torch.sum(
                torch.FloatTensor(normalized_rels) * torch.log(probs))

            # print(scores, probs,
            #       torch.log(probs), normalized_rels,
            #       torch.log(probs) * torch.FloatTensor(normalized_rels),
            #       ranking_loss)

            exposures = vvector[0] * probs
            groups = curr_feats[:, args.group_feat_id]
            if np.all(groups == 0) or np.all(groups == 1):
                fairness_loss = 0.0
            else:
                avg_exposure_0 = torch.sum(
                    torch.FloatTensor(1 - groups) * exposures) / torch.sum(
                        1 - torch.FloatTensor(groups))
                avg_exposure_1 = torch.sum(
                    torch.FloatTensor(groups) * exposures) / torch.sum(
                        torch.FloatTensor(groups))
                # print(avg_exposure_0, avg_exposure_1)
                fairness_loss = torch.pow(
                    torch.clamp(avg_exposure_1 - avg_exposure_0, min=0), 2)
            loss = args.lambda_reward * ranking_loss + args.lambda_group_fairness * fairness_loss

            loss.backward()
            optimizer.step()
            # break

            if i % args.evaluate_interval == 0 and i != 0:
                results = evaluate_model(
                    model,
                    vdr,
                    fairness_evaluation=False,
                    group_fairness_evaluation=True,
                    deterministic=True,
                    args=args,
                    num_sample_per_query=100)
                print(results)
    return model


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    import pickle as pkl
    dr = YahooDataReader(None)
    dr.data = pkl.load(open("GermanCredit/german_train_rank.pkl", "rb"))
    vdr = YahooDataReader(None)
    vdr.data = pkl.load(open("GermanCredit/german_test_rank.pkl", "rb"))
    args = parse_my_args_reinforce()
    torch.set_num_threads(args.num_cores)
    args.group_feat_id = 3
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

    model = demographic_parity_train(model, dr, vdr, vvector(200), args)

    results = evaluate_model(
        model,
        vdr,
        fairness_evaluation=False,
        group_fairness_evaluation=True,
        deterministic=True,
        args=args,
        num_sample_per_query=100)
    print(results)
