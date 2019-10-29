import numpy as np
from torch.autograd import Variable
import torch
import argparse
import random


# from loss_utils import get_z_ids
def shuffle_combined(a, b):
    combined = list(zip(a, b))
    random.shuffle(combined)

    a[:], b[:] = zip(*combined)
    return a, b


def compute_test_errors(test_z, predictions):
    errors = [[0, 0], [0, 0]]
    total_errors = 0
    for i in [0, 1]:
        for j in [0, 1]:
            for i1 in test_z[i][j]:
                if not predictions[i1]:
                    errors[i][j] += 1
                    total_errors += 1
    for i in [0, 1]:
        for j in [0, 1]:
            errors[i][j] = float(errors[i][j]) / len(test_z[i][j])
    total_errors = float(total_errors) / len(predictions)
    return (total_errors, errors)


def get_error_rates(test_x, test_z_assignment, model):
    predictions = model(Variable(torch.FloatTensor(test_x[:, 0, :]))) - model(
        Variable(torch.FloatTensor(test_x[:, 1, :])))
    total_error = (torch.mean((predictions < 0).float())).data[0]
    # z_ass_1, z_ass_2, _, _, _, _ = get_z_ids(constraint_type)
    z0 = np.argwhere(np.asarray(test_z_assignment == 0)).flatten()
    z1 = np.argwhere(np.asarray(test_z_assignment == 1)).flatten()
    z2 = np.argwhere(np.asarray(test_z_assignment == 2)).flatten()
    z3 = np.argwhere(np.asarray(test_z_assignment == 3)).flatten()
    error0 = (torch.mean(
        (predictions < 0)[torch.LongTensor(z0)].float())).data[0]
    error1 = (torch.mean(
        (predictions < 0)[torch.LongTensor(z1)].float())).data[0]
    error2 = (torch.mean(
        (predictions < 0)[torch.LongTensor(z2)].float())).data[0]
    error3 = (torch.mean(
        (predictions < 0)[torch.LongTensor(z3)].float())).data[0]
    return total_error, error0, error1, error2, error3


def parse_my_args():
    parser = argparse.ArgumentParser(
        description='Train a fair ranking classifier')
    parser.add_argument(
        '--gen',
        dest='generate_dataset',
        action='store_true',
        help='Whether to generate a new dataset or used a stored one')
    parser.add_argument(
        '--run-name',
        dest='run_name',
        type=str,
        default=None,
        help='Identifier that the logs from the\
         current run will be stored under.\
         Default: (current date and time)')
    parser.add_argument(
        '--leaky',
        dest='add_leaky_feature',
        action='store_true',
        help='Add a leaky feature that favors type 2 or type 0 errors\
         over type 1 and type 3')
    hyperparam_g = parser.add_argument_group(
        "hyperparams",
        "Hyper parameters for the training of the neural network")
    hyperparam_g.add_argument(
        "--c1",
        dest='c1',
        type=float,
        default=1.0,
        help="L2 regularization coefficient")
    hyperparam_g.add_argument(
        "--c2",
        dest='c2',
        type=float,
        default=1.0,
        help="Fairness constraint coefficient")
    hyperparam_g.add_argument(
        "--c",
        dest='c',
        type=float,
        default=1.0,
        help="Margin loss coefficient")
    hyperparam_g.add_argument(
        "--lr", dest='lr', type=float, default=0.00001, help="learning rate")
    hyperparam_g.add_argument(
        '--epochs',
        dest='epochs',
        type=int,
        default=1000,
        help="Number of training epochs")
    hyperparam_g.add_argument(
        '-b', dest='batchsize', type=int, default=128, help="Batchsize")
    hyperparam_g.add_argument(
        '--ctype',
        dest='constraint_type',
        type=int,
        default=1,
        help="Constraint Type. "
        "Default=1 means equate G0 vs G1 errors to G1 vs G0 errors")

    parser.set_defaults(verbose=False)
    parser.set_defaults(generate_dataset=False)
    args = parser.parse_args()
    if args.add_leaky_feature and args.generate_dataset:
        print("Generating the dataset again. This might take a while....")
        print(
            "(If you are sure that there is a leaky feature data already in pkl,"
            " you can turn off the generate_dataset option)")
    return args


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def parse_my_args_reinforce():
    parser = argparse.ArgumentParser(
        description='Reinforce algorithm for learning to rank')
    parser.add_argument(
        '--train',
        dest='train_dir',
        type=str,
        default=None,
        help='training directory')
    parser.add_argument(
        '--test',
        dest='test_dir',
        type=str,
        default=None,
        help='test directory')
    parser.add_argument(
        '--train_pkl',
        dest='train_pkl',
        type=str,
        default="yahoo/train.pkl",
        help='training directory')
    parser.add_argument(
        '--test_pkl',
        dest='test_pkl',
        type=str,
        default="yahoo/test.pkl",
        help='test directory')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0)
    hyperparam_g = parser.add_argument_group(
        "hyperparams",
        "Hyper parameters for the training of the neural network")
    hyperparam_g.add_argument(
        "--hidden",
        dest='hidden_layer',
        type=int,
        default=128,
        help="Size of hidden layer (Default=128)")
    hyperparam_g.add_argument(
        "--cores",
        dest='num_cores',
        type=int,
        default=1,
        help="Number of CPU cores used (Default=1)")
    hyperparam_g.add_argument(
        "-D", dest='input_dim', type=int, default=700, help="Input dimensions")
    hyperparam_g.add_argument(
        "--lr",
        dest='lr',
        type=str,
        default="0.00001",
        help="learning rate(s)")
    hyperparam_g.add_argument(
        '--epochs',
        dest='epochs',
        type=str,
        default="20",
        help="Number of training epochs")
    hyperparam_g.add_argument(
        "--l2",
        dest='weight_decay',
        type=str,
        default="0.000",
        help="Lambda for weight decay")
    hyperparam_g.add_argument(
        "--sample_size",
        dest='sample_size',
        type=str,
        default="10",
        help="Sample size")
    parser.add_argument(
        "--pretrain",
        dest='pretrain',
        action='store_true',
        default=False,
        help="Pretrain with log likelihood")
    parser.add_argument(
        "--check",
        dest='save_checkpoints',
        action='store_true',
        default=False,
        help="Save checkpoint for every epoch")
    parser.add_argument(
        "--noprogressbar",
        dest='progressbar',
        action='store_false',
        default=True,
        help="whether to use progressbar for training/validation progress")
    hyperparam_g.add_argument(
        '--baseline',
        dest='baseline',
        type=str,
        default="value",
        help="Which baseline to use. Options: none/value/max")
    parser.add_argument(
        '--initial_model',
        dest='pretrained_model',
        type=str,
        default=None,
        help="Use the model on this path as the pretrained initial model")
    parser.add_argument(
        '--gpu',
        dest='gpu_id',
        type=int,
        default=None,
        help="GPU id (default = None --> use CPU only)")
    parser.add_argument(
        '--expname',
        dest='expname',
        type=str,
        default=None,
        help="Name of the experiment. Used for logging purposes only right now"
    )
    parser.add_argument(
        '--entreg',
        dest='entropy_regularizer',
        type=float,
        default=0.0,
        help="Lambda for entropy regularization")
    parser.add_argument(
        '--reward_type',
        dest='reward_type',
        type=str,
        default="ndcg",
        help="Reward type: Choose out of dcg/ndcg/avrank")
    parser.add_argument(
        '--eval_int',
        dest='evaluate_interval',
        type=int,
        default=2000,
        help="Evaluate after these many number of steps")
    parser.add_argument(
        '--lindf',
        dest='lambda_ind_fairness',
        type=float,
        default=0.0,
        help="Lambda for the individual fairness cost")
    parser.add_argument(
        '--lgf',
        dest='lambda_group_fairness',
        type=float,
        default=0.0,
        help="Lambda for the group fairness cost")
    parser.add_argument(
        '--lreward',
        dest='lambda_reward',
        type=float,
        default=1.0,
        help="Lambda for reward in the REINFORCE style updates."
        " Can be set to 0 to start all fairness training")
    parser.add_argument(
        '--indfv',
        dest='fairness_version',
        type=str,
        default='asym_disparity',
        help="Current options: squared_residual, cross_entropy, scale_inv_mse"
        "pairwise_disparity, asym_disparity")

    parser.add_argument(
        '--gfv',
        dest='group_fairness_version',
        type=str,
        default='asym_disparity',
        help="Current options: sq_disparity, asym_disparity")
    parser.add_argument(
        '--skip_zero',
        dest='skip_zero_relevance',
        action="store_true",
        default=False,
        help=
        "Whether the fairness constraints should skip the documents with zero "
        "relevance out of the fairness loss term")
    parser.add_argument(
        '--lr_scheduler',
        dest='lr_scheduler',
        action="store_true",
        default=False,
        help=
        "If chosen, we do an exponential decay for lr by reducing it by 0.1 every epoch"
    )
    parser.add_argument(
        '--lr_decay',
        dest='lr_decay',
        type=float,
        default=0.0,
        help="How much do you want to reduce the lr in each step by."
        " Requires --lr_scheduler to be used.")
    parser.add_argument(
        '--summary',
        dest='summary_writing',
        action="store_true",
        default=False,
        help="Whether to write summaries into tensorboardX logs")
    parser.add_argument(
        '--group_feat_id',
        dest='group_feat_id',
        type=int,
        default=0,
        help="index of the feature that contains the group id of the document")
    parser.add_argument(
        '--entreg_decay',
        dest='entreg_decay',
        type=float,
        default=1.0,
        help='How much does entropy regularizer drop by after each epoch')
    # parser.add_argument(
    #     '--macro',
    #     dest='macro_avg',
    #     action='store_true',
    #     default=False,
    #     help=
    #     "Average over the numbers from all the queries rather than
    # micro(average over diffeernt documents and then average over queries)"
    # )
    parser.add_argument(
        '-k',
        dest='eval_rank_limit',
        type=int,
        default=1000,
        help='Maximum rank uptil which the dcg is computed')
    parser.add_argument(
        '--evalk',
        dest='evalk',
        type=int,
        default=1000,
        help=
        'Maximum rank uptil which the dcg is computed (only while computing)')
    parser.add_argument(
        '--pooling',
        dest='pooling',
        type=str,
        default='concat_avg',
        help="whether to use the average or max of the candidate set or not")
    parser.add_argument(
        '--optimizer',
        dest='optimizer',
        type=str,
        default='Adam',
        help="Which optimizer to use")
    parser.add_argument(
        '--early',
        dest='early_stopping',
        action="store_true",
        default=False,
        help="Whether to do early stopping or not (on NDCG)")
    parser.add_argument(
        '--det',
        dest="validation_deterministic",
        action="store_true",
        default=False,
        help="Whether the validation runs use "
        "the deterministic policy or stochastic.")
    parser.add_argument(
        '--model',
        dest='model_type',
        type=str,
        default='NN',
        help="Which model type to use: NN or Linear. Default:NN")
    parser.add_argument(
        '--clamp',
        dest='clamp',
        action="store_true",
        default=False,
        help="Whether the model output is clamped or not")
    parser.add_argument(
        '--eval_temp',
        dest="eval_temperature",
        type=float,
        default=1.0,
        help=
        "When evaluating the policy, what temperature to use in the softmax")

    args = parser.parse_args()
    args = postprocess_args(args)
    return args


def postprocess_args(args):
    args.lr = [float(s) for s in args.lr.split(',')]
    args.baseline = args.baseline.split(',')
    args.sample_size = [int(s) for s in args.sample_size.split(',')]
    args.epochs = [int(s) for s in args.epochs.split(',')]

    args.weight_decay = [float(s) for s in args.weight_decay.split(',')]
    lens = [
        len(args.lr),
        len(args.sample_size),
        len(args.epochs),
        len(args.weight_decay),
        len(args.baseline)
    ]
    if np.any(np.array(lens) - max(lens) != 0):
        print("The number of args for Learning rate, sample size, epochs, "
              "baseline type and weight decay "
              "should be the same")
        import sys
        sys.exit(1)
    if args.eval_rank_limit < 1000:
        args.evalk = args.eval_rank_limit
        print("Override evalk with eval_rank_limit")
    return args


def torchify(u):
    return Variable(torch.FloatTensor(u))


def get_optimizer(params, lr, name="Adam", weight_decay=False, momentum=0.9):
    from torch import optim
    if name == "Adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "Adadelta":
        return optim.Adadelta(params, lr, weight_decay=weight_decay)
    else:
        return optim.SGD(params, lr, momentum, weight_decay=weight_decay)


def exp_lr_scheduler(optimizer,
                     epoch,
                     init_lr=0.001,
                     decay_factor=0.1,
                     lr_decay_epoch=1):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (decay_factor**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


# """
# The code below makes the print command work well with tqdm
# """
# import inspect
# from tqdm import tqdm
#
# # store builtin print
# old_print = print
#
#
# def new_print(*args, **kwargs):
#     # if tqdm.tqdm.write raises error, use builtin print
#     try:
#         tqdm.write(*args, **kwargs)
#     except:
#         old_print(*args, **kwargs)
#
#
# # globaly replace print with new_print
# inspect.builtins.print = new_print
# """
# """
