import numpy as np


def construct_features(user_features, item_features):
    return np.outer(np.append(user_features, 1), np.append(item_features,
                                                           1)).flatten()


def create_candidate_sets_masks(candidate_sets):
    max_len = 0
    for cand_set in candidate_sets:
        max_len = max(max_len, len(cand_set))

    candidate_sets_mask = np.zeros(
        (len(candidate_sets), max_len), dtype=np.int)
    candidate_sets_new = np.zeros((len(candidate_sets), max_len), dtype=np.int)

    for i, cand_set in enumerate(candidate_sets):
        candidate_sets_new[i, 0:len(cand_set)] = cand_set
        candidate_sets_mask[i, 0:len(cand_set)] = np.ones(len(cand_set))
    return candidate_sets_new, candidate_sets_mask


def preprocess_data_2(rating_threshold=3,
                      sensitive_feature_id=0,
                      train_test_ratio=0.8):
    """
    Preprocess coat dataset into a set of ((u,q), d, C)-- user, query, document, and
    candidate set tuples.

    @param: add_leaky_feature
    @param: leaky_feat_sd
    @param sensitive_feature_id -- the index of the sensitive feature\
                                    in the feature vector of the coat
    @param: drop_less_preferred -- percentage of datapoints to drop from training set
                                    where the less_preferred is winning
    @param: drop_items_from_less_preferred_group -- percentage of items to be dropped
                                    from the dataset that belongs to the less preferred_group
    """
    coat_data = np.loadtxt(
        "/home/ashudeep/projects/fair-rank/coat/user_item_features/item_features.ascii"
    )
    user_data = np.loadtxt(
        "/home/ashudeep/projects/fair-rank/coat/user_item_features/user_features.ascii"
    )
    ratings = np.loadtxt("/home/ashudeep/projects/fair-rank/coat/train.ascii")

    # ratings above 3 means relevant, ratings below 3 or equal to 3 means irrelevant
    num_users = ratings.shape[0]
    num_coats = ratings.shape[1]
    feature_matrix = np.zeros((num_users, num_coats, 510))
    filled = np.zeros((num_users, num_coats))

    x = []
    y = []
    candidate_sets = []
    group = []

    test_x, test_y, test_candidate_sets, test_group = [], [], [], []

    for i in range(num_users):
        for j in range(num_coats):
            if ratings[i, j] > 3:
                candidate_set = []
                for k in range(num_coats):
                    if k != j and ratings[i, k] != 0:
                        # do we need to add all the other docs except missing
                        candidate_set.append(k)
                if np.random.random() < train_test_ratio:
                    x.append(i)
                    y.append(j)
                    candidate_sets.append(candidate_set)
                    group.append(int(coat_data[j, sensitive_feature_id]))
                else:
                    test_x.append(i)
                    test_y.append(j)
                    test_candidate_sets.append(candidate_set)
                    test_group.append(int(coat_data[j, sensitive_feature_id]))
            if filled[i, j] == 0:
                feature_matrix[i, j] = construct_features(
                    user_data[i], coat_data[j])
                filled[i, j] = 1

    candidate_sets, candidate_sets_mask = create_candidate_sets_masks(
        candidate_sets)
    test_candidate_sets, test_candidate_sets_mask = create_candidate_sets_masks(
        test_candidate_sets)
    return {
        "x": np.array(x),
        "y": np.array(y),
        "candidate_sets": np.array(candidate_sets),
        "candidate_sets_mask": np.array(candidate_sets_mask),
        "test_x": np.array(test_x),
        "test_y": np.array(test_y),
        "test_candidate_sets": np.array(test_candidate_sets),
        "test_candidate_sets_mask": np.array(test_candidate_sets_mask),
        "feature_matrix": feature_matrix,
        "user_data": user_data,
        "item_data": coat_data,
        "groups": np.array(group, dtype=np.int),
        "test_groups": np.array(test_group, dtype=np.int)
    }


# def preprocess_data_3(rating_threshold=3,
#                       sensitive_feature_id=0,
#                       train_test_ratio=0.8):
#     """
#     Preprocess coat dataset into (xfeats, yfeats, xdata, ydata, relevance)
#     xfeats: features of the users/queries
#     yfeats: features of the items
#     xdata: user indices
#     ydata: item indices
#     relevance: binary relevance vector
#
#     @param sensitive_feature_id -- the index of the sensitive feature\
#                                     in the feature vector of the coat
#     """
#     coat_data = np.loadtxt(
#         "/home/ashudeep/projects/fair-rank/coat/user_item_features/item_features.ascii"
#     )
#     user_data = np.loadtxt(
#         "/home/ashudeep/projects/fair-rank/coat/user_item_features/user_features.ascii"
#     )
#     ratings = np.loadtxt("/home/ashudeep/projects/fair-rank/coat/train.ascii")
#
#     # ratings above 3 means relevant, ratings below 3 or equal to 3 means irrelevant
#     num_users = ratings.shape[0]
#     num_coats = ratings.shape[1]
#     xfeats = user_data
#     yfeats = coat_data
#     xdata = []
#     ydata = []
#     relevance = np.array(
#         ratings > rating_threshold, dtype=np.int)  # assuming full information
#     filled = np.zeros((num_users, num_coats))
#     for i in range(num_users):
#         for j in range(num_coats):
