import numpy as np


def preprocess_data(add_leaky_feature=False,
                    leaky_feat_sd=0.1,
                    preferred_group=0,
                    sensitive_feature_id=0,
                    drop_less_preferred_group_datapoints=0.0,
                    drop_items_from_less_preferred_group=0.0,
                    drop_items_proportional_to_rating=False,
                    create_feature_matrix=False):
    """
    Preprocess coat dataset into a pairwise ranking dataset.

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

    if drop_items_from_less_preferred_group:
        if drop_items_proportional_to_rating:
            avg_ratings = np.sum(
                ratings, axis=0) / np.sum(
                    ratings != 0, axis=0)
            avg_ratings = (avg_ratings / np.sum(avg_ratings))
            probs = avg_ratings * len(
                avg_ratings) * drop_items_from_less_preferred_group
            items_to_drop = (np.random.random(coat_data.shape[0]) < probs) * (
                coat_data[:, sensitive_feature_id] != preferred_group)
        else:
            items_to_drop = (np.random.random(
                coat_data.shape[0]) < drop_items_from_less_preferred_group) * (
                    coat_data[:, sensitive_feature_id] != preferred_group)
        coat_data = coat_data[items_to_drop == 0, :]
        ratings = ratings[:, items_to_drop == 0]

    num_users = ratings.shape[0]
    num_coats = ratings.shape[1]

    feature_matrix = np.zeros((num_users, num_coats, 496))
    filled = np.zeros((num_users, num_coats))
    group = np.ones((num_users, num_coats))*10000

    # omat = np.loadtxt("/home/ashudeep/projects/fair-rank/coat/")

    # the index of the sensitive feature in the feature vector of the coat

    # get ranking data
    p = 0.6  # percentage pairs to sample
    ranking_data = []
    test_ranking_data = []

    test_count = count = 0

    test_z = [[[], []], [[], []]]
    test_z_assignment = []
    z = [[[], []], [[], []]]
    z_assignment = []
    counts = [[0, 0], [0, 0]]
    test_counts = [[0, 0], [0, 0]]
    for i in range(num_users):
        for j in range(num_coats):
            for k in range(num_coats):
                if (ratings[i, j] == 0 or j == k or ratings[i, k] == 0) or (
                        ratings[i, j] == ratings[i, k]):
                    continue
                elif ratings[i, j] > ratings[i, k]:
                    type_of_pair = 2 * int(coat_data[j, 0]) + int(
                        coat_data[k, 0])

                    if add_leaky_feature:
                        if coat_data[j,
                                     sensitive_feature_id] == preferred_group:
                            new_feat1 = np.random.normal(
                                ratings[i, j], leaky_feat_sd)
                        else:
                            new_feat1 = np.random.normal(0.0, leaky_feat_sd)
                        features_j = np.append(
                            np.outer(np.append(user_data[i], 1),
                                     coat_data[j]).flatten(), new_feat1)
                        if coat_data[k,
                                     sensitive_feature_id] == preferred_group:
                            new_feat2 = np.random.normal(
                                ratings[i, k], leaky_feat_sd)
                        else:
                            new_feat2 = np.random.normal(0.0, leaky_feat_sd)

                        features_k = np.append(
                            np.outer(np.append(user_data[i], 1),
                                     coat_data[k]).flatten(), new_feat2)
                    else:
                        features_j = np.outer(
                            np.append(user_data[i], 1),
                            coat_data[j]).flatten()
                        features_k = np.outer(
                            np.append(user_data[i], 1),
                            coat_data[k]).flatten()
                    if not filled[i,j]:
                        feature_matrix[i, j] = features_j
                        filled[i,j] = True
                        group[i,j] = coat_data[j, sensitive_feature_id]
                    if not filled[i, k]:
                        feature_matrix[i, k] = features_k
                        filled[i,k] = True
                        group[i,k] = coat_data[k, sensitive_feature_id]
                    add_to_training = False
                    if np.random.random() < p:
                        # whether to put in training or validation/test
                        if coat_data[j,
                                     sensitive_feature_id] != preferred_group and np.random.random(
                                     ) < drop_less_preferred_group_datapoints:
                            add_to_training = True
                            # make it false if you want to add it to the test set
                        else:
                            add_to_training = True
                            ranking_data.append((features_j, features_k))
                            z[int(coat_data[j, sensitive_feature_id])][int(
                                coat_data[k, sensitive_feature_id])].append(
                                    count)
                            z_assignment.append(type_of_pair)
                            counts[int(coat_data[j, sensitive_feature_id])][
                                int(coat_data[k, sensitive_feature_id])] += 1
                            count += 1
                    if not add_to_training:
                        test_ranking_data.append((features_j, features_k))
                        test_z[int(coat_data[j, sensitive_feature_id])][int(
                            coat_data[k, sensitive_feature_id])].append(
                                test_count)
                        test_counts[int(coat_data[j, sensitive_feature_id])][
                            int(coat_data[k, sensitive_feature_id])] += 1
                        test_count += 1
                        test_z_assignment.append(type_of_pair)
                        # test_z_assignment 1 and 2 represent the pairs
                        # where j and k are in different groups
                        # 0 and 3 represents the pairs where j and k are in the same groups
    z = np.array(z)
    test_z = np.array(test_z)
    z_assignment = np.array(z_assignment)
    test_z_assignment = np.array(test_z_assignment)
    x = np.array(ranking_data)
    test_x = np.array(test_ranking_data)
    counts = np.array(counts)
    test_counts = np.array(test_counts)
    return {
        "x": x,
        "test_x": test_x,
        "z_assignment": z_assignment,
        "test_z_assignment": test_z_assignment,
        "z": z,
        "test_z": test_z,
        "counts": counts,
        "test_counts": test_counts,
        "feature_matrix": feature_matrix,
        "groups": group
    }
