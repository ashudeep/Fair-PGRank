import numpy as np
import os

from progressbar import progressbar
import pickle as pkl


class YahooDataReader:
    """
    Takes in a directory and reads the files in it. The files are in the Yahoo dataset format
    feature_id:value, feature_id:value
    """

    def __init__(self, directory):
        self.directory = directory

    def readfile(self, filename, max_candset_size=1000):
        with open(self.directory + filename) as f:
            feats = []
            for i, line in enumerate(f):
                if i == 0:
                    if line.strip() == '':
                        relevant_docs = []
                        # print("The set of relevant docs is empty. Skip these maybe!")
                        return None, None
                    else:
                        relevant_docs = list(map(int, line.strip().split(',')))
                else:
                    feats.append(list(map(float, line.strip().split(','))))
            num_cands = len(feats)
            feats = np.array(feats)
            while True:
                # keep sampling until we get a non-empty relevant doc set
                # repeated sampling on?
                if num_cands > max_candset_size:
                    chosen_few = np.random.choice(num_cands, max_candset_size)
                    feats_ = feats[chosen_few]
                    rel = []
                    for i, d in enumerate(chosen_few):
                        if d in relevant_docs:
                            rel.append(i)
                    # print(chosen_few, relevant_docs)
                else:
                    rel = relevant_docs
                    feats_ = feats
                if len(rel) < 1:
                    continue
                else:
                    return feats_, rel

    def readdir(self):
        feat_list, rel_list = [], []
        for fil in os.listdir(self.directory):
            feats, rel = self.readfile(fil)
            feat_list.append(feats)
            rel_list.append(rel)
        return feat_list, rel_list

    def pickelize_data(self, outpath=None):
        print("Converting the data to pkl files")
        feats_list, rel_list = self.readdir()
        self.data = (feats_list, rel_list)
        if outpath is not None:
            print("Storing the data in {}".format(outpath))
            pkl.dump(self.data, open(outpath, 'wb'))


def reader_from_pickle(input_filename):
    f = open(input_filename, "rb")
    dr = YahooDataReader("")
    dr.data = pkl.load(f)
    return dr
