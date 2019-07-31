import datetime
from collections import Counter, defaultdict
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from custom_tokenizer import tokenize_and_clean

class Statistics:
    """
    Computes and stores basic statistics, e.g. the average text length.
    Plots and stores the distribution of class frequencies of the input file.
    """

    def __init__(self, filename):

        self.filename = filename
        with open('public_data/inputs/%s.pkl' %filename, 'rb') as inf:
            self.dataset = pd.read_pickle(inf, compression=None)
        self.stats = defaultdict()

        self.get_num_reports()
        self.get_tokens_vocab()
        self.get_label_stats()

        with open("public_data/stats/stats_%s.pkl" % (filename), 'wb') as out:
            pickle.dump(self.stats, out)

    def get_num_reports(self):

        print(str(datetime.datetime.now()).split('.')[0], "Getting reports counts...")
        self.stats["NUM_REPORTS"] = len(self.dataset)

    def get_tokens_vocab(self):

        print(str(datetime.datetime.now()).split('.')[0], "Extracting tokens and vocab...")
        all_tokens = []
        seq_lens = []
        tokens = self.dataset["TEXT"].apply(lambda x: tokenize_and_clean(x))

        for token_list in tokens:
            all_tokens.extend(token_list)
            seq_lens.append(len(token_list))

        self.stats["TOKENS"] = all_tokens
        self.stats["VOCAB"] = list(set(all_tokens))
        self.stats["TOKEN_FREQS"] = Counter(all_tokens)
        self.stats["AVG_TEXT_LEN"] = sum(seq_lens)/len(seq_lens)


    def get_label_stats(self):
        print(str(datetime.datetime.now()).split('.')[0], "Getting label stats...")

        product_freqs = Counter([l for l in list(self.dataset["PRODUCT"]) if l != ""])
        issue_freqs = Counter([l for l in list(self.dataset["ISSUE"]) if l != ""])
        combi_freqs = Counter(list(self.dataset["COMB"]))

        self.stats["DISTR_PRODUCT"] = product_freqs
        self.stats["DISTR_ISSUE"] = issue_freqs
        self.stats["DISTR_COMB"] = combi_freqs

        self.store_plot(product_freqs,  "distr_product", 30)
        #self.store_plot(issue_freqs, "distr_issue", 30)
        #self.store_plot(combi_freqs, "distr_comb", 30)

    def store_plot(self, freqs, plotname, top):

        common = dict(freqs.most_common(top))

        l = list(common.keys())
        for i, c in enumerate(l):
            if c == "Credit reporting, credit repair services, or other personal consumer reports":
                l[i] = "Credit repair services"
            if c == "Money transfer, virtual currency, or money service":
                l[i] = "Money transfer"
            if c == "Payday loan, title loan, or personal loan":
                l[i] = "Payday loan"
        plt.bar(l, common.values())
        plt.xticks(l, l)
        plt.xticks(fontsize=8, rotation=90)
        plt.savefig("public_data/stats/%s_%s.png" % (plotname, self.filename), bbox_inches = "tight")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filename', type=str, help='train, valid, test')
    args = parser.parse_args()

    stats = Statistics(args.filename)
