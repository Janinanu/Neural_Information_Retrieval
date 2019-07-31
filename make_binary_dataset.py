import pickle
import pandas as pd
import random
from collections import defaultdict
import argparse

def collect_examples(label_type, threshold, limit):
    """
    Creates dictionary with class labels as keys and a list of all corresponding documents as values
    :param label_type: "PRODUCT" or "ISSUE"
    :param threshold: class frequency threshold
    :param limit: to take a subset of all available documents as basis for pairs and triplets
    :return: dictionary of class labels and corresponding documents
    """

    with open('public_data/inputs/dataset.pkl', 'rb') as indata:
        data = pd.read_pickle(indata, compression=None)[:limit]

    with open('public_data/stats/stats_train.pkl', 'rb') as stats:
        stats = pickle.load(stats)
    labels = [label for label, freq in stats["DISTR_" + label_type].items() if freq >= threshold]

    collect_pos = defaultdict(list)
    collect_neg = defaultdict(list)

    for label in labels:
        collect_pos[label] = [(i, row["TEXT"]) for i, row in data.iterrows() if row[label_type] == label]
        labeled_neg = data[data[label_type] != label]
        collect_neg[label] = [(i, row["TEXT"]) for i, row in labeled_neg.sample(n=len(collect_pos[label])).iterrows()]

    return collect_pos, collect_neg


def create_binary_dataset(label_type, threshold, bin_type, limit):

    """
    Creates the pairs and triplets, applies train/validation/test split and stores them
    :param label_type: "PRODUCT" or "ISSUE"
    :param threshold: class frequency threshold
    :param bin_type: "pair" or "triplet"
    :param limit: to take a subset of all available documents as basis for pairs and triplets
    :return:
    """
    if bin_type == "pair":

        collect_pos, collect_neg = collect_examples(limit=limit, label_type=label_type, threshold=threshold)

        dict_for_df = defaultdict(lambda: defaultdict(str))
        curr = 0

        for label, pos_texts in collect_pos.items():

            count_examples = len(pos_texts)
            print("Len df pos before", len(dict_for_df["ID_A"]))

            seen_pairs = []
            i = 0
            while i < count_examples:

                a = random.randrange(0, len(pos_texts))
                b = random.randrange(0, len(pos_texts))
                while {a, b}in seen_pairs:
                    a = random.randrange(0, len(pos_texts))
                    b = random.randrange(0, len(pos_texts))
                seen_pairs.append({a, b})

                id_a, text_a = pos_texts[a]
                id_b, text_b = pos_texts[b]

                dict_for_df["ID_A"][curr] = id_a
                dict_for_df["TEXT_A"][curr] = text_a

                dict_for_df["ID_B"][curr] = id_b
                dict_for_df["TEXT_B"][curr] = text_b

                dict_for_df["BINARY_LABEL"][curr] = 1.0

                i += 1
                curr += 1

            print("Len df pos after", len(dict_for_df["ID_A"]))

            neg_texts = collect_neg[label]
            seen_pairs_pos_neg = []

            i = 0
            while i < count_examples:

                a = random.randrange(0, len(pos_texts))
                b = random.randrange(0, len(neg_texts))
                while {a, b} in seen_pairs_pos_neg:
                    a = random.randrange(0, len(pos_texts))
                    b = random.randrange(0, len(neg_texts))
                seen_pairs_pos_neg.append({a, b})

                id_a, text_a = pos_texts[a]
                id_b, text_b = neg_texts[b]

                dict_for_df["ID_A"][curr] = id_a
                dict_for_df["TEXT_A"][curr] = text_a

                dict_for_df["ID_B"][curr] = id_b
                dict_for_df["TEXT_B"][curr] = text_b

                dict_for_df["BINARY_LABEL"][curr] = -1.0
                i += 1
                curr += 1

            print("Len df pos & neg after", len(dict_for_df["ID_A"]))
            print("#" * 20)

    elif bin_type == "triplet":

        collect_pos, collect_neg = collect_examples(limit=limit, label_type=label_type, threshold=threshold)

        dict_for_df = defaultdict(lambda: defaultdict(str))
        curr = 0

        for label, pos_texts in collect_pos.items():

            count_examples = len(pos_texts)
            neg_texts = collect_neg[label]

            print("Len df before", len(dict_for_df["ID_A"]))

            seen_triplets = []
            i = 0
            while i < count_examples:

                a = random.randrange(0, len(pos_texts))
                b = random.randrange(0, len(pos_texts))
                c = random.randrange(0, len(neg_texts))

                while (a, b, c) in seen_triplets:
                    a = random.randrange(0, len(pos_texts))
                    b = random.randrange(0, len(pos_texts))
                    c = random.randrange(0, len(neg_texts))

                seen_triplets.append((a, b, c))

                id_a, text_a = pos_texts[a]
                id_b, text_b = pos_texts[b]
                id_c, text_c = neg_texts[c]

                dict_for_df["ID_A"][curr] = id_a
                dict_for_df["TEXT_A"][curr] = text_a

                dict_for_df["ID_B"][curr] = id_b
                dict_for_df["TEXT_B"][curr] = text_b

                dict_for_df["ID_C"][curr] = id_c
                dict_for_df["TEXT_C"][curr] = text_c

                i += 1
                curr += 1

            print("Len df after", len(dict_for_df["ID_A"]))
            print("#" * 20)

    binary_dataset = pd.DataFrame(dict_for_df)

    binary_dataset = binary_dataset.sample(frac=1)
    train = binary_dataset[:int(0.8 * len(binary_dataset))]
    valid = binary_dataset[int(0.8 * len(binary_dataset)):int(0.9 * len(binary_dataset))]
    test = binary_dataset[int(0.9 * len(binary_dataset)):]

    with open("public_data/inputs/binary_train_%s_%s_%s.pkl"
              % (label_type, str(threshold), bin_type), 'wb') as out:
        train.to_pickle(out, compression=None)

    with open("public_data/inputs/binary_valid_%s_%s_%s.pkl"
              % (label_type, str(threshold), bin_type), 'wb') as out:
        valid.to_pickle(out, compression=None)

    with open("public_data/inputs/binary_test_%s_%s_%s.pkl"
              % (label_type, str(threshold), bin_type), 'wb') as out:
        test.to_pickle(out, compression=None)

    return binary_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--label_type', type=str, nargs="?", default="PRODUCT", help='"PRODUCT", "ISSUE", "COMB"')
    parser.add_argument(
        '--threshold', type=int, nargs="?", default=8000, help='minimum frequency of label in training set')
    parser.add_argument(
        '--bin_type', type=str, nargs="?", default="pair", help='"pair" / "triplet"')
    parser.add_argument(
        '--limit', type=str, nargs="?", default=15000, help='number of examples as basis for binary')
    args = parser.parse_args()
    binary_dataset = create_binary_dataset(label_type=args.label_type,
                                           threshold=args.threshold, bin_type=args.bin_type, limit=args.limit)
