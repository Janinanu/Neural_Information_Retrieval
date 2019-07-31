import pandas as pd
import codecs


class Data:
    """
    Adjusts format of the downloaded dataset and turns it into dataframe,
    stores train/validation/test splits
    """
    def __init__(self):
        self.dataset = self.read_adjust_csv("public_data/inputs/consumer_complaints.csv")
        self.split_data()

    def read_adjust_csv(self, csv_path):

        with codecs.open(csv_path, 'r', encoding='utf-8', errors='ignore') as file:
            df = pd.read_csv(file)
            df = df[pd.notnull(df['Consumer complaint narrative'])]
            df["Complaint ID"] = df["Complaint ID"].apply(lambda x: str(x).split(".")[0])
            df.rename(columns={"Consumer complaint narrative": "TEXT", "Product": "PRODUCT", \
                               "Issue": "ISSUE", "Complaint ID": "COMPLAINT_ID"}, inplace=True)
            col = ["TEXT", "PRODUCT", "ISSUE", "COMPLAINT_ID"]
            df = df[col]
            df["COMB"] = df[["PRODUCT", "ISSUE"]].apply(lambda x: " ".join(x), axis=1)
            df.drop_duplicates(inplace=True)
            df.fillna("", inplace=True)

            self.store_pickle(df, 'dataset')

        return df

    def split_data(self):

        self.dataset = self.dataset.sample(frac=1)
        train = self.dataset[:int(0.8 * len(self.dataset))]
        valid = self.dataset[int(0.8 * len(self.dataset)):int(0.9 * len(self.dataset))]
        test = self.dataset[int(0.9 * len(self.dataset)):]

        self.store_pickle(train, 'train')
        self.store_pickle(valid, 'valid')
        self.store_pickle(test, 'test')

    def store_pickle(self, df, filename):

        with open("public_data/inputs/%s.pkl" %filename, 'wb') as out:
            df.to_pickle(out, compression=None, protocol=4)

if __name__ == "__main__":
    data = Data()


