

import pandas as pd
from sklearn import preprocessing


class preprocess:

    def __init__(self, data_in, data_out):

        self.data_in = data_in
        self.data_out = data_out

    def process(self):

        df = pd.read_csv(self.data_in, low_memory=False)
        df = df.set_index('Unnamed: 0')

        pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
        x = pt.fit_transform(df)
        df_out = pd.DataFrame(x)

        df_out.columns = df.columns
        df_out.index = df.index
        df_out = pd.DataFrame(df_out)

        df_out.to_csv(self.data_out, sep='\t', encoding='utf-8')
