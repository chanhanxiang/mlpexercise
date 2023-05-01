import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import pingouin as pg

class visual(object):
    def __init__(self, df_num):
        self.df_num = df_num

    def out_vis(df_num):
        '''Box-plot visualisation of outliers'''
        fig, axs = plt.subplots(ncols=5, nrows=3, figsize=(12, 16))
        index = 0
        axs = axs.flatten()
        for k,v in df_num.items():
            sns.boxplot(y=k, data=df_num, ax=axs[index], orient="v")
            index += 1
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

    def out_pc(df_num):
        '''Statistical calculation of outliers per variable'''
        for column in df_num.columns:
            median = df_num[column].quantile()
            iqr_1_5 = (df_num[column].quantile(q = 0.75) - df_num[column].quantile(q = 0.25)) * 1.5
            outliers = df_num[(df_num[column]< median - iqr_1_5) | (df_num[column] > median + iqr_1_5)][column].count()
            outliers_pct = round(outliers / df_num[column].count() * 100, 1)
            print("'{}' = {} ({}%) outliers".format(column, outliers, outliers_pct))

    def corr(df_num):
        print("The numeric 10 most correlated pairs, Spearman method:")
        spearman_rank = pg.pairwise_corr(df_num, method='spearman').loc[:,['X','Y','r']]
        pos = spearman_rank.sort_values(kind="quicksort", by=['r'], ascending=False).iloc[:5,:]
        neg = spearman_rank.sort_values(kind="quicksort", by=['r'], ascending=False).iloc[-5:,:]
        con = pd.concat([pos, neg], axis=0)
        display(con.reset_index(drop=True))

        mask = np.triu(df_num.corr(method='spearman'), 1)
        plt.figure(figsize=(19, 9))
        sns.heatmap(df_num.corr(method='spearman'), annot=True, 
                    vmax=1, vmin = -1, square=True, cmap='BrBG', mask=mask);

    