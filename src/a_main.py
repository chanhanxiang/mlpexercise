import sqlite3
import pandas as pd

# Local imports
import a_preprocess
import a_model


if __name__ == "__main__":
    conn = sqlite3.connect(r"../data/fishing.db")
    df = pd.read_sql_query("SELECT * from fishing", conn)
    df = a_preprocess.df_preprocess(df)
    df_cat = a_preprocess.dfcat(df, df_cat=None)
    df_num = a_preprocess.dfnum(df, df_num=None)
    df_num = a_preprocess.IQR(df_num)
    df_new = a_preprocess.rejoin(df, df_cat, df_num, df_new=None)
    df_new = a_preprocess.objmap(df_new)
    df_new = a_preprocess.onehotencode_drop(df_new)
    a_model.run_models(df_new)
