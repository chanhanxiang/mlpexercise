import sqlite3
import pandas as pd

# Local imports
import mlp_preprocess
import mlp_model


if __name__ == "__main__":
    conn = sqlite3.connect(r"../data/fishing.db")
    df = pd.read_sql_query("SELECT * from fishing", conn)
    df = mlp_preprocess.df_preprocess(df)
    df_cat = mlp_preprocess.dfcat(df, df_cat=None)
    df_num = mlp_preprocess.dfnum(df, df_num=None)
    df_num = mlp_preprocess.IQR(df_num)
    df_new = mlp_preprocess.rejoin(df, df_cat, df_num, df_new=None)
    df_new = mlp_preprocess.objmap(df_new)
    df_new = mlp_preprocess.onehotencode_drop(df_new)
    mlp_model.run_models(df_new)
    mlp_model.log_reg(df_new)
    mlp_model.adb_cla(df_new)
    mlp_model.neighbour(df_new)
