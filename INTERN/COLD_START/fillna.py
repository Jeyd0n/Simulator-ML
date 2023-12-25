import pandas as pd
import numpy as np


def fillna_with_mean(
    df: pd.DataFrame, target: str, group: str
) -> pd.DataFrame:
    """
    Fill NaN values for target column by mean values of target

    Parameters
    ----------
    target : str
        Target column to select 

    group : str
        Column to group by

    Returns
    -------
    pd.DataFrame
        Result dataframe with filled NaN


    """
    df = df.copy()
    df[target].fillna(
        value=np.floor(df.groupby(group)[target].transform('mean')),
        inplace=True
    )
 
    return df
