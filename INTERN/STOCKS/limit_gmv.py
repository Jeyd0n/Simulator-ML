import pandas as pd 


def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    limit_gmv = df['price'] * df['stock']
    
    
    return df