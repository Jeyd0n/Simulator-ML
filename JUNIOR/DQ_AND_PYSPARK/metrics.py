"""Metrics."""

from typing import Any, Dict, Union, List
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountNull(Metric):
    """Number of empty values in choosen columns"""

    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = 0

        if self.aggregation == 'any':
            for row in df[self.columns].values:
                row = pd.Series(row)
                if sum(row.isna()) >= 1:
                    k += 1
        elif self.aggregation == 'all':
            for row in df[self.columns].values:
                row = pd.Series(row)
                if len(row) == sum(row.isna()):
                    k += 1

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in choosen columns"""

    columns: List[str]

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = n - len(df.drop_duplicates(subset=self.columns))

        return {"total": n, "count": k, 'delta': k / n}


@dataclass
class CountValue(Metric):
    """Number of values in choosen column"""

    column: str
    value: Union[str, int, float]

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = df[self.column].value_counts()[self.value]

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)

        if self.strict:
            k = len(df[df[self.column] < self.value])
        else:
            k = len(df[df[self.column] <= self.value])

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = 0

        pointer = 0
        if self.strict:
            while pointer < len(df):
                if df[self.column_x][pointer] < df[self.column_y][pointer]:
                    k += 1        
                pointer += 1
        else:
            while pointer < len(df):
                if df[self.column_x][pointer] <= df[self.column_y][pointer]:
                    k += 1        
                pointer += 1

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = 0

        pointer = 0
        if self.strict:
            while pointer < len(df):
                if df[self.column_x][pointer] / df[self.column_y][pointer] < df[self.column_z][pointer]:
                    k += 1        
                pointer += 1
        else:
            while pointer < len(df):
                if df[self.column_x][pointer] / df[self.column_y][pointer] <= df[self.column_z][pointer]:
                    k += 1        
                pointer += 1

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountCB(Metric):
    """Calculate lower/upper bounds for N%-confidence interval"""

    column: str
    conf: float = 0.95

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        lcb = df[self.column].quantile((1 - self.conf) / 2)
        ucb = df[self.column].quantile(self.conf + ((1 - self.conf) / 2))

        return {"lcb": lcb, "ucb": ucb}


@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""

    column: str
    fmt: str = "%Y-%m-%d"

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        a = datetime.now()
        b = datetime.strptime(
            max(df[self.column]
            ), self.fmt)
        lag = (a - b).days

        return {"today": str(a.strftime(self.fmt)), "last_day": str(b.strftime(self.fmt)), "lag": lag}
    

if __name__ == '__main__':
    date = pd.read_csv('JUNIOR/DQ_AND_PYSPARK/ke_daily_sales.csv')

    columns = ['qty', 'price', 'revenue']
    if True == True:
            for row in date[columns].values:
                for value in row:
                    if value == ' None':
                        print(value)

    print(date[columns].values)
    print(CountNull(columns=['qty', 'price', 'revenue']).__call__(df=date))
