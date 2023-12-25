from typing import List
from functools import reduce


def sales_with_tax(sales: List[float], tax_rate: float, threshold: float = 300) -> List[float]:
    return list(
        map(lambda x: x if x < threshold else x + x * tax_rate, sales)
    )


def sum_sales(sales: List[float], threshold: float = 300) -> float:
    # YOUR CODE HERE
    pass


def average_age(ages: List[int], threshold: int = 30) -> float:
    # YOUR CODE HERE
    pass


def increased_prices(prices: List[float], increase_rate: int = 0.2, threshold: float = 300) -> List[float]:
    increased_prices_list = list(map())