import metrics


def test_profit() -> None:
    """
    Test correctness of profit function output 

    Parameters
    ----------
    None

    Returns
    -------
    None


    """
    assert metrics.profit([1, 2, 3], [1, 1, 1]) == 3


def test_margin() -> None:
    """
    Test correctness of profit function output 

    Parameters
    ----------
    None

    Returns
    -------
    None

    
    """
    assert metrics.margin([1, 2, 3], [1, 1, 1]) == 0.5


def test_markup() -> None:
    """
    Test correctness of profit function output 

    Parameters
    ----------
    None

    Returns
    -------
    None

    
    """
    assert metrics.markup([1, 2, 3], [1, 1, 1]) == 1
