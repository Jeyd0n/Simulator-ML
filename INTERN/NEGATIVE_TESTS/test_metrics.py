import metrics


def test_non_int_clicks():
    """
    Test ctr function for type of values of clicks
    """
    try:
        metrics.ctr(1.5, 2)

    except TypeError:
        pass

    else:
        raise AssertionError("Non int clicks not handled")


def test_non_int_views():
    """
    Test ctr function for type of values of views
    """
    try:
        metrics.ctr(2, 3.5)

    except TypeError:
        pass

    else:
        raise AssertionError("Non int views not handled") 


def test_non_positive_clicks():
    """
    Test ctr function for clicks > 0
    """
    try:
        metrics.ctr(2, -3)

    except:
        pass

    else:
        raise AssertionError('Negative value of clicks')



def test_non_positive_views():
    """
    Test ctr function for views > 0
    """
    try:
        metrics.ctr(-2, 3)

    except:
        pass

    else:
        raise AssertionError('Negative value of views')


def test_clicks_greater_than_views():
    """
    Test ctr function for value of clicks less than view 
    """
    try:
        metrics.ctr(5, 3)

    except:
        pass

    else:
        raise AssertionError('Click value greater than views')


def test_zero_views():
    """
    Test ctr function for values of views greater than 0
    """
    try:
        metrics.ctr(5, 0)

    except:
        pass

    else:
        raise AssertionError('Amount of views equal zero')