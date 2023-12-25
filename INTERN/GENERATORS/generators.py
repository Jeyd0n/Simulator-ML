import random


def username_generator(
        n: int,
        first_names=['Sophia', 'Vladimir', 'John'],
        last_names=['Davidson', 'Crossheart', 'Silverhand']
):
    """
    Generate dict of random name + surname for list of names and surname

    Parameters
    ----------
    n : int
        Amount of generations

    first_names : List[str]
        List of names 

    last_names : List[str]
        List of surnames

    Returns
    -------
    dict
        Dict for random name + surname combination


    """
    for idx in range(n):
        yield {
            'id': idx + 1,
            'first_name': random.choice(first_names),
            'last_name': random.choice(last_names)
        }


# Example of use
# custom_first_names = ["Max", "Sophia", "Liam"]
# custom_last_names = ["Miller", "Davis", "Garcia"]
# for user in username_generator(
#     3, 
#     first_names=custom_first_names,
#     last_names=custom_last_names
# ):
#     print(
#         user['id'],
#         user['first_name'],
#         user['last_name']
#     )

# Example of output:
# 1 Max Garcia
# 2 Liam Davis
# 3 Max Miller


def data_generator(n):
    """
    Generate couples (x, y) with x as idx and y as random number in range(0, 100)

    Parameters
    ----------
    n : int
        Num of pairs

    Returns
    -------
    tuple
        couples (x, y) with x as idx and y as random number in range(0, 100)

    
    """
    counter = 0

    for _ in range(n):
        yield (counter, random.randint(0, 100))
        
        counter += 1

# Example of use
# for data in data_generator(3):
#     print(data)

# Example of output:
# 0 49
# 1 27
# 2 88
