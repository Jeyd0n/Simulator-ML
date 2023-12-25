import re
from typing import List


def valid_emails(strings: List[str]) -> List[str]:
    """
    Take list of potential emails and returns only valid ones
    
    Parameters
    ----------
    strings : List[str]
        List of emails to check for valid value

    Returns
    -------
    List[str]
        List of emails with valid value


    """
    valid_email_regex = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}$")

    emails = [email for email in strings if valid_email_regex.match(email)]

    return emails
