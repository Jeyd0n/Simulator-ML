import re
from collections import Counter

import utils


def test_word_count():
    batch = [
        'Я вас любил',
        'Любовь еще быть может',
        'В душе моей угасла не совсем',
        'Но пусть она вас больше не тревожит',
        'Я не хочу печалить вас ничем'
    ]
    for text in batch:
        text = re.sub(',.', '', text)
        counter = Counter(text.split())
        wrd_cnt = utils.word_count([text])

        if wrd_cnt:
            for word in text.split():
                assert counter[word] == wrd_cnt[word]
        else:
            raise AssertionError


def test_word_count_tricky():
    batch = [
        'Я вас любил, любовь еще быть может',
        'В душе моей угасла не совсем',
        'Но пусть она вас больше не тревожит. Я не хочу печалить вас ничем'
    ]
    for text in batch:
        text_re = re.sub(',.', '', text)
        counter = Counter(text_re.split())

        assert counter.keys() == utils.word_count([text]).keys() 
