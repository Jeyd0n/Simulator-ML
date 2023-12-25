def word_count(texts):
    count = {}
    for text in texts:
        for word in text.split():
            count[word] = count.get(word, 0) + 1
    return count