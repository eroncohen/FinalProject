import random

happy_sentences = ["Ha Ha Ha", "You Have Amazing Smile", "Hee Hee Hee", "So Beautiful Smile"]
sad_sentences = ["Dont Worry Be Happy", "I Will Make You Laugh", "Oh No Be Happy", "Dont Be Sad"]


def random_happy_sentence():
    return random.choice(happy_sentences)


def random_sad_sentence():
    return random.choice(sad_sentences)
