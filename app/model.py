from utils import process_tweet


class NaiveBayesModel:

    def __init__(
        self, logprior, loglikelihood
    ):
        self.logprior = logprior
        self.loglikelihood = loglikelihood

    def predict(self, tweet):
        p = 0

        # add the logprior
        p += self.logprior

        for word in process_tweet(tweet):

            # check if the word exists in the loglikelihood dictionary
            if word in self.loglikelihood:
                # add the log likelihood of that word to the probability
                p += self.loglikelihood[word]

        return p
