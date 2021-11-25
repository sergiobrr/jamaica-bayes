from train import train_naive_bayes, count_tweets, prepare_datasets
from model import NaiveBayesModel


def get_trained_model(train_data):
    freqs = count_tweets(
        {}, train_data['train_x'], train_data['train_y']
    )
    logprior, loglikelihood = train_naive_bayes(
        freqs, train_data['train_x'], train_data['train_y']
    )
    return NaiveBayesModel(logprior, loglikelihood)


if __name__ == '__main__':
    train_data, test_data = prepare_datasets()
    model = get_trained_model(train_data)
    print(model.predict('Oh my god this is so weird'))
    import pickle
    pickle.dump(model, open('model.pk', 'wb'))
