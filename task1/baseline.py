import pickle
from math import exp, log
from collections import Counter, defaultdict
from pathlib import Path

# Description: First-order Markov baseline model thatc ounts how often each note follows each 
# previous note in the training data, and predicts the most common next note for each previous 
# note during testing. Even though data.py gives us a sequence of x input notes, this baseline 
# only looks at the last note in the input sequence to predict the next most common note by 
# frequency. Returns perplexity and accuracy score.


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "processed_pitch_data.pkl"
SMOOTHING = 1.0 # prevents p(x) = 0 so that log(0) doesn't break perplexity calculation


def load_data(data_file=DATA_FILE):
    with Path(data_file).open("rb") as f:
        return pickle.load(f)


def accuracy(predictions, targets):
    if not targets:
        return 0.0

    correct = sum(pred == target for pred, target in zip(predictions, targets))
    return correct / len(targets)


def perplexity(probabilities):
    if not probabilities:
        return 0.0

    mean_negative_log_likelihood = -sum(log(prob) for prob in probabilities) / len(probabilities)
    return exp(mean_negative_log_likelihood)


def train_markov_baseline(x_train, y_train):
    transitions = defaultdict(Counter)

    for sequence, target in zip(x_train, y_train):
        previous_pitch = sequence[-1]
        transitions[previous_pitch][target] += 1

    return transitions


def most_likely_next_pitch(markov_model):
    return {
        previous_pitch: counts.most_common(1)[0][0]
        for previous_pitch, counts in markov_model.items()
    }


def markov_predictions(markov_model, x_test):
    best_next_pitch = most_likely_next_pitch(markov_model)
    default_pitch = 0
    predictions = []

    for sequence in x_test:
        previous_pitch = sequence[-1]
        predictions.append(best_next_pitch.get(previous_pitch, default_pitch))

    return predictions


def markov_probabilities(markov_model, x_test, y_test, vocab_size):
    probabilities = []

    for sequence, target in zip(x_test, y_test):
        previous_pitch = sequence[-1]
        counts = markov_model.get(previous_pitch)

        if counts is None:
            probabilities.append(1 / vocab_size)
            continue

        total = sum(counts.values()) + (SMOOTHING * vocab_size)
        probabilities.append((counts[target] + SMOOTHING) / total)

    return probabilities


def main():
    data = load_data()
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    vocab_size = data["vocab_size"]

   
    markov_model = train_markov_baseline(x_train, y_train)
    markov_preds = markov_predictions(markov_model, x_test)
    markov_acc = accuracy(markov_preds, y_test)
    markov_probs = markov_probabilities(
        markov_model,
        x_test,
        y_test,
        vocab_size,
    )
    markov_ppl = perplexity(markov_probs)

    print(f"Loaded: {DATA_FILE}")
    print(f"Training examples: {len(x_train)}")
    print(f"Testing examples: {len(x_test)}")
    print(f"Markov baseline accuracy: {markov_acc:.4f}")
    print(f"Markov baseline perplexity: {markov_ppl:.4f}")


if __name__ == "__main__":
    main()
