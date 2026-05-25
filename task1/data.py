import pickle
import random
from collections import Counter
from pathlib import Path
import pretty_midi

#TODO: follow the instructions in disclaimer.md to install the dataset,
# then extract to GIANT_MIDI-Piano-master/midis and then run. 
# TODO: Feel free to make edits as you see fit for the LSTM model!

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MIDI_DIR = BASE_DIR / "GiantMIDI-Piano-master" / "midis"
DEFAULT_OUTPUT_FILE = BASE_DIR / "processed_pitch_data.pkl"


def get_midi_files(midi_dir, max_files):
    midi_files = sorted(Path(midi_dir).rglob("*.mid"))

    if max_files is not None:
        midi_files = midi_files[:max_files]

    return midi_files

# returns pitch sequence for one MIDI file stored in notes[] 
# as (start_time, pitch) pairs
def extract_pitch_sequence(midi_file):
    midi_data = pretty_midi.PrettyMIDI(str(midi_file))
    notes = []

    # ignore drum tracks as they don't have meaningful pitch information
    for instrument in midi_data.instruments:
        if instrument.is_drum:
            continue

    # store notes as (start_time, pitch) pairs, then sort them by time
        for note in instrument.notes:
            notes.append((note.start, note.pitch))

    notes.sort()
    # throw away start times, only keeps pitches in the order played
    return [pitch for _, pitch in notes]

# cleans up MIDI files by extracting pitch sequences 
# and filtering out unreadable or too-short files, 
# return a list of pitch sequences and # of skipped files
def load_sequences(midi_files, min_notes):
    sequences = []
    skipped = 0

    for midi_file in midi_files:
        try:
            pitches = extract_pitch_sequence(midi_file)
        except Exception as error:
            print(f"Could not read {midi_file.name}: {error}")
            skipped += 1
            continue

        if len(pitches) < min_notes:
            skipped += 1
            continue

        sequences.append(pitches)

    return sequences, skipped


def build_vocabulary(sequences):
    # create a sorted set of unique pitches across all sequences
    all_pitches = sorted(set(pitch 
                                for seq in sequences 
                                        for pitch in seq))
    # map each pitch to a unique integer index
    pitch_to_int = {pitch: i for i, pitch in enumerate(all_pitches)}
    # also create the reverse mapping from integer index back to pitch
    int_to_pitch = {i: pitch for pitch, i in pitch_to_int.items()}
    return pitch_to_int, int_to_pitch


def make_training_examples(sequences, pitch_to_int, sequence_length):
    x = [] # store model inputs
    y = [] # store target outputs 

    for seq_list in sequences:
        # convert the pitch sequence to a sequence of integer indexes
        # ex: [60, 62, 64] -> [0, 1, 2] if those are the only pitches in dict.
        encoded_seq_list = [pitch_to_int[pitch] 
                        for pitch in seq_list]

        # create a list of input sequences and a list of 
        # corresponding target outputs using sliding window 
        for i in range(len(encoded_seq_list) - sequence_length):
            x.append(encoded_seq_list[i : i + sequence_length])
            y.append(encoded_seq_list[i + sequence_length])

    return x, y


def split_data(x, y, test_size, seed):
    # reorganize x and y lists into pairs, 
    # shuffle them together into one list
    pairs = list(zip(x, y))
    random.Random(seed).shuffle(pairs)

    # ex: 1 - (test_size=0.2) = 0.8, so 
    # 80% of data goes to training and 20% to testing
    split_index = int(len(pairs) * (1 - test_size))
    train_pairs = pairs[:split_index]
    test_pairs = pairs[split_index:]

    # separate pairs into x and y lists again 
    # but now for training and testing separately
    x_train = [pair[0] for pair in train_pairs]
    y_train = [pair[1] for pair in train_pairs]
    x_test = [pair[0] for pair in test_pairs]
    y_test = [pair[1] for pair in test_pairs]

    return x_train, y_train, x_test, y_test


#save processed data into a pickle file for later model training/eval.
def save_data(output_file, data):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("wb") as f:
        pickle.dump(data, f)


def preprocess_data(
    #TODO: modify these values to customize data processing
    midi_dir=DEFAULT_MIDI_DIR,
    output_file=DEFAULT_OUTPUT_FILE,
    max_files=500,
    sequence_length=50,
    min_notes=50,
    test_size=0.2,
    seed=42,
    save=True,
):
    midi_files = get_midi_files(midi_dir, max_files)
    sequences, skipped = load_sequences(midi_files, min_notes)

    if len(sequences) == 0:
        raise ValueError("No usable MIDI files were found.")

    pitch_to_int, int_to_pitch = build_vocabulary(sequences)
    x, y = make_training_examples(sequences, pitch_to_int, sequence_length)

    if len(x) == 0:
        raise ValueError("No training examples were made. Try a smaller sequence length.")

    x_train, y_train, x_test, y_test = split_data(
        x=x,
        y=y,
        test_size=test_size,
        seed=seed,
    )

    # compile all the processed data and metadata into a dictionary
    processed_data = {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "pitch_to_int": pitch_to_int,
        "int_to_pitch": int_to_pitch,
        "vocab_size": len(pitch_to_int),
        "sequence_length": sequence_length,
        "midi_files_used": len(midi_files),
        "usable_songs": len(sequences),
        "skipped_files": skipped,
        "sequence_lengths": [len(seq) for seq in sequences],
        "pitch_counts": dict(Counter(pitch for seq in sequences for pitch in seq)),
    }

    if save:
        save_data(output_file, processed_data)

    return processed_data


def main():
    processed_data = preprocess_data()

    print(f"MIDI files used: {processed_data['midi_files_used']}")
    print(f"Usable songs: {processed_data['usable_songs']}")
    print(f"Skipped files: {processed_data['skipped_files']}")
    print(f"Vocabulary size: {processed_data['vocab_size']}")
    print(f"Training examples: {len(processed_data['x_train'])}")
    print(f"Testing examples: {len(processed_data['x_test'])}")
    print(f"Saved to: {DEFAULT_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
