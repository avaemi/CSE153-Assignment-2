import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

# Description: Exploratory data analysis and visualization of the processed pitch dataset.

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "processed_pitch_data.pkl"
# save histograms in plots/ directory
PLOTS_DIR = BASE_DIR / "plots" 

# load dictionary created by data.py
def load_data(data_file=DATA_FILE):
    with Path(data_file).open("rb") as f:
        return pickle.load(f)

# pitch distribution histogram plot
def save_pitch_histogram(pitch_counts, output_file):
    pitches = sorted(pitch_counts)
    counts = [pitch_counts[pitch] for pitch in pitches]

    plt.figure(figsize=(12, 5))
    plt.bar(pitches, counts, width=0.9)
    plt.xlabel("MIDI pitch")
    plt.ylabel("Count")
    plt.title("Pitch distribution")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# histogram plot trimmed 95th percentile 
# so outliers do not crush view
def percentile(values, percent):
    sorted_values = sorted(values)
    index = round((percent / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


# song length distribution histogram
def save_sequence_length_histogram(sequence_lengths, output_file, trim_percent=95):
    cutoff = percentile(sequence_lengths, trim_percent)
    trimmed_lengths = [length for length in sequence_lengths if length <= cutoff]

    plt.figure(figsize=(10, 5))
    plt.hist(trimmed_lengths, bins=30, edgecolor="black")
    plt.xlabel("Notes per song")
    plt.ylabel("Number of songs")
    plt.title(f"Song length distribution (up to {trim_percent}th percentile)")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# look at each training/testing example, specifically last input note -> target next note
# count how many times each interval (target - last input) occurs
# tells us if the pitch tends to make small or big jumps between notes
def get_interval_counts(data):
    int_to_pitch = data["int_to_pitch"]
    x_all = data["x_train"] + data["x_test"]
    y_all = data["y_train"] + data["y_test"]
    interval_counts = Counter()

    for sequence, target in zip(x_all, y_all):
        previous_pitch = int_to_pitch[sequence[-1]]
        next_pitch = int_to_pitch[target]
        interval_counts[next_pitch - previous_pitch] += 1

    return interval_counts

def save_interval_histogram(interval_counts, output_file):
    intervals = sorted(interval_counts)
    counts = [interval_counts[interval] for interval in intervals]

    plt.figure(figsize=(12, 5))
    plt.bar(intervals, counts, width=0.9)
    plt.xlabel("Pitch interval to next note")
    plt.ylabel("Count")
    plt.title("Next-note interval distribution")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def get_transition_counts(data):
    int_to_pitch = data["int_to_pitch"]
    x_all = data["x_train"] + data["x_test"]
    y_all = data["y_train"] + data["y_test"]
    transition_counts = Counter()

    for sequence, target in zip(x_all, y_all):
        previous_pitch = int_to_pitch[sequence[-1]]
        next_pitch = int_to_pitch[target]
        transition_counts[(previous_pitch, next_pitch)] += 1

    return transition_counts


def save_transition_heatmap(transition_counts, pitch_counts, output_file, top_n=15):
    top_pitches = sorted(pitch for pitch, _ in pitch_counts.most_common(top_n))
    pitch_to_pos = {pitch: i for i, pitch in enumerate(top_pitches)}
    matrix = [[0 for _ in top_pitches] for _ in top_pitches]

    for (previous_pitch, next_pitch), count in transition_counts.items():
        if previous_pitch in pitch_to_pos and next_pitch in pitch_to_pos:
            row = pitch_to_pos[previous_pitch]
            col = pitch_to_pos[next_pitch]
            matrix[row][col] = count

    _, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, aspect="auto")
    plt.colorbar(image, ax=ax, label="Transition count")
    ax.set_xticks(range(len(top_pitches)))
    ax.set_yticks(range(len(top_pitches)))
    ax.set_xticklabels(top_pitches, rotation=45, ha="right")
    ax.set_yticklabels(top_pitches)
    ax.set_xlabel("Next MIDI pitch")
    ax.set_ylabel("Previous MIDI pitch")
    ax.set_title(f"Pitch transition heatmap - top {top_n} pitches")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# how many times each MIDI pitch appears in the dataset
def get_pitch_counts(data):
# if .pkl already has pitch counts, use them. 
    if "pitch_counts" in data:
        return Counter(data["pitch_counts"])

# Otherwise, count from y_train and y_test
    int_to_pitch = data["int_to_pitch"]
    encoded_targets = data["y_train"] + data["y_test"]
    return Counter(int_to_pitch[idx] for idx in encoded_targets)


def main():
    data = load_data()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    pitch_counts = get_pitch_counts(data)
    pitch_plot = PLOTS_DIR / "pitch_distribution.png"
    save_pitch_histogram(pitch_counts, pitch_plot)

    interval_counts = get_interval_counts(data)
    interval_plot = PLOTS_DIR / "next_note_intervals.png"
    save_interval_histogram(interval_counts, interval_plot)

    transition_counts = get_transition_counts(data)
    transition_plot = PLOTS_DIR / "pitch_transition_heatmap.png"
    save_transition_heatmap(transition_counts, pitch_counts, transition_plot)

    sequence_lengths = data.get("sequence_lengths", [])
    length_plot = PLOTS_DIR / "song_lengths.png"
    if sequence_lengths:
        save_sequence_length_histogram(sequence_lengths, length_plot)
    total_notes = sum(pitch_counts.values())
    most_common = pitch_counts.most_common(10)

    # some statistics about the dataset
    print(f"Loaded: {DATA_FILE}")
    print(f"MIDI files used: {data['midi_files_used']}")
    print(f"Usable songs: {data['usable_songs']}")
    print(f"Skipped files: {data['skipped_files']}")
    print(f"Vocabulary size: {data['vocab_size']}")
    print(f"Total notes counted: {total_notes}")
    print(f"Training examples: {len(data['x_train'])}")
    print(f"Testing examples: {len(data['x_test'])}")
    print(f"Saved pitch histogram: {pitch_plot}")
    print(f"Saved next-note interval histogram: {interval_plot}")
    print(f"Saved pitch transition heatmap: {transition_plot}")

    if sequence_lengths:
        print(f"Shortest song: {min(sequence_lengths)} notes")
        print(f"Longest song: {max(sequence_lengths)} notes")
        print(f"Average song length: {sum(sequence_lengths) / len(sequence_lengths):.1f} notes")
        print(f"95th percentile song length: {percentile(sequence_lengths, 95)} notes")
        print(f"Saved song length histogram: {length_plot}")
    else:
        print("Song lengths not found. Re-run data.py to include song length metadata.")

    print("Most common pitches:")
    for pitch, count in most_common:
        print(f"  MIDI pitch {pitch}: {count}")

    print("Most common next-note intervals:")
    for interval, count in interval_counts.most_common(10):
        print(f"  {interval:+d} semitones: {count}")


if __name__ == "__main__":
    main()
