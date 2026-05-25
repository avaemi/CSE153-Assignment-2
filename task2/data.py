# Task 2: Symbolic Conditioned Generation
# data.py - Data loading, preprocessing, EDA, and baseline for JSB Chorales
# Dataset: JSB Chorales 16th note resolution (Boulanger-Lewandowski et al. 2012)
# Task: given a melody (soprano), predict the accompanying chords (alto/tenor/bass)

import os
import pickle
import numpy as np
import collections
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = './task2/JSB-Chorales-dataset-master'


def load_data():
    """Load the JSB Chorales dataset from pickle and print basic split info."""
    print("Loading JSB Chorales dataset...")

    # 16th note resolution keeps passing tones and ornaments intact 
    # resolutions like 8th or quarter notes lose rhythmic detail that matters for harmony
    pkl_path = os.path.join(DATA_PATH, 'jsb-chorales-16th.pkl')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    for split in ['train', 'valid', 'test']:
        print(f"  {split}: {len(data[split])} chorales")

    first = data['train'][0]
    print(f"  First training chorale length: {len(first)} timesteps, {len(first[0])} voices")

    return data


def parse_chorales(data_dict):
    """Extract (melody_note, chord) pairs from each chorale across all splits."""
    print("Parsing chorales...")

    parsed = {}
    for split in ['train', 'valid', 'test']:
        split_sequences = []
        for chorale in data_dict[split]:
            sequence = []
            for timestep in chorale:
                if len(timestep) < 4:
                    continue
                melody_note = int(timestep[0])

                # 0 means the voice is resting. we only want sounding notes in the chord
                chord = frozenset(int(v) for v in timestep[1:] if v != 0)

                # frozenset because chord identity is order-independent, just like in music
                # (60, 64, 67) and (67, 60, 64) are the same C major chord
                sequence.append((melody_note, chord))
            split_sequences.append(sequence)
        parsed[split] = split_sequences

    print("  First 5 timesteps of first training chorale:")
    for i, (note, chord) in enumerate(parsed['train'][0][:5]):
        print(f"    t={i}  melody={note}  chord={set(chord)}")

    return parsed


def build_vocabularies(parsed_data):
    """Build note and chord vocabularies from the training set only, with PAD and UNK tokens."""
    print("Building vocabularies...")

    # Only scan train; using valid/test to build the vocab would be data leakage.
    # UNK handles any note or chord in valid/test that the model has never seen before.
    all_notes = set()
    all_chords = set()
    for seq in parsed_data['train']:
        for melody_note, chord in seq:
            all_notes.add(melody_note)
            all_chords.add(chord)

    # PAD=0 reserved for sequence padding, UNK=1 for out-of-vocabulary items
    note_to_idx = {'PAD': 0, 'UNK': 1}
    for note in sorted(all_notes):
        note_to_idx[note] = len(note_to_idx)
    idx_to_note = {idx: note for note, idx in note_to_idx.items()}

    chord_to_idx = {'PAD': 0, 'UNK': 1}
    for chord in sorted(all_chords):
        chord_to_idx[chord] = len(chord_to_idx)
    idx_to_chord = {idx: chord for chord, idx in chord_to_idx.items()}

    print(f"  Note vocabulary size:  {len(note_to_idx)}  (includes PAD and UNK)")
    print(f"  Chord vocabulary size: {len(chord_to_idx)}  (includes PAD and UNK)")

    return note_to_idx, idx_to_note, chord_to_idx, idx_to_chord


def encode_sequences(parsed_data, note_to_idx, chord_to_idx):
    """Convert (melody_note, chord) pairs to (note_idx, chord_idx) integer index pairs."""
    print("Encoding sequences...")

    # Map each note and chord to its vocab index; fall back to UNK=1 for anything
    # not seen in training; this handles valid/test OOV without crashing
    encoded = {}
    for split in ['train', 'valid', 'test']:
        split_sequences = []
        for seq in parsed_data[split]:
            encoded_seq = []
            for melody_note, chord in seq:
                note_idx = note_to_idx.get(melody_note, note_to_idx['UNK'])
                chord_idx = chord_to_idx.get(chord, chord_to_idx['UNK'])
                encoded_seq.append((note_idx, chord_idx))
            split_sequences.append(encoded_seq)
        encoded[split] = split_sequences

    total = sum(len(seq) for seq in encoded['train'])
    print(f"  Encoded {len(encoded['train'])} train chorales ({total} timesteps total)")

    return encoded


def run_eda(parsed_data):
    """Generate and save four EDA plots for the JSB Chorales training set."""
    print("Running EDA...")

    os.makedirs('task2/eda_plots', exist_ok=True)

    train = parsed_data['train']

    all_notes = [note for seq in train for note, _ in seq]
    all_chords = [chord for seq in train for _, chord in seq]
    lengths = [len(seq) for seq in train]

    note_counts = collections.Counter(all_notes)
    chord_counts = collections.Counter(all_chords)

    most_common_note = note_counts.most_common(1)[0][0]
    most_common_chord = chord_counts.most_common(1)[0][0]
    print(f"  Most common melody note:  MIDI {most_common_note}")
    print(f"  Most common chord:        {set(most_common_chord)}")

    total_timesteps = len(all_chords)
    top10 = chord_counts.most_common(10)
    df = pd.DataFrame({
        'chord': [str(set(c)) for c, _ in top10],
        'count': [cnt for _, cnt in top10],
        'pct': [f'{cnt / total_timesteps * 100:.1f}%' for _, cnt in top10],
    })
    print(df.to_string(index=False))

    # Plot 1: melody note distribution; tells us which pitches soprano favors most
    top_notes = note_counts.most_common(20)
    notes, note_freqs = zip(*top_notes)
    _, ax = plt.subplots(figsize=(10, 4))
    ax.bar([str(n) for n in notes], note_freqs)
    ax.set_xlabel('MIDI Note')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 20 Melody Note Frequencies (Train)')
    plt.tight_layout()
    plt.savefig('task2/eda_plots/melody_note_freq.png')
    plt.close()
    # %% [markdown]
    # MIDI 69 (A4, the A above middle C) is by far the most common soprano note.
    # The top 5 notes are all clustered together in pitch, which makes sense since
    # Bach soprano melodies move mostly by small steps within a narrow range.
    # The notes at the far right (60, 61, 63) barely appear, so the model will have
    # very little training signal for those pitches. This is exactly why we need UNK.

    # Plot 2: chord distribution; shows how many distinct chords appear and which dominate
    top_chords = chord_counts.most_common(20)
    chord_labels = [str(set(c)) for c, _ in top_chords]
    chord_freqs = [f for _, f in top_chords]
    _, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(chord_labels)), chord_freqs)
    ax.set_xticks(range(len(chord_labels)))
    ax.set_xticklabels(chord_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Frequency')
    ax.set_title('Top 20 Chord Frequencies (Train)')
    plt.tight_layout()
    plt.savefig('task2/eda_plots/chord_freq.png')
    plt.close()
    # %% [markdown]
    # The top 3 chords appear around 1000-1130 times each, but after that the
    # distribution flattens out quickly with the remaining chords all hovering
    # between 400-850 occurrences. No single chord dominates, which means a global
    # majority baseline would only be right about 2% of the time. This is why our
    # baseline predicts the most common chord per melody note rather than one chord
    # for everything.

    # Plot 3: chorale length histogram; tells us how long sequences are so we know
    # how much padding/truncation we will need when batching
    _, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=30)
    ax.set_xlabel('Number of Timesteps')
    ax.set_ylabel('Count')
    ax.set_title('Chorale Length Distribution (Train)')
    plt.tight_layout()
    plt.savefig('task2/eda_plots/chorale_lengths.png')
    plt.close()
    # %% [markdown]
    # Chorale lengths range from about 100 to 525 timesteps with two clear peaks
    # around 190 and 255. Most chorales fall between 150 and 300 timesteps, but
    # the long tail out to 525 means sequences vary quite a bit in length. This is
    # why we need to pad shorter sequences and truncate longer

    # Plot 4: average melody note by position; reveals if there are structural
    # melodic trends (e.g. chorales tend to end on lower notes)
    median_len = int(np.median(lengths))
    truncated = []
    for seq in train:
        notes_only = [note for note, _ in seq]
        if len(notes_only) >= median_len:
            truncated.append(notes_only[:median_len])
        else:
            padded = notes_only + [notes_only[-1]] * (median_len - len(notes_only))
            truncated.append(padded)
    avg_by_pos = np.mean(truncated, axis=0)
    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(avg_by_pos)
    ax.set_xlabel('Timestep Position')
    ax.set_ylabel('Average MIDI Note')
    ax.set_title(f'Average Melody Note by Position (truncated/padded to median={median_len})')
    plt.tight_layout()
    plt.savefig('task2/eda_plots/avg_melody_by_position.png')
    plt.close()
    # %% [markdown]
    # The melody starts low around 68.9, rises through the first phrase, then follows
    # a wave pattern of rises and falls that reflects Bach's phrase structure. The
    # overall trend is downward, ending around 68.4 at the final timestep. This makes
    # musical sense since Bach typically resolves phrases on the tonic which sits in
    # the lower part of the soprano range. The repeating wave shape shows that phrase
    # boundaries are a real structural feature the LSTM can learn to track over time.

    # Plot 5: note transition heatmap. if certain transitions are far more likely than
    # others, the model needs memory of the previous note to predict well, which is exactly
    # what an LSTM provides over a bag-of-words or feedforward approach
    top15_notes = [note for note, _ in note_counts.most_common(15)]
    note_set = set(top15_notes)
    transition_counts = collections.Counter()
    for seq in train:
        notes_only = [note for note, _ in seq]
        for curr, nxt in zip(notes_only, notes_only[1:]):
            if curr in note_set and nxt in note_set:
                transition_counts[(curr, nxt)] += 1
    sorted_notes = sorted(top15_notes)
    note_pos = {note: i for i, note in enumerate(sorted_notes)}
    matrix = np.zeros((len(sorted_notes), len(sorted_notes)))
    for (curr, nxt), cnt in transition_counts.items():
        matrix[note_pos[curr], note_pos[nxt]] = cnt
    _, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, aspect='auto')
    plt.colorbar(im, ax=ax, label='Transition count')
    ax.set_xticks(range(len(sorted_notes)))
    ax.set_yticks(range(len(sorted_notes)))
    ax.set_xticklabels(sorted_notes, rotation=45, ha='right')
    ax.set_yticklabels(sorted_notes)
    ax.set_xlabel('Next note (MIDI)')
    ax.set_ylabel('Current note (MIDI)')
    ax.set_title('Melody Note Transition Heatmap - Top 15 Notes (Train)')
    plt.tight_layout()
    plt.savefig('task2/eda_plots/note_transitions.png')
    plt.close()
    # %% [markdown]
    # The heatmap is almost entirely dark except along and just next to the diagonal,
    # which means notes almost always move to themselves or to an adjacent pitch.
    # MIDI 69 staying on 69 is the brightest square (over 7000 times), confirming
    # that sustained notes are the single most common transition in Bach. Everything
    # off the diagonal is near zero, meaning large melodic jumps are very rare.
    # A feedforward model that ignores the previous note would miss all of this
    # structure completely, which is the core motivation for using an LSTM.

    # Plot 6: OOV chord analysis; shows that valid/test contain chords the model has
    # never seen during training, which is exactly why we need UNK instead of crashing
    train_chord_set = set(chord for seq in train for _, chord in seq)
    oov_stats = {}
    for split in ['valid', 'test']:
        split_seqs = parsed_data[split]
        unique = set(chord for seq in split_seqs for _, chord in seq)
        unseen = unique - train_chord_set
        oov_stats[split] = {'unique': len(unique), 'unseen': len(unseen)}
        pct = len(unseen) / len(unique) * 100 if unique else 0
        print(f"  {split}: {len(unique)} unique chords, {len(unseen)} unseen ({pct:.1f}%)")

    oov_splits = ['valid', 'test']
    oov_pcts = [
        oov_stats[s]['unseen'] / oov_stats[s]['unique'] * 100 if oov_stats[s]['unique'] else 0
        for s in oov_splits
    ]
    _, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(oov_splits, oov_pcts)
    for bar, pct in zip(bars, oov_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f'{pct:.1f}%', ha='center', va='center')
    ax.set_ylabel('% of unique chords unseen in training')
    ax.set_title('OOV Rate by Split')
    plt.tight_layout()
    plt.savefig('task2/eda_plots/oov_chords.png')
    plt.close()
    # %% [markdown]
    # About 14% of unique chords in validation and 16% in test never appeared in
    # training. This is not a small number, it means roughly 1 in 6 distinct chords
    # the model sees at test time is completely new. This confirms that UNK is not
    # just a safety net but a real necessity. The model cannot memorize its way
    # through this dataset and has to generalize to unseen chord combinations.


def evaluate_baseline(encoded_data, idx_to_note, idx_to_chord):
    """For each melody note, predict its most common training chord; evaluate on test set."""
    print("Evaluating baseline...")

    # Per-note majority is stronger than global majority because melody notes are not
    # independent of harmony. Bach consistently harmonizes the same pitch with similar
    # chords. If this baseline is high, the melody alone carries most of the harmonic
    # information and a model needs to do more than just memorize note-chord co-occurrence.
    train_pairs = collections.defaultdict(list)
    for seq in encoded_data['train']:
        for note_idx, chord_idx in seq:
            train_pairs[note_idx].append(chord_idx)

    overall_chord_counts = collections.Counter(
        chord_idx for chords in train_pairs.values() for chord_idx in chords
    )
    most_common_chord_idx = overall_chord_counts.most_common(1)[0][0]

    note_to_best_chord = {}
    note_support = {}
    for note_idx, chords in train_pairs.items():
        note_to_best_chord[note_idx] = collections.Counter(chords).most_common(1)[0][0]
        note_support[note_idx] = len(chords)

    correct = 0
    total = 0
    for seq in encoded_data['test']:
        for note_idx, chord_idx in seq:
            pred = note_to_best_chord.get(note_idx, most_common_chord_idx)
            if pred == chord_idx:
                correct += 1
            total += 1

    baseline_accuracy = correct / total

    top10_notes = sorted(note_support, key=note_support.get, reverse=True)[:10]
    df = pd.DataFrame({
        'melody_note': [idx_to_note.get(n, n) for n in top10_notes],
        'most_common_chord': [str(set(idx_to_chord.get(note_to_best_chord[n], {}))) for n in top10_notes],
        'support': [note_support[n] for n in top10_notes],
    })
    print(df.to_string(index=False))
    print(f"  Per-note majority baseline accuracy on test set: {baseline_accuracy:.4f} ({correct}/{total} timesteps)")

    return baseline_accuracy, most_common_chord_idx


def save_processed_data(encoded_data, note_to_idx, idx_to_note, chord_to_idx,
                        idx_to_chord, baseline_accuracy, most_common_chord_idx):
    """Save all preprocessed data and vocabularies to a single pickle file."""
    print("Saving processed data...")

    # model.py needs encoded_data for sequences, note/chord vocabs for embedding sizes,
    # and most_common_chord_idx to initialize or compare against the baseline
    payload = {
        'encoded_data': encoded_data,
        'note_to_idx': note_to_idx,
        'idx_to_note': idx_to_note,
        'chord_to_idx': chord_to_idx,
        'idx_to_chord': idx_to_chord,
        'baseline_accuracy': baseline_accuracy,
        'most_common_chord_idx': most_common_chord_idx,
        'vocab_sizes': {
            'note': len(note_to_idx),
            'chord': len(chord_to_idx),
        },
    }

    out_path = 'task2/processed_data.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved {out_path}  ({size_kb:.1f} KB)")


def load_processed_data():
    """Load and return the preprocessed data dict from processed_data.pkl."""
    with open('task2/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print("Loaded processed data from processed_data.pkl")
    return data


if __name__ == '__main__':
    raw = load_data()                                                                                    # step 1: load raw pickle
    parsed = parse_chorales(raw)                                                                         # step 2: extract (note, chord) pairs
    note_to_idx, idx_to_note, chord_to_idx, idx_to_chord = build_vocabularies(parsed)                   # step 3: build train-only vocabs
    encoded = encode_sequences(parsed, note_to_idx, chord_to_idx)                                       # step 4: convert to index sequences
    run_eda(parsed)                                                                                      # step 5: save EDA plots
    baseline_accuracy, most_common_chord_idx = evaluate_baseline(encoded, idx_to_note, idx_to_chord)    # step 6: per-note majority baseline
    save_processed_data(encoded, note_to_idx, idx_to_note, chord_to_idx, idx_to_chord, baseline_accuracy, most_common_chord_idx)  # step 7: save to pkl

