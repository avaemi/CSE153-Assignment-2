# Task 2: Symbolic Conditioned Generation
# model.py - LSTM chord prediction model, training, evaluation, and MIDI generation
# Input: processed_data.pkl from data.py
# Task: given a melody (soprano), predict the accompanying chords (alto/tenor/bass)

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
EMBED_DIM = 32          # melody note embedding size
HIDDEN_DIM = 256        # LSTM hidden units
NUM_LAYERS = 2          # stacked LSTM layers
DROPOUT = 0.3           # dropout between LSTM layers and on embeddings
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 120
PATIENCE = 10           # early stopping patience (epochs without val loss improvement)


# Paths
DATA_PATH = 'task2/processed_data.pkl'
OUTPUT_DIR = 'task2/model_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)



# Dataset
class ChoraleDataset(Dataset):
    """Wraps encoded chorale sequences for DataLoader.

    Each item is a single chorale: a list of (note_idx, chord_idx) pairs.
    Store them as separate integer tensors for melody (input) and chords (target).
    """

    def __init__(self, encoded_sequences):
        self.melodies = []
        self.chords = []
        for seq in encoded_sequences:
            notes = [pair[0] for pair in seq]
            chords = [pair[1] for pair in seq]
            self.melodies.append(torch.tensor(notes, dtype=torch.long))
            self.chords.append(torch.tensor(chords, dtype=torch.long))

    def __len__(self):
        return len(self.melodies)

    def __getitem__(self, idx):
        return self.melodies[idx], self.chords[idx]


def collate_fn(batch):
    """Pad variable-length chorales to the longest sequence in the batch.

    Returns:
        melodies:  (batch, max_len)  padded with 0 (PAD token)
        chords:    (batch, max_len)  padded with -1 so CrossEntropyLoss ignores them
        lengths:   (batch,)          original sequence lengths for pack_padded_sequence
    """
    melodies, chords = zip(*batch)
    lengths = torch.tensor([len(m) for m in melodies], dtype=torch.long)


    melodies_padded = pad_sequence(melodies, batch_first=True, padding_value=0)
    chords_padded = pad_sequence(chords, batch_first=True, padding_value=-1)

    return melodies_padded, chords_padded, lengths



# Model
class MelodyToChordLSTM(nn.Module):
    """LSTM model that reads a melody note sequence and predicts a chord at each timestep.
    Involves: melody_note -> Embedding -> Dropout -> LSTM (2-layer) -> Dropout -> Linear -> chord_logits

    Embedding captures pitch relationships (nearby MIDI notes get similar vectors).
    Two LSTM layers let the model learn both local voice-leading patterns and longer
    phrase-level harmonic progressions. Dropout helps prevent overfitting on the dataset.
    """

    def __init__(self, note_vocab_size, chord_vocab_size, embed_dim, hidden_dim,
                 num_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(note_vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, chord_vocab_size)

    def forward(self, melodies, lengths):
        """
        Parameters:
            melodies: (batch, max_len) integer melody note indices
            lengths:  (batch,) original sequence lengths

        Returns:
            logits: (batch, max_len, chord_vocab_size)
        """
        # Embed melody notes
        x = self.embedding(melodies)            # (batch, max_len, embed_dim)
        x = self.embed_dropout(x)

        # Pack so the LSTM doesn't waste computation on padding tokens
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (batch, max_len, hidden_dim)

        lstm_out = self.output_dropout(lstm_out)
        logits = self.fc(lstm_out)              # (batch, max_len, chord_vocab_size)

        return logits



# Training
def train_one_epoch(model, loader, optimizer, criterion):
    """Run one training epoch and return avg loss."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for melodies, chords, lengths in loader:
        melodies = melodies.to(DEVICE)
        chords = chords.to(DEVICE)
        lengths = lengths.to(DEVICE)

        optimizer.zero_grad()
        logits = model(melodies, lengths)  # (batch, max_len, vocab)

        # Reshape for cross entropy loss
        logits_flat = logits.view(-1, logits.size(-1))
        chords_flat = chords.view(-1)

        loss = criterion(logits_flat, chords_flat)
        loss.backward()

        # Gradient clipping to help LSTM training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        num_tokens = (chords != -1).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    return total_loss / total_tokens


def evaluate(model, loader, criterion):
    """Run eval and return avg loss and chord acc."""
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        correct = 0

        for melodies, chords, lengths in loader:
            melodies = melodies.to(DEVICE)
            chords = chords.to(DEVICE)
            lengths = lengths.to(DEVICE)

            logits = model(melodies, lengths)

            logits_flat = logits.view(-1, logits.size(-1))
            chords_flat = chords.view(-1)

            loss = criterion(logits_flat, chords_flat)

            # Mask out padded positions for acc computation
            mask = (chords_flat != -1)
            preds = logits_flat.argmax(dim=-1)
            correct += (preds[mask] == chords_flat[mask]).sum().item()

            num_tokens = mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        avg_loss = total_loss / total_tokens
        accuracy = correct / total_tokens
        return avg_loss, accuracy


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, patience):
    """Model training using early stopping, LR scheduling, and metric logging."""
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': [],
    }

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    print(f"\nTraining on {DEVICE} for up to {num_epochs} epochs (patience={patience})")
    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Train Acc':>10}  {'Val Acc':>10}  {'LR':>10}")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        _, train_acc = evaluate(model, train_loader, criterion)

        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"{epoch:5d}  {train_loss:10.4f}  {val_loss:10.4f}  {train_acc:10.4f}  {val_acc:10.4f}  {current_lr:10.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} (best was epoch {best_epoch})")
            break

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
        print(f"Restored best model from epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    return history



# Plotting Curves
def plot_training_curves(history, baseline_accuracy):
    """Save plots for training and validation loss, accuracy, and learning rate."""

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot 1: Loss curves
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'), dpi=150)
    plt.close()


    # Plot 2: Acc curves with baseline reference line
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history['train_acc'], label='Train Accuracy', linewidth=2)
    ax.plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax.axhline(y=baseline_accuracy, color='r', linestyle='--', linewidth=1.5,
               label=f'Baseline ({baseline_accuracy:.2%})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Chord Accuracy')
    ax.set_title('Chord Prediction Accuracy vs. Baseline')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_curves.png'), dpi=150)
    plt.close()


    # Plot 3: Learning rate schedule
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(epochs, history['lr'], linewidth=2, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'lr_schedule.png'), dpi=150)
    plt.close()

    print(f"Saved training plots to {OUTPUT_DIR}/")



# MIDI Generation
def generate_midi(model, test_dataset, idx_to_note, idx_to_chord, num_chorales=3):
    """Predict chords for test chorales and save as MIDI files.

    For each selected test chorale:
    - Feed the ground-truth melody into the model
    - Take the chord prediction at each timestep
    - Reconstruct a 4-voice MIDI file (soprano + predicted alto/tenor/bass)
    """
    try:
        from midiutil import MIDIFile
    except ImportError:
        print("Midiutil not installed")
        return

    model.eval()
    midi_dir = os.path.join(OUTPUT_DIR, 'midi')
    os.makedirs(midi_dir, exist_ok=True)

    # Pick the first few test chorales
    num_to_generate = min(num_chorales, len(test_dataset))

    for i in range(num_to_generate):
        melody_tensor, true_chords_tensor = test_dataset[i]
        melody_tensor = melody_tensor.unsqueeze(0).to(DEVICE)
        length = torch.tensor([len(test_dataset.melodies[i])], dtype=torch.long)

        with torch.no_grad():
            logits = model(melody_tensor, length)
            pred_indices = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        # Convert idxs back to MIDI note numbers
        seq_len = length.item()
        midi = MIDIFile(1)
        track = 0
        channel = 0
        tempo = 120
        time_per_step = 0.25  # 16th notes at 120 BPM

        midi.addTempo(track, 0, tempo)

        for t in range(seq_len):
            # Soprano (melody) - ground truth
            note_idx = test_dataset.melodies[i][t].item()
            note_val = idx_to_note.get(note_idx, None)
            if note_val is not None and isinstance(note_val, int):
                midi.addNote(track, channel, note_val, t * time_per_step,
                             time_per_step, volume=100)

            # Predicted chord voices - alto, tenor, bass
            chord_idx = int(pred_indices[t])
            chord_val = idx_to_chord.get(chord_idx, None)
            if chord_val is not None and isinstance(chord_val, frozenset):
                for pitch in sorted(chord_val):
                    if isinstance(pitch, int):
                        midi.addNote(track, channel, pitch, t * time_per_step,
                                     time_per_step, volume=80)

        out_path = os.path.join(midi_dir, f'chorale_pred_{i+1}.mid')
        with open(out_path, 'wb') as f:
            midi.writeFile(f)
        print(f"Saved {out_path}")

    print(f"Generated {num_to_generate} MIDI files in {midi_dir}/")



# Final Eval Summary
def final_evaluation(model, test_loader, baseline_accuracy, idx_to_chord):
    """Print a detailed summary comparing LSTM accuracy to baseline on the test set."""
    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        print("\n")
        print("Final Test Set Eval:\n")
        print(f"Baseline accuracy: {baseline_accuracy:.4f}  ({baseline_accuracy:.2%})")
        print(f"LSTM model test accuracy: {test_acc:.4f}  ({test_acc:.2%})")
        print(f"LSTM model test loss: {test_loss:.4f}")
        improvement = test_acc - baseline_accuracy
        print(f"Improvement over baseline: {improvement:+.4f}  ({improvement:+.2%})")
        if test_acc > baseline_accuracy:
            print("LSTM beats baseline\n")
        else:
            print("LSTM doesn't beat baseline\n")

        # Bar chart: Baseline vs LSTM test accuracy
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = ['Baseline', 'LSTM']
        accs = [baseline_accuracy, test_acc]
        colors = ['#d9534f', '#5cb85c']
        bars = ax.bar(labels, accs, color=colors, width=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{acc:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        ax.set_ylabel('Chord Accuracy')
        ax.set_title('Test Accuracy: Baseline vs LSTM')
        ax.set_ylim(0, max(accs) * 1.3)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'test_accuracy_comparison.png'), dpi=150)
        plt.close()

        return test_acc, test_loss




def main():
    # Load processed data from data.py
    print("Processed data:")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    encoded_data = data['encoded_data']
    note_to_idx = data['note_to_idx']
    idx_to_note = data['idx_to_note']
    chord_to_idx = data['chord_to_idx']
    idx_to_chord = data['idx_to_chord']
    baseline_accuracy = data['baseline_accuracy']
    most_common_chord_idx = data['most_common_chord_idx']
    note_vocab_size = data['vocab_sizes']['note']
    chord_vocab_size = data['vocab_sizes']['chord']

    print(f"Note vocab: {note_vocab_size}")
    print(f"Chord vocab: {chord_vocab_size}")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")


    # Create datasets and dataloaders
    train_dataset = ChoraleDataset(encoded_data['train'])
    val_dataset = ChoraleDataset(encoded_data['valid'])
    test_dataset = ChoraleDataset(encoded_data['test'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, drop_last=False)

    print(f"Train: {len(train_dataset)} chorales, {len(train_loader)} batches")
    print(f"Valid: {len(val_dataset)} chorales, {len(val_loader)} batches")
    print(f"Test: {len(test_dataset)} chorales, {len(test_loader)} batches")


    # Build model
    model = MelodyToChordLSTM(
        note_vocab_size=note_vocab_size,
        chord_vocab_size=chord_vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model params: {total_params:,} total, {trainable_params:,} trainable")
    print(model)


    # Training
    history = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, 
                            learning_rate=LEARNING_RATE, patience=PATIENCE)

    plot_training_curves(history, baseline_accuracy)


    # Final eval on test set
    test_acc, test_loss = final_evaluation(model, test_loader, baseline_accuracy, idx_to_chord)


    # Generate MIDI output
    print("\nGenerating MIDI output based on test set predictions")
    generate_midi(model, test_dataset, idx_to_note, idx_to_chord, num_chorales=3)


    # Save the trained model
    model_path = os.path.join(OUTPUT_DIR, 'lstm_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'hyperparameters': {
            'embed_dim': EMBED_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
        },
        'note_vocab_size': note_vocab_size,
        'chord_vocab_size': chord_vocab_size,
        'test_accuracy': test_acc,
        'baseline_accuracy': baseline_accuracy,
        'history': history,
    }, model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == '__main__':
    main()
