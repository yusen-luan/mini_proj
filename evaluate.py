import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import ctc_decode
from sklearn.metrics import precision_score, recall_score, f1_score
from training import load_and_preprocess_data, IMAGE_WIDTH, IMAGE_HEIGHT
from CTCModel import build_ctc_model


def decode_predictions(predictions, int_to_char, max_length):
    """
    Decode CTC predictions to text.
    
    Args:
        predictions: Model predictions (batch_size, time_steps, num_classes)
        int_to_char: Dictionary mapping integers to characters
        max_length: Maximum length of labels
        
    Returns:
        list: Decoded text strings (converted to lowercase)
    """
    batch_size = predictions.shape[0]
    input_length = predictions.shape[1]
    
    # Use CTC decode (greedy)
    decoded, _ = ctc_decode(
        predictions,
        input_length=tf.fill((batch_size,), input_length),
        greedy=True
    )
    
    decoded_texts = []
    for i in range(batch_size):
        # Convert decoded labels to characters (skip padding and invalid values)
        decoded_labels = [
            int_to_char[int(x)] 
            for x in decoded[0][i, :max_length].numpy() 
            if int(x) >= 0 and int(x) in int_to_char
        ]
        # Convert to lowercase for case-insensitive comparison
        decoded_texts.append(''.join(decoded_labels).lower())
    
    return decoded_texts


def calculate_character_metrics(predictions_list, labels_list, char_to_int):
    """
    Calculate character-level precision, recall, and F1 score.
    
    Args:
        predictions_list: List of predicted strings (lowercase)
        labels_list: List of ground truth strings (lowercase)
        char_to_int: Dictionary mapping characters to integers
        
    Returns:
        tuple: (precision, recall, f1)
    """
    all_pred_chars = []
    all_true_chars = []
    
    # For each sequence pair, align and compare characters
    for pred, true in zip(predictions_list, labels_list):
        max_len = max(len(pred), len(true))
        
        # Pad shorter sequence with a special token (-1) for alignment
        # Try both original case and lowercase for char_to_int lookup
        pred_chars = [char_to_int.get(c, char_to_int.get(c.lower(), -1)) for c in pred] + [-1] * (max_len - len(pred))
        true_chars = [char_to_int.get(c, char_to_int.get(c.lower(), -1)) for c in true] + [-1] * (max_len - len(true))
        
        all_pred_chars.extend(pred_chars)
        all_true_chars.extend(true_chars)
    
    # Calculate metrics (using 'weighted' average to handle class imbalance)
    precision = precision_score(all_true_chars, all_pred_chars, average='weighted', zero_division=0)
    recall = recall_score(all_true_chars, all_pred_chars, average='weighted', zero_division=0)
    f1 = f1_score(all_true_chars, all_pred_chars, average='weighted', zero_division=0)
    
    return precision, recall, f1


def calculate_sequence_accuracy(predictions_list, labels_list):
    """
    Calculate sequence-level accuracy (exact match).
    
    Args:
        predictions_list: List of predicted strings
        labels_list: List of ground truth strings
        
    Returns:
        float: Sequence accuracy
    """
    correct = sum(1 for pred, true in zip(predictions_list, labels_list) if pred == true)
    return correct / len(predictions_list)


def calculate_character_accuracy(predictions_list, labels_list):
    """
    Calculate character-level accuracy.
    
    Args:
        predictions_list: List of predicted strings
        labels_list: List of ground truth strings
        
    Returns:
        float: Character accuracy
    """
    total_chars = 0
    correct_chars = 0
    
    for pred, true in zip(predictions_list, labels_list):
        # Compare character by character up to the length of the shorter string
        min_len = min(len(pred), len(true))
        correct_chars += sum(1 for i in range(min_len) if pred[i] == true[i])
        total_chars += max(len(pred), len(true))
    
    return correct_chars / total_chars if total_chars > 0 else 0


def evaluate_model(model_path, test_dataset, int_to_char, char_to_int, test_labels, max_length, num_characters):
    """
    Evaluate the trained model on test data.
    
    Args:
        model_path: Path to saved model
        test_dataset: Test dataset
        int_to_char: Dictionary mapping integers to characters
        char_to_int: Dictionary mapping characters to integers
        test_labels: List of ground truth labels (strings)
        max_length: Maximum label length
        num_characters: Number of unique characters
        
    Returns:
        dict: Dictionary containing all metrics
    """
    print(f"Loading model from {model_path}...")
    # Recreate the model architecture and load weights
    # This avoids Lambda layer serialization issues
    model = build_ctc_model(IMAGE_HEIGHT, IMAGE_WIDTH, num_characters)
    model.load_weights(model_path)
    
    print("Making predictions on test data...")
    all_predictions = []
    all_labels = []
    
    for batch_idx, (images, labels, label_lengths) in enumerate(test_dataset):
        # Get model predictions
        predictions = model.predict(images, verbose=0)
        
        # Decode predictions
        decoded_texts = decode_predictions(predictions, int_to_char, max_length)
        
        # Get true labels
        for i in range(len(label_lengths)):
            actual_length = int(label_lengths[i].numpy())
            true_label_indices = labels[i, :actual_length].numpy()
            true_label = ''.join([int_to_char[int(idx)] for idx in true_label_indices])
            # Convert to lowercase for case-insensitive comparison
            true_label = true_label.lower()
            
            all_predictions.append(decoded_texts[i])
            all_labels.append(true_label)
    
    print(f"\nEvaluated {len(all_predictions)} samples")
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Sequence accuracy (exact match)
    seq_accuracy = calculate_sequence_accuracy(all_predictions, all_labels)
    
    # Character-level accuracy
    char_accuracy = calculate_character_accuracy(all_predictions, all_labels)
    
    # Character-level precision, recall, F1
    precision, recall, f1 = calculate_character_metrics(all_predictions, all_labels, char_to_int)
    
    metrics = {
        'sequence_accuracy': seq_accuracy,
        'character_accuracy': char_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics, all_predictions, all_labels


def display_sample_predictions(predictions, labels, num_samples=10):
    """Display random sample predictions vs ground truth."""
    print("\n" + "="*80)
    print("Sample Predictions (Random):")
    print("="*80)
    
    # Randomly select indices
    num_samples = min(num_samples, len(predictions))
    random_indices = np.random.choice(len(predictions), size=num_samples, replace=False)
    
    for idx in sorted(random_indices):
        match = "✓" if predictions[idx] == labels[idx] else "✗"
        print(f"{match} True: {labels[idx]:20s} | Pred: {predictions[idx]:20s}")
    
    print("="*80)


def main():
    """Main evaluation function."""
    # Configuration
    model_path = "./model/ctc_model.keras"
    train_dir = "./data/relabelled"
    test_dir = "./data/test"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using training.py")
        return
    
    # Load and preprocess data
    print("Loading test data...")
    _, _, test_dataset, char_to_int, int_to_char, max_length = load_and_preprocess_data(
        train_dir, test_dir
    )
    
    # Load original test labels for comparison
    # Remove "-0" suffix from labels (filenames are like "label-0.png")
    test_dir_path = Path(test_dir)
    test_labels = [image.stem.rsplit('-', 1)[0] if image.stem.endswith('-0') else image.stem 
                   for image in sorted(test_dir_path.glob("*.png"))]
    
    # Get number of characters for model building
    num_characters = len(char_to_int)
    
    # Evaluate model
    metrics, predictions, labels = evaluate_model(
        model_path, test_dataset, int_to_char, char_to_int, test_labels, max_length, num_characters
    )
    
    # Display results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Sequence Accuracy (Exact Match): {metrics['sequence_accuracy']*100:.2f}%")
    print(f"Character Accuracy:               {metrics['character_accuracy']*100:.2f}%")
    print(f"Character-level Precision:        {metrics['precision']*100:.2f}%")
    print(f"Character-level Recall:           {metrics['recall']*100:.2f}%")
    print(f"Character-level F1 Score:         {metrics['f1_score']*100:.2f}%")
    print("="*80)
    
    # Display sample predictions
    display_sample_predictions(predictions, labels, num_samples=20)
    
    return metrics


if __name__ == "__main__":
    main()

