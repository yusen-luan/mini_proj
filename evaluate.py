import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import ctc_decode
from sklearn.metrics import precision_score, recall_score, f1_score
from training import load_and_preprocess_data, IMAGE_WIDTH, IMAGE_HEIGHT, extract_label_from_filename
from CTCModel import build_ctc_model
import cv2


def denoise(image, threshold=50, kernel_size=3):
    """
    Apply median filtering only to black pixels (low intensity).
    
    Args:
        image: Input image (BGR format)
        threshold: Pixel intensity threshold for considering pixels as "black"
        kernel_size: Size of median filter kernel
        
    Returns:
        Denoised image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_mask = gray < threshold
    median_filtered = cv2.medianBlur(image, kernel_size)
    result = image.copy()
    result[black_mask] = median_filtered[black_mask]
    return result


def preprocess_image_advanced(img_path):
    """
    Advanced preprocessing: denoise, enhance, sharpen, and binarize.
    This follows the preprocessing pipeline from the Preprocessing.ipynb notebook.
    
    Args:
        img_path: Path to image file
        
    Returns:
        Binarized image (grayscale, single channel)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read {img_path}")
        return None

    # Step 1: Denoise - apply median filtering to black pixels
    denoised_img = denoise(img, threshold=50, kernel_size=3)

    # Step 2: Increase saturation and brightness
    hsv = cv2.cvtColor(denoised_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 1.8, 0, 255)
    v = np.clip(v * 1.3, 0, 255)
    enhanced_hsv = cv2.merge([h, s, v]).astype(np.uint8)
    enhanced_color = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # Step 3: Add contrast using CLAHE
    gray = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Step 4: Sharpen
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(contrast, -1, kernel_sharpen)

    # Step 5: Binarize using adaptive thresholding
    binary = cv2.adaptiveThreshold(sharpened, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,
                                   11, 2)
    
    return binary


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
    
    # Load training data to get character mappings
    print("Loading character mappings from training data...")
    _, _, _, char_to_int, int_to_char, max_length = load_and_preprocess_data(
        train_dir, test_dir
    )
    
    # Load and preprocess test data with advanced preprocessing
    print("Loading and preprocessing test data with denoising and binarization...")
    test_dir_path = Path(test_dir)
    test_image_paths = sorted(test_dir_path.glob("*.png"))
    
    # Load original test labels
    test_labels = [extract_label_from_filename(image.stem) for image in test_image_paths]
    
    # Apply advanced preprocessing to test images
    print("Applying advanced preprocessing (denoise, enhance, sharpen, binarize)...")
    test_images = []
    for img_path in test_image_paths:
        binary_img = preprocess_image_advanced(img_path)
        if binary_img is not None:
            # Convert to tensor and resize to expected dimensions
            # Add channel dimension (grayscale)
            binary_img = np.expand_dims(binary_img, axis=-1)
            # Resize with padding to match expected dimensions
            binary_img_tensor = tf.image.resize_with_pad(
                binary_img, IMAGE_HEIGHT, IMAGE_WIDTH
            )
            test_images.append(binary_img_tensor)
    
    # Encode test labels
    test_encoded_labels = [[char_to_int[char] for char in label.lower()] for label in test_labels]
    test_label_lengths = [len(label) for label in test_encoded_labels]
    
    # Pad labels to max_length
    def pad_label(label, max_length, padding_value=0):
        return label + [padding_value] * (max_length - len(label))
    
    test_encoded_labels_padded = [pad_label(label, max_length) for label in test_encoded_labels]
    
    # Create test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_encoded_labels_padded, test_label_lengths))
    test_dataset = test_dataset.batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print(f"Preprocessed {len(test_images)} test images")
    
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

