import numpy as np
from pathlib import Path
import tensorflow as tf
from CTCModel import build_ctc_model
import matplotlib.pyplot as plt


# Define constants
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 750
BATCH_SIZE = 16
NUM_EPOCHS = 500


def preprocess_image(image_path):
    """Preprocess an image by reading, decoding, and resizing with padding."""
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)  # Grayscale image
    # Use resize_with_pad to maintain aspect ratio and add padding as needed
    image = tf.image.resize_with_pad(image, IMAGE_HEIGHT, IMAGE_WIDTH)
    return image


def pad_label(label, max_length, padding_value=0):
    """Pad label with padding_value to reach max_length."""
    padded = label + [padding_value] * (max_length - len(label))
    return padded


def load_and_preprocess_data(train_dir, test_dir):
    """
    Load and preprocess training and test data.
    
    Args:
        train_dir (str): Path to training data directory
        test_dir (str): Path to test data directory
        
    Returns:
        tuple: (train_dataset, validation_dataset, test_dataset, char_to_int, int_to_char, max_length)
    """
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)
    
    # Load training data
    train_image_paths = [str(image) for image in sorted(train_dir.glob("*.png"))]
    # Remove "-0" suffix from labels (filenames are like "label-0.png")
    train_labels = [image.stem.rsplit('-', 1)[0] if image.stem.endswith('-0') else image.stem 
                    for image in sorted(train_dir.glob("*.png"))]
    
    # Load test data
    test_image_paths = [str(image) for image in sorted(test_dir.glob("*.png"))]
    # Remove "-0" suffix from labels (filenames are like "label-0.png")
    test_labels = [image.stem.rsplit('-', 1)[0] if image.stem.endswith('-0') else image.stem 
                   for image in sorted(test_dir.glob("*.png"))]
    
    print(f"Training samples: {len(train_image_paths)}")
    print(f"Test samples: {len(test_image_paths)}")
    
    # Combine all labels to find all unique characters and max length
    all_labels = train_labels + test_labels
    
    # Maximum length of any captcha in the dataset
    max_length = max([len(label) for label in all_labels])
    print(f"Maximum captcha length: {max_length}")
    
    # Create a set of all unique characters in the labels
    all_possible_characters = sorted(set("".join(all_labels)))
    print(f"Number of unique characters: {len(all_possible_characters)}")
    print(f"Characters: {''.join(all_possible_characters)}")
    
    # Create a mapping of characters to integers and integers to characters
    char_to_int = {char: i for i, char in enumerate(all_possible_characters)}
    int_to_char = {i: char for char, i in char_to_int.items()}
    
    # Preprocess training images and labels
    print("Preprocessing training data...")
    train_images = [preprocess_image(image_path) for image_path in train_image_paths]
    train_encoded_labels = [[char_to_int[char] for char in label] for label in train_labels]
    # Store actual label lengths before padding
    train_label_lengths = [len(label) for label in train_encoded_labels]
    # Pad all labels to max_length
    train_encoded_labels_padded = [pad_label(label, max_length) for label in train_encoded_labels]
    
    # Preprocess test images and labels
    print("Preprocessing test data...")
    test_images = [preprocess_image(image_path) for image_path in test_image_paths]
    test_encoded_labels = [[char_to_int[char] for char in label] for label in test_labels]
    # Store actual label lengths before padding
    test_label_lengths = [len(label) for label in test_encoded_labels]
    # Pad all labels to max_length
    test_encoded_labels_padded = [pad_label(label, max_length) for label in test_encoded_labels]
    
    # Create TensorFlow Datasets with (image, label, label_length) tuples
    train_dataset_full = tf.data.Dataset.from_tensor_slices((train_images, train_encoded_labels_padded, train_label_lengths))
    train_dataset_full = train_dataset_full.shuffle(buffer_size=len(train_images))
    
    # Split training data into train and validation (80/20 split)
    train_size = int(0.8 * len(train_image_paths))
    train_dataset = train_dataset_full.take(train_size).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = train_dataset_full.skip(train_size).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Create test dataset (for final evaluation)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_encoded_labels_padded, test_label_lengths))
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    print(f"\nDatasets created successfully!")
    print(f"Training batches: ~{train_size // BATCH_SIZE}")
    print(f"Validation batches: ~{(len(train_image_paths) - train_size) // BATCH_SIZE}")
    print(f"Test batches: ~{len(test_image_paths) // BATCH_SIZE}")
    
    return train_dataset, validation_dataset, test_dataset, char_to_int, int_to_char, max_length


def plot_training_history(history, save_path="./evaluation/training_loss_plot.png"):
    """
    Plot training and validation loss over epochs and save the figure.
    
    Args:
        history: Training history object from model.fit()
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    
    # Plot validation loss if available
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved to {save_path}")
    plt.close()


def train_model(train_dataset, validation_dataset, num_characters, model_save_path="./model/ctc_model.keras"):
    """
    Train the CTC model and save it.
    
    Args:
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        num_characters (int): Number of unique characters
        model_save_path (str): Path to save the trained model
        
    Returns:
        tuple: (model, history)
    """
    # Build the model
    model = build_ctc_model(IMAGE_HEIGHT, IMAGE_WIDTH, num_characters)
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.summary()
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
    ]
    
    # Train the model
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    
    # Save the trained model
    print(f"\nSaving model to {model_save_path}...")
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_save_path)
    print("Model saved successfully!")
    
    # Plot and save training history
    plot_training_history(history)
    
    return model, history


def main():
    """Main function to run the training pipeline."""
    # Define data directories
    train_dir = "./data/relabelled"
    test_dir = "./data/test"
    
    # Load and preprocess data
    train_dataset, validation_dataset, test_dataset, char_to_int, int_to_char, max_length = load_and_preprocess_data(
        train_dir, test_dir
    )
    
    # Train the model
    num_characters = len(char_to_int)
    model, history = train_model(
        train_dataset, 
        validation_dataset, 
        num_characters,
        model_save_path="./model/ctc_model.keras"
    )
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    test_loss = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

