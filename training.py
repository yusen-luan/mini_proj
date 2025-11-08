import numpy as np
from pathlib import Path
import tensorflow as tf
from CTCModel import build_ctc_model
import matplotlib.pyplot as plt


# Define constants
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 750
BATCH_SIZE = 16
NUM_EPOCHS = 100


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


def augment_image(image):
    """
    Apply random augmentations to training images on-the-fly.
    Called during training for each image in each epoch.
    Uses conservative translation and scaling to avoid cutting off characters.
    
    Uses functional TF operations (not Keras layers) to avoid Variable creation issues.
    
    Args:
        image: TensorFlow tensor of shape (height, width, 1)
        
    Returns:
        Augmented image tensor
    """
    # Get original shape
    original_shape = tf.shape(image)
    height = tf.cast(original_shape[0], tf.float32)
    width = tf.cast(original_shape[1], tf.float32)
    
    # Conservative random scaling - ONLY ZOOM OUT (95% to 100%) to avoid cutting characters
    # Zooming out is safe because we pad with white space, never lose content
    scale_factor = tf.random.uniform([], 0.95, 1.0)
    new_height = tf.cast(height * scale_factor, tf.int32)
    new_width = tf.cast(width * scale_factor, tf.int32)
    
    # Resize image (smaller)
    image = tf.image.resize(image, [new_height, new_width])
    
    # Pad back to original size with white (255) - this is safe, never crops
    image = tf.image.resize_with_crop_or_pad(
        image, 
        target_height=tf.cast(height, tf.int32), 
        target_width=tf.cast(width, tf.int32)
    )
    
    # Very small random translation (±2% instead of ±5%) to avoid cutting off edge characters
    # This is conservative but still provides position variation
    max_shift_x = tf.cast(width * 0.02, tf.int32)
    shift_x = tf.random.uniform([], -max_shift_x, max_shift_x, dtype=tf.int32)
    
    max_shift_y = tf.cast(height * 0.02, tf.int32)
    shift_y = tf.random.uniform([], -max_shift_y, max_shift_y, dtype=tf.int32)
    
    # Apply translation with sufficient padding to ensure no content loss
    padded_height = tf.cast(height, tf.int32) + tf.abs(max_shift_y) * 2
    padded_width = tf.cast(width, tf.int32) + tf.abs(max_shift_x) * 2
    
    # Pad with white (255) - typical captcha background
    image = tf.image.resize_with_crop_or_pad(image, padded_height, padded_width)
    
    # Translate by cropping at offset (content is preserved in padding)
    offset_y = tf.abs(max_shift_y) - shift_y
    offset_x = tf.abs(max_shift_x) - shift_x
    
    image = tf.image.crop_to_bounding_box(
        image,
        offset_height=offset_y,
        offset_width=offset_x,
        target_height=tf.cast(height, tf.int32),
        target_width=tf.cast(width, tf.int32)
    )
    
    # Add very light Gaussian noise to help with minor artifacts
    # Keep this minimal to avoid degrading character readability
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=1.5)
    image = image + noise
    
    # Clip values to valid range [0, 255]
    image = tf.clip_by_value(image, 0., 255.)
    
    return image


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
    train_dataset = train_dataset_full.take(train_size)
    validation_dataset = train_dataset_full.skip(train_size)
    
    # Apply augmentation ONLY to training set (not validation)
    # This happens on-the-fly during training - each epoch gets different augmentations
    print("Applying on-the-fly data augmentation to training set...")
    train_dataset = train_dataset.map(
        lambda img, lbl, lbl_len: (augment_image(img), lbl, lbl_len),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch for both datasets
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
    
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


def load_latest_checkpoint(checkpoint_dir):
    """
    Load the latest checkpoint from the checkpoint directory.
    
    Args:
        checkpoint_dir: Path to checkpoint directory
        
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Find all checkpoint files
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.keras"))
    
    if not checkpoints:
        # Check for best_model.keras
        best_model = checkpoint_dir / "best_model.keras"
        if best_model.exists():
            return str(best_model)
        return None
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(checkpoints[0])


def train_model(train_dataset, validation_dataset, num_characters, model_save_path="./model/ctc_model.keras", resume_from_checkpoint=True):
    """
    Train the CTC model and save it.
    
    Args:
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        num_characters (int): Number of unique characters
        model_save_path (str): Path to save the trained model
        resume_from_checkpoint (bool): Whether to resume from latest checkpoint if available
        
    Returns:
        tuple: (model, history)
    """
    checkpoint_dir = Path(model_save_path).parent / "checkpoints"
    
    # Check for existing checkpoint to resume from
    if resume_from_checkpoint:
        latest_checkpoint = load_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"\n{'='*60}")
            print(f"Found existing checkpoint: {latest_checkpoint}")
            print(f"Loading model to resume training...")
            print(f"{'='*60}\n")
            try:
                model = tf.keras.models.load_model(latest_checkpoint)
                print("Successfully loaded checkpoint!")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                print("Starting training from scratch...")
                model = build_ctc_model(IMAGE_HEIGHT, IMAGE_WIDTH, num_characters)
        else:
            print("\nNo checkpoint found. Starting training from scratch...")
            model = build_ctc_model(IMAGE_HEIGHT, IMAGE_WIDTH, num_characters)
    else:
        print("\nStarting training from scratch (resume_from_checkpoint=False)...")
        model = build_ctc_model(IMAGE_HEIGHT, IMAGE_WIDTH, num_characters)
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam())
    model.summary()
    
    # Define callbacks
    # Create checkpoint directory
    checkpoint_dir = Path(model_save_path).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Custom callback to save checkpoints every N epochs
    class PeriodicCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, checkpoint_dir, save_every_n_epochs=10):
            super().__init__()
            self.checkpoint_dir = Path(checkpoint_dir)
            self.save_every_n_epochs = save_every_n_epochs
            
        def on_epoch_end(self, epoch, logs=None):
            # Save checkpoint every N epochs (epoch is 0-indexed)
            if (epoch + 1) % self.save_every_n_epochs == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.keras"
                self.model.save(checkpoint_path)
                print(f"\nSaved checkpoint: {checkpoint_path}")
    
    callbacks = [
        # Save best model based on validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        # Save checkpoint every 10 epochs (for recovery)
        PeriodicCheckpoint(checkpoint_dir, save_every_n_epochs=10),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # Early stopping with best weights restoration
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased patience for better convergence
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

