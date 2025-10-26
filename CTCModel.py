import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Reshape, Bidirectional, LSTM, Dense, Lambda, Rescaling, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="Custom", name="CTCModel")
class CTCModel(Model):
    """Custom CTC Model for OCR with custom training and test steps."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    def train_step(self, data):
        """Custom training step for CTC loss."""
        # Unpack the data (image, label, label_length)
        images, labels, label_lengths = data
        
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(images, training=True)
            
            # Calculate CTC loss
            batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
            input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
            input_length = input_length * tf.ones(shape=(batch_len,), dtype="int64")
            
            # Reshape input_length to (batch_size, 1) for ctc_batch_cost
            input_length = tf.reshape(input_length, [-1, 1])
            
            # Use actual label lengths and ensure proper shape (batch_size, 1)
            label_length = tf.cast(label_lengths, dtype="int64")
            if len(label_length.shape) == 1:
                label_length = tf.reshape(label_length, [-1, 1])
            
            loss = ctc_batch_cost(labels, y_pred, input_length, label_length)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        """Custom test step for CTC loss."""
        # Unpack the data (image, label, label_length)
        images, labels, label_lengths = data
        
        # Forward pass
        y_pred = self(images, training=False)
        
        # Calculate CTC loss
        batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len,), dtype="int64")
        
        # Reshape input_length to (batch_size, 1) for ctc_batch_cost
        input_length = tf.reshape(input_length, [-1, 1])
        
        # Use actual label lengths and ensure proper shape (batch_size, 1)
        label_length = tf.cast(label_lengths, dtype="int64")
        if len(label_length.shape) == 1:
            label_length = tf.reshape(label_length, [-1, 1])
        
        loss = ctc_batch_cost(labels, y_pred, input_length, label_length)
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        
        return {"loss": self.loss_tracker.result()}
    
    @property
    def metrics(self):
        return [self.loss_tracker]


def build_ctc_model(image_height, image_width, num_characters):
    """
    Build a CRNN model with CTC loss for OCR.
    
    Args:
        image_height (int): Height of input images
        image_width (int): Width of input images
        num_characters (int): Number of unique characters (excluding CTC blank)
        
    Returns:
        CTCModel: Compiled CTC model
    """
    # Define the input layer
    input_data = Input(shape=(image_height, image_width, 1), name='input_image')
    
    # Standardize values to be in the [0, 1] range
    x = Rescaling(1./255)(input_data)
    
    # Transpose the tensor to shape (None, image_width, image_height, 1)
    x = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]), 
               output_shape=(image_width, image_height, 1),
               name="transpose")(x)
    
    # Convolutional layers
    x = Conv2D(64, (3, 3), activation="relu", kernel_initializer=tf.keras.initializers.he_normal(), padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), name="pool1")(x)
    
    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer=tf.keras.initializers.he_normal(), padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), name="pool2")(x)
    
    x = Conv2D(256, (3, 3), activation="relu", kernel_initializer=tf.keras.initializers.he_normal(), padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 1), name="pool3")(x)  # Pooling over time dimension
    
    x = Reshape(target_shape=(image_width // 8, (image_height // 4) * 256), name="reshape")(x)
    x = Dense(128, activation="relu", kernel_initializer=tf.keras.initializers.he_normal())(x)
    x = Dropout(0.2)(x)
    
    # Recurrent layers (Bidirectional LSTM)
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)
    
    # Output layer (CTC)
    # Add 1 for the CTC blank label
    output = Dense(num_characters + 1, activation='softmax')(x)
    
    # Create the model with custom training step
    model = CTCModel(inputs=input_data, outputs=output, name="OCR_model")
    
    return model

