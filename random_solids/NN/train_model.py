# This version keeps the three inputs separate, processes each with a small CNN,
#  concatenates, and predicts a single output matrix. Set channels-last
#  ((H, W, 1) per input). For stacked-channel input, you can also use 
# a single Input(shape=(H, W, 3)).
# ------------------------------------------------------------
# 3 inputs → 1 output matrix (Keras, regression)
# ------------------------------------------------------------
import os
import numpy as np

# Configure threading before importing TensorFlow to prevent mutex issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configure TensorFlow threading
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Custom function to copy coordinates from input to output
def copy_coordinates(inputs):
    """
    Copy first 4 columns (coordinates) from input to output, replacing predicted values.
    This ensures coordinates pass through the model unaltered.
    
    Args:
        inputs: [predicted, top_view] where:
            predicted: model output (batch, height, 104, 1)
            top_view: input 1 (batch, height, 104, 1)
    
    Returns:
        output: (batch, height, 104, 1) with columns 0-3 from top_view,
                columns 4+ from predicted
    """
    predicted, top_view = inputs
    # Extract first 4 columns from top view: [:, :, :4, :]
    coords = top_view[:, :, :4, :]  # (batch, height, 4, 1)
    # Extract remaining columns from predicted: [:, :, 4:, :]
    connectivity = predicted[:, :, 4:, :]  # (batch, height, 100, 1)
    # Concatenate: [coords, connectivity]
    return tf.concat([coords, connectivity], axis=2)

# Custom loss function for imbalanced connectivity data
def weighted_mse_loss(edge_weight=100.0, mask_first_cols=0):
    """
    Weighted MSE loss that heavily penalizes errors on edges.
    Masks first N columns (coordinates) to focus only on connectivity.
    
    Args:
        edge_weight: Weight multiplier for edge pixels (default 100x)
        mask_first_cols: Number of columns to mask (0 = no masking)
    
    Returns:
        Loss function that can be used in model.compile()
    """
    def loss(y_true, y_pred):
        # Create mask to zero out first columns (coordinates)
        if mask_first_cols > 0:
            shape = tf.shape(y_true)
            batch_size = shape[0]
            n_rows = shape[1]
            n_cols = shape[2]
            
            # Create column mask: [0, 0, 0, 0, 1, 1, 1, ...]
            col_indices = tf.range(n_cols)
            mask = tf.cast(col_indices >= mask_first_cols, dtype='float32')
            # Reshape mask to broadcast: (1, 1, cols, 1)
            mask = tf.reshape(mask, (1, 1, n_cols, 1))
        else:
            mask = 1.0
        
        # Create weight mask: edges (value > 1.0) weighted more heavily
        edge_weights = tf.where(y_true > 1.0, edge_weight, 1.0)
        
        # Combine both masks
        final_weights = edge_weights * mask
        
        # Compute weighted MSE
        squared_diff = tf.square(y_true - y_pred)
        weighted_loss = final_weights * squared_diff
        
        # Normalize by sum of weights to get proper loss scale
        total_loss = tf.reduce_sum(weighted_loss)
        total_weight = tf.reduce_sum(final_weights) + 1e-7
        
        return total_loss / total_weight
    
    return loss


def weighted_mae_metric(edge_weight=100.0, mask_first_cols=0):
    """
    Weighted MAE metric for monitoring training.
    Masks first N columns like the loss function.
    """
    def metric(y_true, y_pred):
        # Create mask to zero out first columns
        if mask_first_cols > 0:
            shape = tf.shape(y_true)
            n_cols = shape[2]
            
            col_indices = tf.range(n_cols)
            mask = tf.cast(col_indices >= mask_first_cols, dtype='float32')
            mask = tf.reshape(mask, (1, 1, n_cols, 1))
        else:
            mask = 1.0
        
        edge_weights = tf.where(y_true > 1.0, edge_weight, 1.0)
        final_weights = edge_weights * mask
        
        abs_diff = tf.abs(y_true - y_pred)
        weighted_error = final_weights * abs_diff
        
        total_error = tf.reduce_sum(weighted_error)
        total_weight = tf.reduce_sum(final_weights) + 1e-7
        
        return total_error / total_weight
    
    return metric

def branch(inp, base=32, dropout_rate=0.3):
    x = Conv2D(base, 3, padding='same')(inp); x = BatchNormalization()(x); x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(base, 3, padding='same')(x);   x = BatchNormalization()(x); x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    return x

def build_three_to_one_model(h, w, task='regression', num_classes=None, dropout_rate=0.4, 
                            use_weighted_loss=True, edge_weight=100.0, output_scale=22.5):
    """
    Build 3-input to 1-output CNN model.
    
    Args:
        h, w: Input height and width
        task: 'regression' or 'classification'
        num_classes: Number of classes (for classification)
        dropout_rate: Dropout rate
        use_weighted_loss: Use weighted MSE loss for imbalanced data
        edge_weight: Weight multiplier for edges (default 100x)
        output_scale: Scale factor for output (default 2.0 to match connectivity values)
    """
    inp1 = Input(shape=(h, w, 1))
    inp2 = Input(shape=(h, w, 1))
    inp3 = Input(shape=(h, w, 1))

    b1 = branch(inp1, base=32, dropout_rate=dropout_rate)
    b2 = branch(inp2, base=32, dropout_rate=dropout_rate)
    b3 = branch(inp3, base=32, dropout_rate=dropout_rate)

    x = Concatenate()([b1, b2, b3])  # (H, W, 96)
    x = Dropout(dropout_rate + 0.1)(x)  # Higher dropout after concatenation
    x = Conv2D(64, 3, padding='same')(x); x = BatchNormalization()(x); x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(64, 3, padding='same')(x); x = BatchNormalization()(x); x = ReLU()(x)
    x = Dropout(dropout_rate)(x)

    if task == 'regression':
        # Output layer with ReLU to ensure non-negative values
        out = Conv2D(1, 1, padding='same', activation='relu')(x)
        
        # Scale output to match target range [0, 45] to include coordinates
        # Coordinates range: idx[0-100], x[0-35], y[0-20], z[0-20]
        # Connectivity: [0-2]
        if output_scale != 1.0:
            out = Lambda(lambda x: x * output_scale, name='output_scaling')(out)
        
        # Copy coordinates from input (columns 0-3) to output, replacing predicted values
        # This ensures coordinates are preserved exactly from the top view input
        out = Lambda(copy_coordinates, name='copy_coords')([out, inp1])
        print("  Using coordinate copying: columns 0-3 from inp1, columns 4+ predicted")
        
        # Use weighted loss for imbalanced data
        if use_weighted_loss:
            loss = weighted_mse_loss(edge_weight=edge_weight)
            metrics = ['mae', weighted_mae_metric(edge_weight=edge_weight)]
            print(f"  Using weighted MSE loss (edge_weight={edge_weight}x, output_scale={output_scale})")
        else:
            loss = 'mse'
            metrics = ['mae']
            print(f"  Using standard MSE loss (output_scale={output_scale})")
    else:
        # Per-pixel classification: C channels + softmax
        assert num_classes is not None
        out = Conv2D(num_classes, 1, padding='same', activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']

    model = Model(inputs=[inp1, inp2, inp3], outputs=out)
    model.compile(optimizer=Adam(1e-3), loss=loss, metrics=metrics)
    return model

# ---- Usage with connectivity matrices ----
def train_connectivity_model(data_file='training_data.npz', epochs=50, batch_size=8, model_file='connectivity_model.h5'):
    """
    Train neural network model on connectivity matrices.
    
    Args:
        data_file: Path to prepared training data file
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_file: Path to save trained model (default: connectivity_model.h5)
        
    Returns:
        model: Trained Keras model
        history: Training history
    """
    # Load prepared data
    print(f"Loading training data from: {data_file}")
    data = np.load(data_file)
    
    X1 = data['X1']  # (N_samples, 100, 104) - top view
    X2 = data['X2']  # (N_samples, 100, 104) - front view
    X3 = data['X3']  # (N_samples, 100, 104) - side view
    Y = data['Y']    # (N_samples, 100, 104) - solid connectivity (ground truth)
    
    print(f"Loaded {len(X1)} samples")
    print(f"  X1 shape: {X1.shape}")
    print(f"  X2 shape: {X2.shape}")
    print(f"  X3 shape: {X3.shape}")
    print(f"  Y shape: {Y.shape}")
    
    # Get dimensions
    h, w = X1.shape[1], X1.shape[2]  # Should be (100, 104)
    
    # Build model with weighted loss and output scaling
    print(f"\nBuilding model for input size: ({h}, {w})")
    print(f"  Addressing imbalanced data (99% zeros, 1% edges)")
    print(f"  Model learns both coordinates and connectivity (view-to-solid mapping)")
    model = build_three_to_one_model(
        h, w, 
        task='regression',
        use_weighted_loss=True,
        edge_weight=100.0,  # Weight edges 100x more than non-edges
        output_scale=22.5    # Scale output to [0, 45] range for coords+connectivity
    )
    model.summary()
    
    # Expand channels: (N, H, W) → (N, H, W, 1)
    X1c = np.expand_dims(X1, -1)
    X2c = np.expand_dims(X2, -1)
    X3c = np.expand_dims(X3, -1)
    Yc = np.expand_dims(Y, -1)
    
    print(f"\nTraining model...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Validation split: 0.2")
    print(f"  Dropout rate: 0.4 (branches), 0.5 (concatenation)")
    print(f"  Learning rate: 1e-3 (Adam)")
    print(f"  Early stopping: patience=15, monitoring val_loss")
    print(f"  Learning rate reduction: patience=10, factor=0.5")
    print(f"\nData statistics:")
    print(f"  Target full matrix: {Y.min():.3f} to {Y.max():.3f}")
    print(f"  Connectivity cols (4+): 0.0 to 2.0")
    print(f"  Non-zero ratio: {(Y > 0.5).sum() / Y.size * 100:.2f}%")
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'best_connectivity_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1,
            min_lr=1e-6
        )
    ]
    
    # Train model
    history = model.fit(
        [X1c, X2c, X3c], Yc,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\nTraining complete!")
    
    # Save final model (best weights already restored by EarlyStopping)
    model.save(model_file)
    print(f"Final model saved to: {model_file}")
    print(f"Best model saved to: best_connectivity_model.h5")
    
    return model, history


# ---- Command-line interface ----
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train neural network on connectivity matrices'
    )
    parser.add_argument(
        '--data', type=str, default='training_data.npz',
        help='Path to training data file (default: training_data.npz)'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8,
        help='Batch size (default: 8)'
    )
    parser.add_argument(
        '--model', type=str, default='connectivity_model.h5',
        help='Path to save trained model (default: connectivity_model.h5)'
    )
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_connectivity_model(
        data_file=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_file=args.model
    )
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Training and Validation MAE')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Training history plot saved to: training_history.png")
    plt.show()
