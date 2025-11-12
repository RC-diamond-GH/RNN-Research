"""
Example script demonstrating RNN usage.

This script shows how to:
1. Create an RNN model
2. Perform forward pass
3. Compute loss and gradients
4. Train the model on a simple sequence task
"""

import numpy as np
from rnn import RNN


def generate_addition_problem(seq_len: int = 10, num_samples: int = 100):
    """
    Generate addition problem dataset.
    
    The task is to sum two numbers from a sequence.
    For simplicity, we'll use a binary representation.
    
    Args:
        seq_len: Length of input sequence
        num_samples: Number of samples to generate
        
    Returns:
        X: Input sequences (num_samples, seq_len, input_size)
        y: Target outputs (num_samples, output_size)
    """
    input_size = 2  # Two numbers to add
    output_size = 1  # Sum
    
    X = []
    y = []
    
    for _ in range(num_samples):
        # Generate two random numbers
        a = np.random.randint(0, 10)
        b = np.random.randint(0, 10)
        target = a + b
        
        # Create sequence: first number, then second number, then zeros
        sequence = np.zeros((seq_len, input_size))
        sequence[0, 0] = a / 10.0  # Normalize
        sequence[1, 1] = b / 10.0  # Normalize
        
        X.append(sequence)
        y.append(target / 20.0)  # Normalize target
    
    return np.array(X), np.array(y)


def mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Compute mean squared error."""
    return np.mean((y_pred - y_true) ** 2)


def train_rnn_example():
    """Train RNN on addition problem."""
    print("=" * 60)
    print("RNN Training Example - Addition Problem")
    print("=" * 60)
    
    # Hyperparameters
    input_size = 2
    hidden_size = 16
    output_size = 1
    seq_len = 10
    num_samples = 200
    learning_rate = 0.01
    num_epochs = 100
    batch_size = 10
    
    # Generate data
    print("\nGenerating dataset...")
    X, y = generate_addition_problem(seq_len, num_samples)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split into train and test
    split_idx = int(0.8 * num_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create model
    print("\nInitializing RNN model...")
    model = RNN(input_size, hidden_size, output_size, seed=42)
    print(f"Model initialized with {hidden_size} hidden units")
    
    # Training loop
    print("\nStarting training...")
    print("-" * 60)
    
    gradient_norms = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            batch_loss = 0.0
            
            # Process each sample in batch
            for j in range(len(batch_X)):
                # Forward pass
                Y_pred, H = model.forward(batch_X[j])  # (seq_len, output_size)
                
                # Use last output as prediction
                y_pred = Y_pred[-1, 0]  # Last time step, first (only) output
                y_true = batch_y[j]
                
                # Compute loss (MSE)
                loss = (y_pred - y_true) ** 2
                batch_loss += loss
                
                # Backward pass
                dY = np.zeros_like(Y_pred)
                dY[-1, 0] = 2 * (y_pred - y_true)  # Gradient of MSE
                model.backward(dY)
            
            # Update parameters
            model.update_parameters(learning_rate)
            model.reset_cache()
            
            epoch_loss += batch_loss / len(batch_X)
            num_batches += 1
            
            # Track gradient norm
            if num_batches == 1:  # Track first batch of each epoch
                grad_norm = model.get_gradient_norm()
                gradient_norms.append(grad_norm)
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluate on test set
        if (epoch + 1) % 10 == 0:
            test_loss = 0.0
            for i in range(len(X_test)):
                Y_pred, _ = model.forward(X_test[i])
                y_pred = Y_pred[-1, 0]
                y_true = y_test[i]
                test_loss += (y_pred - y_true) ** 2
            test_loss /= len(X_test)
            
            grad_norm = gradient_norms[-1] if gradient_norms else 0.0
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_loss:.6f} | "
                  f"Test Loss: {test_loss:.6f} | Grad Norm: {grad_norm:.4f}")
    
    print("-" * 60)
    print("\nTraining completed!")
    
    # Final evaluation
    print("\nFinal Evaluation:")
    print("-" * 60)
    
    # Show some predictions
    print("\nSample predictions (last 5 test samples):")
    for i in range(max(0, len(X_test) - 5), len(X_test)):
        Y_pred, _ = model.forward(X_test[i])
        y_pred = Y_pred[-1, 0] * 20.0  # Denormalize
        y_true = y_test[i] * 20.0  # Denormalize
        
        # Recover original numbers from input
        a = int(X_test[i][0, 0] * 10)
        b = int(X_test[i][1, 1] * 10)
        
        print(f"  {a} + {b} = {y_true:.1f} (predicted: {y_pred:.1f}, error: {abs(y_pred - y_true):.2f})")
    
    return model, gradient_norms


def simple_forward_example():
    """Simple example showing forward pass."""
    print("\n" + "=" * 60)
    print("Simple Forward Pass Example")
    print("=" * 60)
    
    # Create a small RNN
    input_size = 3
    hidden_size = 5
    output_size = 2
    seq_len = 4
    
    model = RNN(input_size, hidden_size, output_size, seed=42)
    
    # Create random input sequence
    X = np.random.randn(seq_len, input_size)
    print(f"\nInput sequence shape: {X.shape}")
    print(f"Input sequence:\n{X}")
    
    # Forward pass
    Y, H = model.forward(X)
    
    print(f"\nOutput sequence shape: {Y.shape}")
    print(f"Output sequence:\n{Y}")
    print(f"\nHidden states shape: {H.shape}")
    print(f"Hidden states:\n{H}")
    
    print("\nForward pass completed successfully!")


if __name__ == "__main__":
    # Run simple forward example
    simple_forward_example()
    
    # Run training example
    model, gradient_norms = train_rnn_example()
    
    print(f"\nGradient norms tracked: {len(gradient_norms)} epochs")
    print(f"Average gradient norm: {np.mean(gradient_norms):.4f}")
    print(f"Max gradient norm: {np.max(gradient_norms):.4f}")
    print(f"Min gradient norm: {np.min(gradient_norms):.4f}")

