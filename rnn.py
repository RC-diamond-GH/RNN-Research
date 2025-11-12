"""
RNN (Recurrent Neural Network) Implementation from Scratch using NumPy

This module implements a vanilla RNN cell and model with forward and backward passes.
The implementation includes:
- RNN cell with tanh activation
- Forward propagation through time
- Backpropagation through time (BPTT)
- Gradient clipping for stability
"""

import numpy as np
from typing import Tuple, List, Optional


class RNNCell:
    """
    A single RNN cell that processes one time step.
    
    The RNN cell computes: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
    """
    
    def __init__(self, input_size: int, hidden_size: int, seed: Optional[int] = None):
        """
        Initialize RNN cell parameters.
        
        Args:
            input_size: Size of input vector at each time step
            hidden_size: Size of hidden state vector
            seed: Random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if seed is not None:
            np.random.seed(seed)
        
        # Weight matrices and bias
        # W_hh: hidden-to-hidden weights (hidden_size x hidden_size)
        # W_xh: input-to-hidden weights (input_size x hidden_size)
        # b_h: hidden bias (hidden_size,)
        
        # Xavier/Glorot initialization
        scale = np.sqrt(1.0 / (input_size + hidden_size))
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.W_xh = np.random.randn(input_size, hidden_size) * scale
        self.b_h = np.zeros((hidden_size,))
        
        # Gradients
        self.dW_hh = np.zeros_like(self.W_hh)
        self.dW_xh = np.zeros_like(self.W_xh)
        self.db_h = np.zeros_like(self.b_h)
        
        # Cache for backward pass
        self.last_input = None
        self.last_hidden = None
        self.last_pre_activation = None
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass through RNN cell.
        
        Args:
            x: Input vector at current time step (input_size,)
            h_prev: Hidden state from previous time step (hidden_size,)
            
        Returns:
            h: New hidden state (hidden_size,)
        """
        # Compute pre-activation
        pre_activation = np.dot(h_prev, self.W_hh) + np.dot(x, self.W_xh) + self.b_h
        
        # Apply tanh activation
        h = np.tanh(pre_activation)
        
        # Cache for backward pass
        self.last_input = x
        self.last_hidden = h_prev
        self.last_pre_activation = pre_activation
        
        return h
    
    def backward(self, dh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass through RNN cell.
        
        Args:
            dh: Gradient w.r.t. hidden state (hidden_size,)
            
        Returns:
            dx: Gradient w.r.t. input (input_size,)
            dh_prev: Gradient w.r.t. previous hidden state (hidden_size,)
        """
        # Gradient through tanh: d(tanh(z))/dz = 1 - tanh(z)^2
        dpre_activation = dh * (1 - np.tanh(self.last_pre_activation) ** 2)
        
        # Gradients w.r.t. parameters
        self.dW_hh += np.outer(self.last_hidden, dpre_activation)
        self.dW_xh += np.outer(self.last_input, dpre_activation)
        self.db_h += dpre_activation
        
        # Gradients w.r.t. inputs
        dx = np.dot(dpre_activation, self.W_xh.T)
        dh_prev = np.dot(dpre_activation, self.W_hh.T)
        
        return dx, dh_prev
    
    def reset_gradients(self):
        """Reset accumulated gradients to zero."""
        self.dW_hh.fill(0)
        self.dW_xh.fill(0)
        self.db_h.fill(0)
    
    def clip_gradients(self, max_norm: float = 5.0):
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            max_norm: Maximum gradient norm
        """
        # Compute total norm
        total_norm = np.sqrt(
            np.sum(self.dW_hh ** 2) +
            np.sum(self.dW_xh ** 2) +
            np.sum(self.db_h ** 2)
        )
        
        if total_norm > max_norm:
            clip_factor = max_norm / total_norm
            self.dW_hh *= clip_factor
            self.dW_xh *= clip_factor
            self.db_h *= clip_factor


class RNN:
    """
    Full RNN model with multiple time steps.
    
    Supports forward propagation through time and backpropagation through time (BPTT).
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seed: Optional[int] = None
    ):
        """
        Initialize RNN model.
        
        Args:
            input_size: Size of input vector at each time step
            hidden_size: Size of hidden state
            output_size: Size of output vector
            seed: Random seed for reproducibility
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # RNN cell
        self.cell = RNNCell(input_size, hidden_size, seed)
        
        # Output layer weights
        if seed is not None:
            np.random.seed(seed + 1)
        scale = np.sqrt(1.0 / hidden_size)
        self.W_hy = np.random.randn(hidden_size, output_size) * scale
        self.b_y = np.zeros((output_size,))
        
        # Gradients for output layer
        self.dW_hy = np.zeros_like(self.W_hy)
        self.db_y = np.zeros_like(self.b_y)
        
        # Cache for backward pass
        self.hidden_states = []
        self.inputs = []
        self.outputs = []
    
    def forward(self, X: np.ndarray, h0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through entire sequence.
        
        Args:
            X: Input sequence (seq_len, batch_size, input_size) or (seq_len, input_size)
            h0: Initial hidden state (batch_size, hidden_size) or (hidden_size,). If None, uses zeros.
            
        Returns:
            Y: Output sequence (seq_len, batch_size, output_size) or (seq_len, output_size)
            H: Hidden states (seq_len, batch_size, hidden_size) or (seq_len, hidden_size)
        """
        # Handle single sample vs batch
        if X.ndim == 2:
            X = X[:, np.newaxis, :]  # (seq_len, 1, input_size)
            single_sample = True
        else:
            single_sample = False
        
        seq_len, batch_size, _ = X.shape
        
        # Initialize hidden state
        if h0 is None:
            h = np.zeros((batch_size, self.hidden_size))
        else:
            h = h0.copy()
            if h.ndim == 1:
                h = h[np.newaxis, :]
        
        # Storage for hidden states and outputs
        H = []
        Y = []
        
        # Forward through time
        for t in range(seq_len):
            x_t = X[t]  # (batch_size, input_size)
            h_new = np.zeros((batch_size, self.hidden_size))
            
            # Process each sample in batch
            for b in range(batch_size):
                h_new[b] = self.cell.forward(x_t[b], h[b])
            
            h = h_new
            
            # Compute output
            y_t = np.dot(h, self.W_hy) + self.b_y
            Y.append(y_t)
            H.append(h.copy())
        
        # Convert to numpy arrays
        Y = np.array(Y)  # (seq_len, batch_size, output_size)
        H = np.array(H)  # (seq_len, batch_size, hidden_size)
        
        # Cache for backward pass
        self.hidden_states = H
        self.inputs = X
        self.outputs = Y
        
        # Return in original format
        if single_sample:
            Y = Y[:, 0, :]
            H = H[:, 0, :]
        
        return Y, H
    
    def backward(self, dY: np.ndarray) -> np.ndarray:
        """
        Backward pass through entire sequence (BPTT).
        
        Args:
            dY: Gradient w.r.t. outputs (seq_len, batch_size, output_size) or (seq_len, output_size)
            
        Returns:
            dX: Gradient w.r.t. inputs (seq_len, batch_size, input_size) or (seq_len, input_size)
        """
        # Handle single sample vs batch
        if dY.ndim == 2:
            dY = dY[:, np.newaxis, :]
            single_sample = True
        else:
            single_sample = False
        
        seq_len, batch_size, _ = dY.shape
        
        # Reset gradients
        self.cell.reset_gradients()
        self.dW_hy.fill(0)
        self.db_y.fill(0)
        
        # Initialize gradient w.r.t. hidden state
        dh_next = np.zeros((batch_size, self.hidden_size))
        dX = []
        
        # Store pre-activations for backward pass (we need to recompute them)
        pre_activations = []
        for t in range(seq_len):
            x_t = self.inputs[t]
            h_prev = self.hidden_states[t-1] if t > 0 else np.zeros((batch_size, self.hidden_size))
            pre_act = np.dot(h_prev, self.cell.W_hh) + np.dot(x_t, self.cell.W_xh) + self.cell.b_h
            pre_activations.append(pre_act)
        
        # Backward through time
        for t in reversed(range(seq_len)):
            # Gradient from output layer
            dy_t = dY[t]  # (batch_size, output_size)
            h_t = self.hidden_states[t]  # (batch_size, hidden_size)
            
            # Gradients w.r.t. output layer parameters
            self.dW_hy += np.dot(h_t.T, dy_t)
            self.db_y += np.sum(dy_t, axis=0)
            
            # Gradient w.r.t. hidden state (from output + next time step)
            dh_t = np.dot(dy_t, self.W_hy.T) + dh_next
            
            # Backward through RNN cell
            x_t = self.inputs[t]  # (batch_size, input_size)
            h_prev = self.hidden_states[t-1] if t > 0 else np.zeros((batch_size, self.hidden_size))
            pre_act = pre_activations[t]
            
            dx_t = np.zeros_like(x_t)
            dh_prev = np.zeros_like(h_prev)
            
            # Process each sample in batch
            for b in range(batch_size):
                # Gradient through tanh
                dpre_activation = dh_t[b] * (1 - np.tanh(pre_act[b]) ** 2)
                
                # Gradients w.r.t. parameters
                self.cell.dW_hh += np.outer(h_prev[b], dpre_activation)
                self.cell.dW_xh += np.outer(x_t[b], dpre_activation)
                self.cell.db_h += dpre_activation
                
                # Gradients w.r.t. inputs
                dx_t[b] = np.dot(dpre_activation, self.cell.W_xh.T)
                dh_prev[b] = np.dot(dpre_activation, self.cell.W_hh.T)
            
            dX.insert(0, dx_t)
            dh_next = dh_prev
        
        dX = np.array(dX)  # (seq_len, batch_size, input_size)
        
        # Clip gradients
        self.cell.clip_gradients()
        
        # Return in original format
        if single_sample:
            dX = dX[:, 0, :]
        
        return dX
    
    def update_parameters(self, learning_rate: float):
        """
        Update parameters using accumulated gradients.
        
        Args:
            learning_rate: Learning rate for gradient descent
        """
        # Update RNN cell parameters
        self.cell.W_hh -= learning_rate * self.cell.dW_hh
        self.cell.W_xh -= learning_rate * self.cell.dW_xh
        self.cell.b_h -= learning_rate * self.cell.db_h
        
        # Update output layer parameters
        self.W_hy -= learning_rate * self.dW_hy
        self.b_y -= learning_rate * self.db_y
    
    def get_gradient_norm(self) -> float:
        """
        Compute the norm of all gradients.
        
        Returns:
            Total gradient norm
        """
        cell_norm = np.sqrt(
            np.sum(self.cell.dW_hh ** 2) +
            np.sum(self.cell.dW_xh ** 2) +
            np.sum(self.cell.db_h ** 2)
        )
        output_norm = np.sqrt(
            np.sum(self.dW_hy ** 2) +
            np.sum(self.db_y ** 2)
        )
        return np.sqrt(cell_norm ** 2 + output_norm ** 2)
    
    def reset_cache(self):
        """Clear cached values from forward pass."""
        self.hidden_states = []
        self.inputs = []
        self.outputs = []

