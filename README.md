# karpathynns

# Neural Networks: Zero to Hero 

This repository contains my implementations and exercises from Andrej Karpathy's *Neural Networks: Zero to Hero* series.

---

## **Micrograd: A Minimal Autograd Engine**
- Built a minimal autograd engine to understand backpropagation and computational graphs at a scalar level.  
- Implemented forward and backward passes for basic mathematical operations.  
- Created a class-based framework to compute gradients of scalar-valued functions.  
- **Exercises**:
  1. Derive and implement analytical gradients for custom functions.
  2. Approximate gradients using finite differences, including symmetric derivative methods.
  3. Extend the `Value` class to support additional operations (e.g., multiplication, logarithms).

---

## **Bigram Language Model**
- Developed a bigram character-level language model for autoregressive text generation.  
- Constructed and trained a model to predict the next character based on a single preceding character.  
- Implemented a probabilistic sampling mechanism for text generation.  
- **Exercises**:
  1. Extend the model to a trigram architecture for improved predictions.
  2. Split the dataset into training, development, and test sets for robust evaluation.
  3. Explore regularization techniques for parameter smoothing.

---

## **MLP with BatchNorm**
- Designed and trained a multilayer perceptron (MLP) with Batch Normalization for improved gradient flow.  
- Added BatchNorm layers to stabilize and accelerate training.  
- Analyzed activations and gradients to diagnose training challenges.  
- **Exercises**:
  1. Train an MLP with all weights and biases initialized to zero; analyze partial training behavior.
  2. Fold BatchNorm parameters into the preceding linear layer for inference-time optimization.

---

## **WaveNet-Inspired CNN**
- Implemented a convolutional neural network inspired by *WaveNet: A Generative Model for Raw Audio* (van den Oord et al., 2016).  
- Designed a hierarchical, tree-like structure for character-level text generation.  
- Explored dilated convolutional architectures for efficient hierarchical modeling.

---

## **Manual Backpropagation**
- Re-implemented backpropagation manually for a two-layer MLP with BatchNorm.  
- Built gradients for all variables and layers, bypassing PyTorch’s `autograd`.  
- Gained in-depth understanding of gradient flow through neural network layers.  
- **Exercises**:
  1. Backpropagate manually through BatchNorm layers and cross-entropy loss.
  2. Train the MLP entirely with a custom backward pass implementation.

---

## **GPT Implementation**
- Constructed a Generatively Pretrained Transformer (GPT) from scratch based on *"Attention is All You Need"* (Vaswani et al., 2017).  
- Implemented multi-head self-attention, positional encodings, and Transformer blocks.  
- Trained the model on a custom dataset for text generation tasks.  
- **Exercises**:
  1. Merge multiple attention heads into a single parallelized computation.
  2. Pretrain on large datasets and fine-tune on smaller datasets for improved generalization.
  3. Add new features from recent Transformer papers and evaluate their impact.

---

## **GPT Tokenizer**
- Designed a Byte Pair Encoding (BPE) tokenizer to preprocess text for GPT models.  
- Implemented encoding and decoding functionalities.  
- Refined tokenization using regular expressions to replicate GPT-4 behavior.  
- **Exercises**:
  1. Train a tokenizer on a custom dataset, visualizing token merges.
  2. Compare a regex-based tokenizer with a standard implementation for token consistency.
  3. Recreate GPT-4’s tokenizer behavior, including byte shuffling and merge recovery.

---
