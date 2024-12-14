#!/usr/bin/env python3
"""
LORA XLM ROBERTA Training Script
This script implements fine-tuning of the XLM-RoBERTa model using LORA (Low-Rank Adaptation) 
for toxic content classification. It handles multiple classification labels including toxic, 
severe_toxic, obscene, threat, insult, and identity_hate.

Date: December 2024
"""

# Required library imports
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from peft import LoraConfig, TaskType, get_peft_model
from imblearn.over_sampling import RandomOverSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Global constants and configurations
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MODEL_NAME = "FacebookAI/xlm-roberta-base"  # Pre-trained model identifier
FILE_PATH = '../../data/kaggle_cleaned.csv'  # Path to the training data

# Set up GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# LORA Configuration for model fine-tuning
LORA_CONFIG = LoraConfig(
    r=8,  # Rank of the update matrices
    lora_alpha=32,  # Scaling factor for the trained weight matrices
    lora_dropout=0.1,  # Dropout probability for LORA layers
    bias="none",  # No bias parameters will be trained
    task_type=TaskType.SEQ_CLS,  # Sequence classification task
)

class BinaryDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for binary classification tasks.
    Handles the tokenization and preparation of text data for the model.
    
    Args:
        texts (list): List of input texts
        labels (list): List of corresponding labels
        tokenizer: Hugging Face tokenizer
        max_length (int): Maximum sequence length for tokenization
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns a single tokenized sample with its label
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels
        """
        # Tokenize the text with padding and truncation
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Return the tensors in the required format
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def lora_xlm_train(current_label):
    """
    Trains a LORA-adapted XLM-RoBERTa model for a specific label.
    
    Args:
        current_label (list): Single-element list containing the target label name
    """
    # Load and preprocess the dataset
    print(f"Loading and preparing data for {current_label}")
    dataset = pd.read_csv(FILE_PATH)
    dataset = dataset[dataset['data_source'] == 'Kaggle_labeled']  # Filter for labeled data
    dataset[LABEL_COLS] = dataset[LABEL_COLS].astype(int)  # Convert labels to integers
    dataset = dataset[['cleaned_text'] + current_label]  # Select relevant columns

    # Initialize tokenizer and create dataset
    print("Initializing tokenizer and creating dataset")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = BinaryDataset(
        texts=dataset['cleaned_text'].tolist(),
        labels=dataset[current_label].values,
        tokenizer=tokenizer
    )

    # Split dataset into training and validation sets
    print("Splitting dataset into train and validation sets")
    train_set, valid_set = random_split(dataset, lengths=[0.8, 0.2], 
                                      generator=torch.Generator().manual_seed(42))

    # Initialize model with LORA adaptation
    print("Initializing model with LORA configuration")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model = get_peft_model(model, LORA_CONFIG)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",  # Save the model at the end of each epoch
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=500,  # Maximum number of epochs
        logging_steps=10,  # Log every 10 steps
        save_steps=10,  # Save checkpoints every 10 steps
        save_total_limit=2,  # Keep only the last 2 checkpoints
        learning_rate=2e-5,
        weight_decay=0.01,  # L2 regularization
        push_to_hub=False,
        logging_dir="./logs",
        report_to="none",  # Disable external reporting
        load_best_model_at_end=True,  # Load the best model after training
        metric_for_best_model="eval_loss",
    )

    # Set up early stopping
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=10  # Stop if no improvement for 10 epochs
    )

    # Initialize trainer
    print("Setting up trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=valid_set,
        tokenizer=tokenizer,
        callbacks=[early_stopping]
    )

    # Train the model
    print(f"Starting training for {current_label}")
    trainer.train()

    # Save the trained model
    print(f"Saving model for {current_label}")
    model.save_pretrained(f"./models/basic_lora_xlm_{current_label}")

def main():
    """
    Main execution function that handles the training process for all labels.
    Prints progress information and handles the overall training workflow.
    """
    print(f"Device: {device}")
    print(f"Transformers version: {transformers.__version__}")

    # Train models for each label
    for label in LABEL_COLS:
        print(f"\nStarting training process for {label}")
        print("=" * 50)
        lora_xlm_train([label])
        print(f"Training completed for {label}")
        print("-" * 50)
    
    print("\nAll training completed successfully!")

if __name__ == "__main__":
    main()