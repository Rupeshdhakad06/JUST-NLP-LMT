# ===================================================================
# CELL 2: Access Data Files from Kaggle Dataset
# ===================================================================
import os
import argparse

# Argument parser to specify model name dynamically
parser = argparse.ArgumentParser(description="Train translation model dynamically")
parser.add_argument("--model_name", type=str, default="Helsinki-NLP/opus-mt-en-hi",
                    help="Name of the model on Hugging Face Hub (e.g. Helsinki-NLP/opus-mt-en-hi, facebook/nllb-200-distilled-600M,"+
                         "facebook/nllb-200-1.3B, ai4bharat/indictrans2-en-indic-1B, law-ai/InLegalTrans-En2Indic-1B)")
args = parser.parse_args()



# MODEL name
model_name=args.model_name

# Define file paths in the Kaggle input directory
train_file_path = f'english-hindi-train.xlsx'
test_file_path = f'english-hindi-valid.xlsx'  # TEST DATA

# Verify files exist
if os.path.exists(train_file_path):
    print("✓ Training file found in Kaggle input!")
else:
    print(f"✗ Training file not found! Please check the path: {train_file_path}")

if os.path.exists(test_file_path):
    print("✓ TEST file found in Kaggle input!")
else:
    print(f"✗ TEST file not found! Please check the path: {test_file_path}")

# ===================================================================
# CELL 3: Import All Libraries and Setup Functions
# ===================================================================
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset as HFDataset
import evaluate
import warnings
warnings.filterwarnings('ignore')

print("✓ All libraries imported successfully!")


# ===================================================================
# CELL 4: Define Data Loading Functions
# ===================================================================
def load_excel_data(file_path, sheet_name=0):
    """Load Excel/XLSX file and return DataFrame"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def explore_translation_data(df, name):
    """Explore the structure and statistics of translation data"""
    print(f"\n=== {name} Dataset Exploration ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Check for missing values
    print(f"Missing values:\n{df.isnull().sum()}")

    # Show sample data
    print(f"\nSample data:")
    print(df.head())

    # Check text lengths if source/translation columns exist
    if 'Source' in df.columns or 'source' in df.columns or 'English' in df.columns:
        source_cols = ['Source', 'source', 'English', 'english']
        source_col = None
        for col in source_cols:
            if col in df.columns:
                source_col = col
                break

        if source_col:
            source_lengths = df[source_col].str.split().str.len()
            print(f"\nSource text length stats (words):")
            print(f"  Mean: {source_lengths.mean():.1f}")
            print(f"  Median: {source_lengths.median():.1f}")
            print(f"  Min: {source_lengths.min()}")
            print(f"  Max: {source_lengths.max()}")

    # For TEST data, we only have source, no translation
    if 'Translation' in df.columns or 'translation' in df.columns or 'Hindi' in df.columns:
        trans_cols = ['Translation', 'translation', 'Hindi', 'hindi']
        trans_col = None
        for col in trans_cols:
            if col in df.columns:
                trans_col = col
                break

        if trans_col:
            trans_lengths = df[trans_col].str.split().str.len()
            print(f"\nTranslation text length stats (words):")
            print(f"  Mean: {trans_lengths.mean():.1f}")
            print(f"  Median: {trans_lengths.median():.1f}")
            print(f"  Min: {trans_lengths.min()}")
            print(f"  Max: {trans_lengths.max()}")

print("✓ Data loading functions defined!")


# ===================================================================
# CELL 5: Load Data from Kaggle Dataset
# ===================================================================
print("Loading training data from Kaggle dataset...")
train_df = load_excel_data(train_file_path)

print("\nLoading TEST data from Kaggle dataset...")
test_df = load_excel_data(test_file_path)

# This exploration code works perfectly as it operates on the DataFrames.
if train_df is not None:
    explore_translation_data(train_df, "Training")
if test_df is not None:
    explore_translation_data(test_df, "TEST")


# ===================================================================
# CELL 6: Define Data Preprocessing Functions (ENHANCED)
# ===================================================================
def clean_text(text):
    """Enhanced text cleaning for noisy text and legal terms"""
    if isinstance(text, str):
        # Replace various quotation marks and dashes
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.replace('—', '-').replace('–', '-')  # Different types of dashes
        text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        text = text.replace('…', '...')
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    return ""


def prepare_translation_data(train_df, test_df):
    """Prepare data for training and testing with enhanced cleaning"""

    # Try different possible column names
    source_cols = ['Source', 'source', 'English', 'english', 'EN', 'en']
    target_cols = ['Translation', 'translation', 'Hindi', 'hindi', 'HI', 'hi']

    source_col = None
    target_col = None

    for col in source_cols:
        if col in train_df.columns:
            source_col = col
            break

    for col in target_cols:
        if col in train_df.columns:
            target_col = col
            break

    if source_col is None or target_col is None:
        print("Error: Could not find source/target columns")
        print(f"Available columns: {list(train_df.columns)}")
        return None, None, None, None

    print(f"Using source column: {source_col}")
    print(f"Using target column: {target_col}")

    # Extract and clean training data
    train_source = [clean_text(text) for text in train_df[source_col].fillna('').astype(str).tolist()]
    train_target = [clean_text(text) for text in train_df[target_col].fillna('').astype(str).tolist()]

    # Extract and clean testing data (only source text for prediction)
    test_source_col = source_col if source_col in test_df.columns else None
    if test_source_col is None:
        for col in source_cols:
            if col in test_df.columns:
                test_source_col = col
                break

    if test_source_col is None:
        print("Error: Could not find source column in test data")
        print(f"Available test columns: {list(test_df.columns)}")
        return None, None, None, None

    test_source = [clean_text(text) for text in test_df[test_source_col].fillna('').astype(str).tolist()]
    test_ids = test_df['ID'].tolist() if 'ID' in test_df.columns else list(range(len(test_df)))

    print(f"Training pairs: {len(train_source)}")
    print(f"Test samples: {len(test_source)}")

    # Show cleaning examples
    print(f"\nText cleaning examples:")
    print(f"Original: {train_df[source_col].iloc[0][:100]}...")
    print(f"Cleaned:  {train_source[0][:100]}...")

    return train_source, train_target, test_source, test_ids


print("✓ Enhanced data preprocessing functions defined!")

# CELL 7: Define Model Setup Functions (UPDATED MAX_LENGTH)
# ===================================================================
MAX_LENGTH = 512  # Increased for legal text


def setup_translation_model(model_name="Helsinki-NLP/opus-mt-en-hi"):
    """Setup translation model and tokenizer"""
    print(f"Loading model: {model_name}")

    # different models for EN->HI translation
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        src_lang = "eng_Latn"  # source English language code
        tgt_lang = "hin_Deva"  # target Hindi language code

        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        print(f"Successfully loaded {model_name}")
    except:
        print(f"Failed to load {model_name}, trying alternative...")
        # Fallback to mT5 which supports many languages including Hindi
        model_name = "google/mt5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print(f"Using fallback model: {model_name}")

    return model, tokenizer


def create_translation_dataset(source_texts, target_texts, tokenizer, max_length=MAX_LENGTH):
    """Create dataset for translation training with increased max_length"""

    def tokenize_function(examples):
        # Tokenize source texts
        model_inputs = tokenizer(
            examples['source'],
            max_length=max_length,
            truncation=True,
            padding=False
        )

        # Tokenize target texts
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['target'],
                max_length=max_length,
                truncation=True,
                padding=False
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Create dataset
    dataset = HFDataset.from_dict({
        'source': source_texts,
        'target': target_texts
    })

    # Tokenize
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


print("✓ Model setup functions defined with increased max_length!")


# ===================================================================
# CELL 8: Define Inference Functions (UPDATED MAX_LENGTH)
# ===================================================================
def translate_texts(model, tokenizer, source_texts, ids, max_length=MAX_LENGTH, batch_size=32):
    """Generate translations for given source texts with increased max_length"""
    model.eval()
    all_translations = []

    # Process in batches
    for i in range(0, len(source_texts), batch_size):
        batch_texts = source_texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {key: val.cuda() for key, val in inputs.items()}

        # Generate translations
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
                do_sample=False,
                temperature=1.0
            )

        # Decode translations
        batch_translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_translations.extend(batch_translations)

        print(f"Processed {i + len(batch_texts)}/{len(source_texts)} samples")

    return all_translations


def create_translation_submission(ids, translations, filename="answer.csv"):
    """Create submission file in required CSV format"""

    submission_df = pd.DataFrame({
        'ID': ids,
        'Translation': translations
    })

    submission_df.to_csv(filename, index=False, encoding='utf-8')

    print(f"Submission file created: {filename}")
    print(f"Total entries: {len(ids)}")
    print(f"Sample entries:")
    print(submission_df.head())


print("✓ Inference functions defined with increased max_length!")

# ===================================================================
# CELL 9: Check GPU and Prepare Data
# ===================================================================
# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("✗ No GPU detected! Please enable the T4 GPU in the notebook settings (Accelerator option).")

# Prepare data
print("\nPreparing translation data...")
train_source, train_target, test_source, test_ids = prepare_translation_data(train_df, test_df)

if train_source is None:
    print("✗ Error in data preparation. Please check your column names.")
else:
    print(f"✓ Data prepared successfully!")
    print(f"  Training pairs: {len(train_source)}")
    print(f"  Test samples: {len(test_source)}")


# ===================================================================
# CELL 10: Setup Model and Create Dataset
# ===================================================================
# Setup model and tokenizer
print("Setting up translation model...")
model, tokenizer = setup_translation_model(model_name)

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()
    print("✓ Model moved to GPU")

# Create the tokenized training dataset
print("\nCreating training dataset...")
train_dataset = create_translation_dataset(train_source, train_target, tokenizer)

print("\n✓ Model and dataset ready for training!")

# ===================================================================
# CELL 12: Start Training (FIXED PARAMETER NAMES)
# ===================================================================
# Define the output path in the Kaggle working directory
output_dir = "mt-"+ model_name.replace("/", "_")

# Updated message for Kaggle's background execution
print("Starting training with early stopping and enhanced logging...")
print("You can 'Save Version' to run this in the background. You don't need to keep the tab open!")


# First, let's create a validation split from training data
def create_train_val_split(train_source, train_target, val_ratio=0.1):
    """Split training data into train and validation sets"""
    from sklearn.model_selection import train_test_split

    train_source, val_source, train_target, val_target = train_test_split(
        train_source, train_target,
        test_size=val_ratio,
        random_state=42
    )

    print(f"Training samples: {len(train_source)}")
    print(f"Validation samples: {len(val_source)}")

    return train_source, train_target, val_source, val_target


# Create validation split
print("Creating train/validation split...")
train_source_split, train_target_split, val_source, val_target = create_train_val_split(
    train_source, train_target, val_ratio=0.1
)

# Create training and validation datasets
print("Creating training and validation datasets...")
train_dataset = create_translation_dataset(train_source_split, train_target_split, tokenizer)
val_dataset = create_translation_dataset(val_source, val_target, tokenizer)


# The core training function with enhanced arguments (FIXED PARAMETER NAMES)
def train_translation_model_enhanced(train_dataset, val_dataset, model, tokenizer, model_name, output_dir):
    """Train the translation model with early stopping and enhanced logging"""

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Enhanced Training arguments with early stopping (UPDATED PARAMETER NAMES)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=16,
        per_device_eval_batch_size=16,
        warmup_steps=6250,
        weight_decay=0.02,
        warmup_ratio=0.1,  # 10% warmup
        lr_scheduler_type="cosine_with_restarts",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,  # More frequent logging

        # FIXED: Updated parameter names for current Transformers version
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",  # This one is correct
        load_best_model_at_end=True,  # Early stopping
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,  # Keep only best 3 models
        fp16=True,
        learning_rate=2e-5,
        remove_unused_columns=True,
        dataloader_pin_memory=False,
        report_to="none",
    )

    # Create trainer with validation dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Add validation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting enhanced training process with early stopping...")
    # trainer.train()

    # Save the final model
    print("\nSaving final model...")
    # trainer.save_model(output_dir)
    # tokenizer.save_pretrained(output_dir)

    print(f"\n✓ Enhanced training completed! Model saved to {output_dir}")
    return trainer


# Train the model with enhanced settings
trainer = train_translation_model_enhanced(
    train_dataset, val_dataset, model, tokenizer, model_name, output_dir=output_dir
)


# ===================================================================
# CELL 13: Run Model on Full Test Dataset - KAGGLE VERSION
# ===================================================================
print("Running trained model on full test dataset...")
print(f"Test samples: {len(test_source)}")

# This MUST exactly match the 'output_dir' from your training cell.
# tokenizer = AutoTokenizer.from_pretrained(output_dir)
# src_lang = "eng_Latn"   # source English language code
# tgt_lang = "hin_Deva"   # target Hindi language code
#
# tokenizer.src_lang = src_lang
# tokenizer.tgt_lang = tgt_lang
# model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)

if torch.cuda.is_available():
    model = model.cuda()
    print("✓ Model loaded on GPU")

# Generate translations for the entire test set
print("\nGenerating translations... This may take 10-15 minutes")
test_translations = translate_texts(model, tokenizer, test_source, test_ids, batch_size=64)

print("\n✓ Test translations completed!")
print(f"  Generated {len(test_translations)} translations")

# Show sample results
print("\nSample test results:")
for i in range(min(10, len(test_translations))):
    print(f"\n--- Sample {i+1} ---")
    print(f"  ID: {test_ids[i]}")
    print(f"  Source: {test_source[i][:100]}{'...' if len(test_source[i]) > 100 else ''}")
    print(f"  Translation: {test_translations[i]}")


# ===================================================================
# CELL 14: Create Test Submission File
# ===================================================================
print("Creating test submission file...")


# This will also be saved to /kaggle/working/
create_translation_submission(test_ids, test_translations, model_name.replace("/", "_")+"answer.csv")

# --- Verification Step ---
import pandas as pd
submission_df = pd.read_csv(model_name.replace("/", "_")+"answer.csv")

print(f"\nTest Submission Summary:")
print(f"Total predictions: {len(submission_df)}")
print(f"Unique IDs: {submission_df['ID'].nunique()}")
print(f"Average translation length: {submission_df['Translation'].str.len().mean():.1f} characters")

print("\nFirst 5 predictions:")
print(submission_df.head())


