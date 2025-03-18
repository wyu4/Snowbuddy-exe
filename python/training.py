import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

def load_and_preprocess_data(file_path=r"API\data\train.csv", training_fraction=0.25):  # Reduced to 25% of data
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Create binary labels
    df["offensive"] = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].max(axis=1)
    df = df[["comment_text", "offensive"]]
    df.columns = ["text", "label"]
    
    # Convert labels
    df["label_name"] = df["label"].map({0: "safe_for_snowflake", 1: "offensive"})
    
    # Initial split (stratified)
    train_df, _ = train_test_split(df, test_size=0.5, stratify=df["label"], random_state=42)
    
    # Final split with smaller subset
    train_df, test_df = train_test_split(
        train_df, 
        test_size=0.2, 
        stratify=train_df["label"], 
        random_state=42
    )
    
    return train_df, test_df

# --------------------
# 2. Model Training
# --------------------
def train_model(train_df, test_df):
    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Tokenization function
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    # Model configuration
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label={0: "safe_for_snowflake", 1: "offensive"},
        label2id={"safe_for_snowflake": 0, "offensive": 1}
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    # Metrics calculation
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    # Custom trainer for class weights
    class_weights = torch.tensor([1.0, 5.0])  # Adjust based on your dataset

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Added **kwargs
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()
    
    return model, tokenizer

# --------------------
# 3. Evaluation
# --------------------
def evaluate_model(model_path, tokenizer_path, test_df):
    # Load tokenizer and model from the saved directory
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    # Move model to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create pipeline with truncation
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,  # Ensure truncation is enabled
        max_length=512    # Set max_length to 512
    )
    
    # Test on sample data
    test_samples = test_df.sample(5)[["text", "label_name"]].values.tolist()
    
    print("\nSample Predictions:")
    for text, true_label in test_samples:
        pred = classifier(text)[0]
        print(f"Text: {text[:100]}...")
        print(f"True: {true_label} | Predicted: {pred['label']} ({pred['score']:.2f})\n")
    
    # Full evaluation
    predictions = classifier(test_df["text"].tolist())
    predicted_labels = [pred["label"] for pred in predictions]
    
    print("\nClassification Report:")
    print(classification_report(test_df["label_name"], predicted_labels))
