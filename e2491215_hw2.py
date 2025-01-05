import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Set random seed for reproducibility
torch.manual_seed(42)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def load_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    return df

# Custom Dataset class
class ParliamentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Training function
def train_model(model, train_loader, val_loader, device, criterion, epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print('-' * 10)
        
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f'Average training loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Average validation loss: {avg_val_loss:.4f}')
        print('\nClassification Report:')
        print(classification_report(val_labels, val_preds))

def main():
    print("Loading data...")
    df = load_data('/kaggle/input/train-ideology-power/power/power-ba-train.tsv')
    
    print("\nClass distribution in dataset:")
    print(df['label'].value_counts(normalize=True))
    
    # Check for missing values
    print("\nMissing values in dataset:")
    print(df.isnull().sum())
    
    # Print sample sizes
    print(f"\nTotal samples: {len(df)}")
    print(f"Samples with Bosnian text: {df['text'].notna().sum()}")
    print(f"Samples with English text: {df['text_en'].notna().sum()}")
    
    # Dictionary to store models and results
    models = {}
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Training for both English and Bosnian text
    for text_column in ['text', 'text_en']:
        print(f"\n{'='*50}")
        print(f"Training on {text_column}")
        print(f"{'='*50}")
        
        texts = df[text_column].values
        labels = df['label'].values
        
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"\nClass weights: {class_weights}")
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"\nTraining set size: {len(train_texts)}")
        print(f"Validation set size: {len(val_texts)}")
        
        train_dataset = ParliamentDataset(train_texts, train_labels, tokenizer)
        val_dataset = ParliamentDataset(val_texts, val_labels, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-multilingual-cased',
            num_labels=2,
            problem_type="single_label_classification"
        )
        model.to(device)
        
        # Initialize loss function with class weights
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        
        print(f"\nTraining model for {text_column}...")
        train_model(model, train_loader, val_loader, device, criterion)
     
        models[text_column] = model
        
        model_save_path = f'mbert_{text_column}_model'
        model.save_pretrained(model_save_path)
        print(f"\nModel saved to {model_save_path}")

if __name__ == "__main__":
    main()