import torch
import torch.nn as nn
from transformers import HubertModel, HubertConfig
from peft import LoraConfig, get_peft_model


class HubertForSER(nn.Module):
    """
    HuBERT model for Speech Emotion Recognition.
    This model can be fine-tuned in three different ways:
    1. Full fine-tuning (all parameters)
    2. Parameter-efficient fine-tuning (PEFT) on QKV projection layers
    3. PEFT on classifier layer only
    """
    
    def __init__(self, num_emotions, fine_tuning_type="full", hidden_size=256):
        super().__init__()
        
        # Load pretrained HuBERT model
        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        
        # Projection layer
        self.projector = nn.Linear(768, hidden_size)  # HuBERT base has 768 hidden dim
        
        # Batch normalization as mentioned in the paper
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)  # *2 because we concatenate mean and std
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size * 2, num_emotions)
        
        # Apply the specified fine-tuning approach
        self.fine_tuning_type = fine_tuning_type
        
        if fine_tuning_type == "full":
            # Full fine-tuning - all parameters are trainable
            pass
        
        elif fine_tuning_type == "qkv":
            # Freeze all parameters except QKV projection layers
            for name, param in self.hubert.named_parameters():
                param.requires_grad = False
                
            # Apply LoRA to QKV projection layers
            config = LoraConfig(
                r=8,  # Rank of the update matrices
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )
            self.hubert = get_peft_model(self.hubert, config)
            
        elif fine_tuning_type == "classifier":
            # Freeze all parameters except classifier layer
            for param in self.hubert.parameters():
                param.requires_grad = False
            
            # Instead of using LoRA for the classifier, we'll just make it trainable
            # This is more straightforward and avoids issues with applying LoRA to a single Linear layer
            for param in self.classifier.parameters():
                param.requires_grad = True
            
            # Also make sure batch_norm and projector are trainable
            for param in self.batch_norm.parameters():
                param.requires_grad = True
            for param in self.projector.parameters():
                param.requires_grad = True
    
    def forward(self, input_values, attention_mask=None):
        # Get HuBERT outputs
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get the hidden states from the last layer
        hidden_states = outputs.last_hidden_state
        
        # Apply projection
        projected = self.projector(hidden_states)
        
        # Apply STD pooling (as described in the paper)
        # Calculate mean
        mean_pooled = torch.mean(projected, dim=1)
        # Calculate standard deviation
        std_pooled = torch.std(projected, dim=1)
        # Concatenate mean and std
        pooled = torch.cat([mean_pooled, std_pooled], dim=1)
        
        # Apply batch normalization
        pooled = self.batch_norm(pooled)
        
        # Apply dropout
        pooled = self.dropout(pooled)
        
        # Apply classifier
        logits = self.classifier(pooled)
        
        return logits
    
    def count_parameters(self):
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
