import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import json

from data_loader import create_data_loaders
from vqa_model import WoundWiseVQAModel
from evaluation import evaluate_model, compute_metrics

def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        # Prepare inputs
        images = batch['images']
        query_texts = [inp for inp in batch['text_inputs']]
        target_texts = [resp[0] if resp else "" for resp in batch['responses']]
        
        # Forward pass
        outputs = model(
            images=images,
            query_texts=query_texts,
            target_texts=target_texts
        )
        
        loss = outputs.get('loss', torch.tensor(0.0))
        
        if loss.item() > 0:
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to wandb
            if wandb.run:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': scheduler.get_last_lr()[0]
                })
    
    return total_loss / max(num_batches, 1)

def validate_model(
    model: nn.Module,
    dataloader,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    
    model.eval()
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch['images']
            query_texts = [inp for inp in batch['text_inputs']]
            target_texts = [resp for resp in batch['responses']]
            
            # Generate predictions
            for i, (image, query) in enumerate(zip(images, query_texts)):
                if image:
                    pred = model.generate_response(image, query)
                    predictions.append(pred)
                    references.append(target_texts[i])
    
    # Compute metrics
    metrics = compute_metrics(predictions, references)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train WoundWise VQA Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--images_path", type=str, required=True, help="Path to images")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--language", type=str, default="en", choices=["en", "zh"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project="woundwise-vqa",
            config=args.__dict__
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize model
    model = WoundWiseVQAModel(
        model_name=args.model_name,
        language=args.language
    ).to(device)
    
    # Create data loaders
    train_loader, valid_loader, test_loader = create_data_loaders(
        data_path=args.data_path,
        images_path=args.images_path,
        language=args.language,
        batch_size=args.batch_size,
        processor=model.processor,
        tokenizer=model.text_tokenizer
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_score = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch + 1
        )
        
        # Validate
        metrics = validate_model(model, valid_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Metrics: {metrics}")
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss_epoch': train_loss,
                **{f'valid_{k}': v for k, v in metrics.items()}
            })
        
        # Save best model
        current_score = metrics.get('bleu', 0.0)
        if current_score > best_score:
            best_score = current_score
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, "best_model.pt")
            )
            print(f"New best model saved with score: {best_score:.4f}")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = validate_model(model, test_loader, device)
    print(f"Test Metrics: {test_metrics}")
    
    # Save final results
    results = {
        'best_validation_score': best_score,
        'test_metrics': test_metrics,
        'args': args.__dict__
    }
    
    with open(os.path.join(args.output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    if args.use_wandb:
        wandb.log({**{f'test_{k}': v for k, v in test_metrics.items()}})
        wandb.finish()

if __name__ == "__main__":
    main()
