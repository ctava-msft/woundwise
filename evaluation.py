import numpy as np
from typing import List, Dict, Any
from sacrebleu import BLEU, CHRF
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from collections import defaultdict

# Optional imports with fallbacks
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: rouge_score not available. ROUGE metrics will be skipped.")
    ROUGE_AVAILABLE = False

def compute_bleu_scores(predictions: List[str], references: List[List[str]], language: str = "en") -> Dict[str, float]:
    """Compute BLEU scores using sacrebleu."""
    
    # Prepare references for sacrebleu (transpose list of lists)
    if references and isinstance(references[0], list):
        # Multiple references per prediction
        ref_lists = list(zip(*references))
    else:
        # Single reference per prediction
        ref_lists = [references]
    
    # Choose tokenizer based on language
    tokenize = '13a' if language == 'en' else 'zh'
    
    try:
        bleu = BLEU(tokenize=tokenize)
        chrf = CHRF()
        
        # Compute corpus-level scores
        bleu_score = bleu.corpus_score(predictions, ref_lists)
        chrf_score = chrf.corpus_score(predictions, ref_lists)
        
        return {
            'bleu': bleu_score.score,
            'chrf': chrf_score.score
        }
    except Exception as e:
        print(f"Error computing BLEU/chrF scores: {e}")
        return {'bleu': 0.0, 'chrf': 0.0}

def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores."""
    
    if not ROUGE_AVAILABLE:
        print("ROUGE scorer not available, skipping ROUGE metrics")
        return {}
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            if isinstance(ref, list):
                ref = ref[0]  # Use first reference if multiple
            
            scores = scorer.score(ref, pred)
            for metric, score in scores.items():
                rouge_scores[f'{metric}_f'].append(score.fmeasure)
                rouge_scores[f'{metric}_p'].append(score.precision)
                rouge_scores[f'{metric}_r'].append(score.recall)
        
        # Average scores
        avg_scores = {}
        for metric, scores in rouge_scores.items():
            avg_scores[metric] = np.mean(scores)
        
        return avg_scores
    except Exception as e:
        print(f"Error computing ROUGE scores: {e}")
        return {}

def compute_meteor_scores(predictions: List[str], references: List[str]) -> float:
    """Compute METEOR scores."""
    
    try:
        meteor_scores = []
        
        for pred, ref in zip(predictions, references):
            if isinstance(ref, list):
                ref = ref[0]  # Use first reference if multiple
            
            try:
                score = meteor_score([ref.split()], pred.split())
                meteor_scores.append(score)
            except:
                meteor_scores.append(0.0)
        
        return np.mean(meteor_scores)
    except Exception as e:
        print(f"Error computing METEOR scores: {e}")
        return 0.0

def compute_nltk_bleu_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU scores using NLTK as fallback."""
    
    try:
        # Prepare references for NLTK
        tokenized_refs = []
        tokenized_preds = []
        
        for pred, ref in zip(predictions, references):
            if isinstance(ref, list):
                ref = ref[0]  # Use first reference if multiple
            
            tokenized_preds.append(pred.split())
            tokenized_refs.append([ref.split()])
        
        # Compute BLEU scores
        bleu1 = corpus_bleu(tokenized_refs, tokenized_preds, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(tokenized_refs, tokenized_preds, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(tokenized_refs, tokenized_preds, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(tokenized_refs, tokenized_preds, weights=(0.25, 0.25, 0.25, 0.25))
        
        return {
            'nltk_bleu1': bleu1,
            'nltk_bleu2': bleu2,
            'nltk_bleu3': bleu3,
            'nltk_bleu4': bleu4
        }
    except Exception as e:
        print(f"Error computing NLTK BLEU scores: {e}")
        return {}

def compute_length_stats(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute length statistics."""
    
    pred_lengths = [len(pred.split()) for pred in predictions]
    ref_lengths = []
    
    for ref in references:
        if isinstance(ref, list):
            ref_lengths.append(len(ref[0].split()))
        else:
            ref_lengths.append(len(ref.split()))
    
    return {
        'avg_pred_length': np.mean(pred_lengths),
        'avg_ref_length': np.mean(ref_lengths),
        'length_ratio': np.mean(pred_lengths) / max(np.mean(ref_lengths), 1)
    }

def compute_metrics(
    predictions: List[str],
    references: List[Any],
    language: str = "en"
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    
    if not predictions or not references:
        return {}
    
    # Ensure we have the same number of predictions and references
    min_len = min(len(predictions), len(references))
    predictions = predictions[:min_len]
    references = references[:min_len]
    
    # Flatten references if they're lists of lists
    flat_references = []
    for ref in references:
        if isinstance(ref, list) and ref:
            flat_references.append(ref[0])  # Use first reference
        elif isinstance(ref, str):
            flat_references.append(ref)
        else:
            flat_references.append("")  # Empty string for missing references
    
    metrics = {}
    
    try:
        # BLEU and chrF scores using sacrebleu
        bleu_scores = compute_bleu_scores(predictions, [flat_references], language)
        metrics.update(bleu_scores)
    except Exception as e:
        print(f"Error computing sacrebleu scores: {e}")
        # Fallback to NLTK BLEU
        try:
            nltk_bleu_scores = compute_nltk_bleu_scores(predictions, flat_references)
            metrics.update(nltk_bleu_scores)
        except Exception as e2:
            print(f"Error computing NLTK BLEU scores: {e2}")
            metrics.update({'bleu': 0.0, 'chrf': 0.0})
    
    try:
        # ROUGE scores (optional)
        rouge_scores = compute_rouge_scores(predictions, flat_references)
        metrics.update(rouge_scores)
    except Exception as e:
        print(f"Error computing ROUGE scores: {e}")
    
    try:
        # METEOR score
        meteor = compute_meteor_scores(predictions, flat_references)
        metrics['meteor'] = meteor
    except Exception as e:
        print(f"Error computing METEOR score: {e}")
        metrics['meteor'] = 0.0
    
    try:
        # Length statistics
        length_stats = compute_length_stats(predictions, flat_references)
        metrics.update(length_stats)
    except Exception as e:
        print(f"Error computing length statistics: {e}")
    
    return metrics

def evaluate_model(
    model,
    dataloader,
    device,
    language: str = "en"
) -> Dict[str, Any]:
    """Evaluate model on a dataset."""
    
    model.eval()
    predictions = []
    references = []
    metadata_list = []
    
    import torch
    from tqdm import tqdm
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images']
            query_texts = batch.get('text_inputs', [])
            batch_references = batch['responses']
            batch_metadata = batch['metadata']
            
            # Handle different text input formats
            if isinstance(query_texts, dict):
                # Tokenized inputs from transformers
                query_texts = batch.get('query_text', [''] * len(images))
            elif not isinstance(query_texts, list):
                query_texts = [''] * len(images)
            
            # Generate predictions for each item in batch
            for i, image_list in enumerate(images):
                try:
                    if image_list and len(image_list) > 0:
                        # Use first image if multiple images
                        query = query_texts[i] if i < len(query_texts) else ""
                        pred = model.generate_response(image_list[0], query)
                        predictions.append(pred)
                        
                        # Handle references
                        ref = batch_references[i] if i < len(batch_references) else []
                        if isinstance(ref, str):
                            references.append([ref])
                        elif isinstance(ref, list):
                            references.append(ref)
                        else:
                            references.append([""])
                        
                        # Handle metadata
                        meta = batch_metadata[i] if i < len(batch_metadata) else {}
                        metadata_list.append(meta)
                        
                except Exception as e:
                    print(f"Error generating response for sample {i}: {e}")
                    predictions.append("")
                    references.append([""])
                    metadata_list.append({})
    
    # Compute overall metrics
    overall_metrics = compute_metrics(predictions, references, language)
    
    # Compute metrics by wound type
    wound_type_metrics = defaultdict(lambda: {'predictions': [], 'references': []})
    
    for pred, ref, meta in zip(predictions, references, metadata_list):
        wound_type = meta.get('wound_type', 'unknown')
        wound_type_metrics[wound_type]['predictions'].append(pred)
        wound_type_metrics[wound_type]['references'].append(ref)
    
    # Calculate metrics per wound type
    per_type_metrics = {}
    for wound_type, data in wound_type_metrics.items():
        if len(data['predictions']) > 0:
            type_metrics = compute_metrics(
                data['predictions'],
                data['references'],
                language
            )
            per_type_metrics[wound_type] = type_metrics
    
    return {
        'overall_metrics': overall_metrics,
        'per_wound_type_metrics': per_type_metrics,
        'num_samples': len(predictions),
        'predictions': predictions,
        'references': references
    }

def print_evaluation_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted way."""
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Overall metrics
    print("\nOVERALL METRICS:")
    print("-"*30)
    overall = results['overall_metrics']
    for metric, value in overall.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Per wound type metrics
    if 'per_wound_type_metrics' in results:
        print("\nPER WOUND TYPE METRICS:")
        print("-"*30)
        for wound_type, metrics in results['per_wound_type_metrics'].items():
            print(f"\n{wound_type.upper()}:")
            for metric, value in metrics.items():
                if metric in ['bleu', 'meteor', 'rouge1_f', 'rougeL_f']:
                    print(f"  {metric}: {value:.4f}")
    
    print(f"\nTotal samples evaluated: {results['num_samples']}")
    print("="*50)
