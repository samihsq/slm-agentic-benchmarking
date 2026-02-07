"""
F1 Score evaluator for Episodic Memory Benchmark.

Calculates precision, recall, and F1 scores for list-based answers
with fuzzy matching for dates and entity names.
"""

import re
from typing import List, Set, Tuple
from difflib import SequenceMatcher


def normalize_date(date_str: str) -> str:
    """
    Normalize date string for comparison.
    
    Examples:
        "Sep 22, 2026" -> "2026-09-22"
        "September 22, 2026" -> "2026-09-22"
        "22 Sep 2026" -> "2026-09-22"
    """
    date_str = date_str.strip().lower()
    
    # Month abbreviations
    months = {
        'jan': '01', 'january': '01',
        'feb': '02', 'february': '02',
        'mar': '03', 'march': '03',
        'apr': '04', 'april': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07',
        'aug': '08', 'august': '08',
        'sep': '09', 'september': '09',
        'oct': '10', 'october': '10',
        'nov': '11', 'november': '11',
        'dec': '12', 'december': '12',
    }
    
    # Try to extract year, month, day
    # Pattern: "month day, year" or "day month year"
    for month_name, month_num in months.items():
        if month_name in date_str:
            # Extract numbers
            numbers = re.findall(r'\d+', date_str)
            if len(numbers) >= 2:
                # Determine which is day and which is year
                nums = [int(n) for n in numbers]
                year = next((n for n in nums if n > 1900), None)
                day = next((n for n in nums if 1 <= n <= 31), None)
                
                if year and day:
                    return f"{year:04d}-{month_num}-{day:02d}"
    
    # If normalization fails, return original (lowercased)
    return date_str


def normalize_string(s: str) -> str:
    """
    Normalize a string for comparison.
    
    - Lowercase
    - Remove extra whitespace
    - Remove punctuation except hyphens
    """
    s = s.lower().strip()
    # Remove punctuation except hyphens
    s = re.sub(r'[^\w\s-]', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s)
    return s


def fuzzy_match(str1: str, str2: str, threshold: float = 0.85) -> bool:
    """
    Check if two strings match using fuzzy string matching.
    
    Args:
        str1: First string
        str2: Second string
        threshold: Similarity threshold (0-1)
    
    Returns:
        True if strings are similar enough
    """
    # Exact match after normalization
    norm1 = normalize_string(str1)
    norm2 = normalize_string(str2)
    
    if norm1 == norm2:
        return True
    
    # Fuzzy match using sequence matcher
    similarity = SequenceMatcher(None, norm1, norm2).ratio()
    return similarity >= threshold


def match_items(predicted: List[str], ground_truth: List[str], fuzzy: bool = True) -> Tuple[int, int, int]:
    """
    Match predicted items against ground truth with fuzzy matching.
    
    Args:
        predicted: List of predicted items
        ground_truth: List of ground truth items
        fuzzy: Enable fuzzy matching
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    predicted_set = set(predicted)
    ground_truth_set = set(ground_truth)
    
    if not fuzzy:
        # Exact matching
        true_positives = len(predicted_set & ground_truth_set)
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)
        return true_positives, false_positives, false_negatives
    
    # Fuzzy matching
    matched_pred = set()
    matched_gt = set()
    
    # Try to match each predicted item with ground truth
    for pred_item in predicted:
        for gt_item in ground_truth:
            if gt_item in matched_gt:
                continue
            
            # Try date normalization first
            if re.search(r'\d{4}', pred_item) and re.search(r'\d{4}', gt_item):
                norm_pred = normalize_date(pred_item)
                norm_gt = normalize_date(gt_item)
                if norm_pred == norm_gt:
                    matched_pred.add(pred_item)
                    matched_gt.add(gt_item)
                    break
            
            # Try fuzzy string matching
            if fuzzy_match(pred_item, gt_item):
                matched_pred.add(pred_item)
                matched_gt.add(gt_item)
                break
    
    true_positives = len(matched_pred)
    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - len(matched_gt)
    
    return true_positives, false_positives, false_negatives


def calculate_f1(predicted: List[str], ground_truth: List[str], fuzzy: bool = True) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        predicted: List of predicted items
        ground_truth: List of ground truth items
        fuzzy: Enable fuzzy matching
    
    Returns:
        Tuple of (precision, recall, f1_score)
    """
    if not ground_truth:
        # No ground truth - perfect if no predictions, zero otherwise
        return (1.0, 1.0, 1.0) if not predicted else (0.0, 0.0, 0.0)
    
    if not predicted:
        # No predictions but ground truth exists
        return (0.0, 0.0, 0.0)
    
    tp, fp, fn = match_items(predicted, ground_truth, fuzzy=fuzzy)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1


def parse_list_response(response: str) -> List[str]:
    """
    Parse a model response into a list of items.
    
    Handles various formats:
    - JSON arrays: ["item1", "item2"]
    - Bullet lists: • item1\n• item2
    - Numbered lists: 1. item1\n2. item2
    - Comma-separated: item1, item2, item3
    
    Args:
        response: Model response string
    
    Returns:
        List of extracted items
    """
    import json
    
    response = response.strip()
    
    # Try to parse as JSON array
    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if item]
        elif isinstance(parsed, dict) and "answer" in parsed:
            answer = parsed["answer"]
            if isinstance(answer, list):
                return [str(item).strip() for item in answer if item]
            else:
                # Single answer in dict
                return [str(answer).strip()] if answer else []
    except json.JSONDecodeError:
        pass
    
    # Try to extract from JSON code block
    json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except json.JSONDecodeError:
            pass
    
    # Try to find a list in the response
    lines = response.split('\n')
    items = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Bullet points
        if re.match(r'^[•\-\*]\s+', line):
            item = re.sub(r'^[•\-\*]\s+', '', line).strip()
            if item:
                items.append(item)
        
        # Numbered lists
        elif re.match(r'^\d+[\.\)]\s+', line):
            item = re.sub(r'^\d+[\.\)]\s+', '', line).strip()
            if item:
                items.append(item)
    
    if items:
        return items
    
    # Try comma-separated
    if ',' in response:
        items = [item.strip() for item in response.split(',') if item.strip()]
        if items:
            return items
    
    # Single item response
    # Clean up common prefixes
    response = re.sub(r'^(answer|response|result)[:\s]*', '', response, flags=re.IGNORECASE)
    response = response.strip()
    
    if response:
        return [response]
    
    return []


class F1Evaluator:
    """Evaluator for episodic memory tasks using F1 score."""
    
    def __init__(self, fuzzy_matching: bool = True):
        """
        Initialize the evaluator.
        
        Args:
            fuzzy_matching: Enable fuzzy string matching
        """
        self.fuzzy_matching = fuzzy_matching
    
    def evaluate(self, predicted_response: str, ground_truth: List[str]) -> dict:
        """
        Evaluate a model response against ground truth.
        
        Args:
            predicted_response: Model's response string
            ground_truth: List of correct answers
        
        Returns:
            Dictionary with precision, recall, f1, and matched items
        """
        predicted_items = parse_list_response(predicted_response)
        precision, recall, f1 = calculate_f1(
            predicted_items,
            ground_truth,
            fuzzy=self.fuzzy_matching
        )
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted_items": predicted_items,
            "ground_truth": ground_truth,
            "num_predicted": len(predicted_items),
            "num_ground_truth": len(ground_truth),
        }
    
    def evaluate_batch(self, results: List[Tuple[str, List[str]]]) -> dict:
        """
        Evaluate a batch of results.
        
        Args:
            results: List of (predicted_response, ground_truth) tuples
        
        Returns:
            Dictionary with aggregate metrics
        """
        scores = [self.evaluate(pred, gt) for pred, gt in results]
        
        avg_precision = sum(s["precision"] for s in scores) / len(scores) if scores else 0.0
        avg_recall = sum(s["recall"] for s in scores) / len(scores) if scores else 0.0
        avg_f1 = sum(s["f1"] for s in scores) / len(scores) if scores else 0.0
        
        return {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "num_questions": len(scores),
            "individual_scores": scores,
        }
