"""
Recall Task Generator

Generates simple recall tasks from episodic memory narratives:
- Given a passage and a keyword
- Find the sentence containing that keyword
"""

import random
import re
from typing import List, Dict, Any, Set
from pathlib import Path


class RecallTaskGenerator:
    """Generate recall tasks from episodic memory narratives."""
    
    def __init__(
        self,
        min_passage_tokens: int = 100,
        max_passage_tokens: int = 2000,
        seed: int = 42,
    ):
        """
        Initialize the generator.
        
        Args:
            min_passage_tokens: Minimum passage size in tokens
            max_passage_tokens: Maximum passage size in tokens
            seed: Random seed for reproducibility
        """
        self.min_passage_tokens = min_passage_tokens
        self.max_passage_tokens = max_passage_tokens
        random.seed(seed)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        return sentences
    
    def extract_entities(self, sentence: str) -> Set[str]:
        """
        Extract potential entity keywords from a sentence.
        
        Simple heuristic: capitalized words (proper nouns).
        """
        # Find capitalized words (likely proper nouns)
        words = sentence.split()
        entities = set()
        
        for i, word in enumerate(words):
            # Clean punctuation
            clean_word = re.sub(r'[^\w\s-]', '', word)
            
            # Skip if too short or common words
            if len(clean_word) < 3:
                continue
            if clean_word.lower() in {'the', 'and', 'but', 'for', 'with', 'that', 'this'}:
                continue
            
            # Check if capitalized (and not start of sentence)
            if clean_word and clean_word[0].isupper():
                entities.add(clean_word)
                
                # Also capture multi-word entities (e.g., "Benjamin Green")
                if i + 1 < len(words):
                    next_word = re.sub(r'[^\w\s-]', '', words[i + 1])
                    if next_word and next_word[0].isupper():
                        entities.add(f"{clean_word} {next_word}")
        
        return entities
    
    def extract_dates(self, sentence: str) -> Set[str]:
        """Extract dates from a sentence."""
        dates = set()
        
        # Pattern: "Month Day, Year" or "Day Month Year"
        date_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b',
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                dates.add(match.group(0))
        
        return dates
    
    def extract_keywords(self, sentence: str) -> Set[str]:
        """Extract all keywords (entities + dates) from a sentence."""
        keywords = set()
        keywords.update(self.extract_entities(sentence))
        keywords.update(self.extract_dates(sentence))
        return keywords
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate (1 token ≈ 4 characters)."""
        return len(text) // 4
    
    def generate_passage(self, sentences: List[str], target_tokens: int) -> List[str]:
        """
        Generate a passage of approximately target_tokens length.
        
        Args:
            sentences: Pool of sentences to select from
            target_tokens: Desired token count
        
        Returns:
            List of sentences forming the passage
        """
        if not sentences:
            return []
        
        passage = []
        current_tokens = 0
        
        # Start from a random position
        start_idx = random.randint(0, max(0, len(sentences) - 10))
        
        for i in range(start_idx, len(sentences)):
            sentence = sentences[i]
            sentence_tokens = self.estimate_tokens(sentence)
            
            if current_tokens + sentence_tokens > target_tokens and passage:
                break
            
            passage.append(sentence)
            current_tokens += sentence_tokens
            
            if current_tokens >= target_tokens:
                break
        
        return passage
    
    def generate_tasks_from_narrative(
        self,
        narrative: str,
        num_tasks: int = 100,
        difficulty_distribution: Dict[str, float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate recall tasks from a narrative.
        
        Args:
            narrative: Full narrative text
            num_tasks: Number of tasks to generate
            difficulty_distribution: Distribution of difficulties
                {"easy": 0.4, "medium": 0.4, "hard": 0.2}
        
        Returns:
            List of recall task dictionaries
        """
        if difficulty_distribution is None:
            difficulty_distribution = {"easy": 0.4, "medium": 0.4, "hard": 0.2}
        
        # Split into sentences
        sentences = self.split_into_sentences(narrative)
        
        if len(sentences) < 10:
            print(f"Warning: Only {len(sentences)} sentences available")
            return []
        
        # Extract keywords for each sentence
        sentence_keywords = []
        for i, sentence in enumerate(sentences):
            keywords = self.extract_keywords(sentence)
            if keywords:
                sentence_keywords.append({
                    "index": i,
                    "sentence": sentence,
                    "keywords": list(keywords),
                })
        
        if not sentence_keywords:
            print("Warning: No keywords found in narrative")
            return []
        
        # Generate tasks
        tasks = []
        
        # Determine difficulty levels
        difficulties = []
        for difficulty, proportion in difficulty_distribution.items():
            count = int(num_tasks * proportion)
            difficulties.extend([difficulty] * count)
        
        # Fill to exact num_tasks
        while len(difficulties) < num_tasks:
            difficulties.append(random.choice(list(difficulty_distribution.keys())))
        
        random.shuffle(difficulties)
        
        # Generate each task
        for task_idx, difficulty in enumerate(difficulties[:num_tasks]):
            # Select a random sentence with keywords
            sent_data = random.choice(sentence_keywords)
            target_sentence = sent_data["sentence"]
            target_index = sent_data["index"]
            keyword = random.choice(sent_data["keywords"])
            
            # Generate passage based on difficulty
            if difficulty == "easy":
                target_tokens = random.randint(100, 400)  # Short passage
            elif difficulty == "medium":
                target_tokens = random.randint(400, 1000)  # Medium passage
            else:  # hard
                target_tokens = random.randint(1000, 2000)  # Long passage
            
            # Create passage centered around target sentence
            passage_start = max(0, target_index - 5)
            passage_end = min(len(sentences), target_index + 20)
            passage_sentences = sentences[passage_start:passage_end]
            
            # Ensure target sentence is in the passage
            if target_sentence not in passage_sentences:
                passage_sentences.insert(random.randint(0, len(passage_sentences)), target_sentence)
            
            # Trim to target token count
            passage_text = " ".join(passage_sentences)
            actual_tokens = self.estimate_tokens(passage_text)
            
            # Create task
            task = {
                "task_id": f"recall_{task_idx:04d}",
                "difficulty": difficulty,
                "passage": passage_text,
                "keyword": keyword,
                "target_sentence": target_sentence,
                "passage_tokens": actual_tokens,
                "num_sentences": len(passage_sentences),
            }
            
            tasks.append(task)
        
        return tasks
    
    def generate_from_episodic_memory_dataset(
        self,
        dataset: Dict[str, Any],
        num_tasks: int = 500,
    ) -> List[Dict[str, Any]]:
        """
        Generate recall tasks from an episodic memory dataset.
        
        Args:
            dataset: Episodic memory dataset (from EpisodicMemoryDataset.load_dataset)
            num_tasks: Number of tasks to generate
        
        Returns:
            List of recall tasks
        """
        narrative = dataset.get("narrative", "")
        
        if not narrative:
            raise ValueError("No narrative found in dataset")
        
        return self.generate_tasks_from_narrative(narrative, num_tasks)


def generate_recall_tasks(
    episodic_dataset: Dict[str, Any],
    num_tasks: int = 500,
    output_file: str = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate recall tasks.
    
    Args:
        episodic_dataset: Loaded episodic memory dataset
        num_tasks: Number of tasks to generate
        output_file: Optional path to save tasks as JSON
    
    Returns:
        List of recall tasks
    """
    generator = RecallTaskGenerator()
    tasks = generator.generate_from_episodic_memory_dataset(episodic_dataset, num_tasks)
    
    if output_file:
        import json
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        print(f"Saved {len(tasks)} recall tasks to {output_file}")
    
    return tasks
