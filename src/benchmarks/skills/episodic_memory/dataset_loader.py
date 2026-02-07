"""
Dataset loader for Episodic Memory benchmark.

Downloads and loads the Tulving Episodic Memory dataset from Figshare.
"""

import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

import pandas as pd


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class EpisodicMemoryDataset:
    """Loader for Episodic Memory benchmark dataset."""
    
    # Figshare dataset URL
    FIGSHARE_URL = "https://figshare.com/ndownloader/files/28244480"
    
    # Dataset configurations (maps num_chapters to directory name)
    DATASETS = {
        20: "Udefault_Sdefault_seed0",
        200: "Udefault_Sdefault_seed0",
        2000: "Udefault_Sdefault_seed0",
    }
    
    # Book directory mappings (approximate chapter counts)
    BOOK_DIRS = {
        20: "model_claude-3-5-sonnet-20240620_itermax_10_Idefault_nbchapters_19_nbtokens_10397",
        200: "model_claude-3-5-sonnet-20240620_itermax_10_Idefault_nbchapters_196_nbtokens_102870",
        2000: "model_claude-3-5-sonnet-20240620_itermax_10_Idefault_nbchapters_1967_nbtokens_1033475",
    }
    
    def __init__(self, data_dir: str = "data/episodic_memory", verbose: bool = False):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory to store/load dataset
            verbose: Enable verbose output
        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, force: bool = False) -> Path:
        """
        Download the dataset from Figshare.
        
        Args:
            force: Force re-download even if exists
        
        Returns:
            Path to extracted dataset directory
        """
        zip_path = self.data_dir / "episodic_memory_data.zip"
        extracted_dir = self.data_dir / "episodic-memory-benchmark"
        
        # Download the dataset
        if not zip_path.exists() or force:
            if self.verbose:
                print(f"Downloading dataset from Figshare...")
                print(f"URL: {self.FIGSHARE_URL}")
                print(f"Destination: {zip_path}")
            
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
                urllib.request.urlretrieve(
                    self.FIGSHARE_URL,
                    zip_path,
                    reporthook=t.update_to
                )
            
            if self.verbose:
                print(f"Download complete: {zip_path}")
        
        # Extract the dataset
        if self.verbose:
            print(f"Extracting dataset...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        if self.verbose:
            print(f"Extraction complete: {extracted_dir}")
        
        return extracted_dir
    
    def load_dataset(
        self,
        num_chapters: int = 20,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Load a specific dataset configuration.
        
        Args:
            num_chapters: Number of chapters (20, 200, or 2000)
            force_download: Force re-download of dataset
        
        Returns:
            Dictionary containing:
                - narrative: Full text of all chapters
                - chapters: List of individual chapter texts
                - events: List of event dictionaries
                - qa_pairs: List of question-answer pairs
                - num_tokens: Approximate token count
        """
        if num_chapters not in self.DATASETS:
            raise ValueError(f"Invalid num_chapters: {num_chapters}. Must be one of {list(self.DATASETS.keys())}")
        
        # Check if data is already extracted (skip download)
        dataset_name = self.DATASETS[num_chapters]
        dataset_dir = self.data_dir / dataset_name
        
        if not dataset_dir.exists():
            # Download if needed
            extracted_dir = self.download_dataset(force=force_download)
            dataset_dir = extracted_dir / "data" / dataset_name
        
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_dir}\n"
                f"Please download the dataset from: https://doi.org/10.6084/m9.figshare.28244480\n"
                f"And extract to: {self.data_dir}/"
            )
        
        # Get the book directory for this size
        book_dir_name = self.BOOK_DIRS.get(num_chapters)
        if not book_dir_name:
            raise ValueError(f"No book directory mapping for {num_chapters} chapters")
        
        book_dir = dataset_dir / "books" / book_dir_name
        
        if not book_dir.exists():
            raise FileNotFoundError(f"Book directory not found: {book_dir}")
        
        if self.verbose:
            print(f"Loading dataset from: {book_dir}")
        
        # Load the book.json file
        book_file = book_dir / "book.json"
        if not book_file.exists():
            raise FileNotFoundError(f"book.json not found in {book_dir}")
        
        with open(book_file, 'r', encoding='utf-8') as f:
            book_data = json.load(f)
        
        # The book.json is a string containing the full narrative
        if isinstance(book_data, str):
            narrative = book_data
            chapters = narrative.split("\n\n")  # Simple split by double newline
        else:
            raise ValueError(f"Unexpected book.json format: {type(book_data)}")
        
        # Load events from the dataset root
        events_file = dataset_dir / "events.json"
        events = []
        if events_file.exists():
            with open(events_file, 'r', encoding='utf-8') as f:
                events = json.load(f)

        # Load QA pairs from parquet (Figshare format)
        qa_pairs: List[Dict[str, Any]] = []
        qa_file = book_dir / "df_qa.parquet"
        if qa_file.exists():
            df = pd.read_parquet(qa_file)

            def map_question_type(retrieval_type: str, get_mode: str) -> str:
                rt = (retrieval_type or "").strip().lower()
                gm = (get_mode or "").strip().lower()

                if gm == "chronological":
                    if rt == "times":
                        return "time_chronological"
                    if rt == "spaces":
                        return "space_chronological"
                    if rt == "event contents":
                        return "content_chronological"
                    return "chronological_list"

                if gm == "latest":
                    if rt == "times":
                        return "time_latest"
                    if rt == "spaces":
                        return "space_latest"
                    return "latest"

                # Default bucket: simple recall / attribute retrieval
                return "simple_recall"

            for i, row in df.iterrows():
                question = row.get("question", "")
                retrieval_type = row.get("retrieval_type", "")
                get_mode = row.get("get", "all")

                correct = row.get("correct_answer", [])
                # correct_answer is typically a numpy array of strings
                if hasattr(correct, "tolist"):
                    correct_list = correct.tolist()
                elif isinstance(correct, (list, tuple)):
                    correct_list = list(correct)
                elif correct is None:
                    correct_list = []
                else:
                    correct_list = [correct]

                ground_truth = [str(x) for x in correct_list if x is not None and str(x).strip() != ""]

                qa_pairs.append(
                    {
                        "id": f"q_{i}",
                        "question": str(question),
                        "type": map_question_type(str(retrieval_type), str(get_mode)),
                        "ground_truth": ground_truth,
                        # helpful metadata
                        "retrieval_type": str(retrieval_type),
                        "get": str(get_mode),
                        "cue": str(row.get("cue", "")),
                        "cue_completed": str(row.get("cue_completed", "")),
                        "n_items_correct_answer": int(row.get("n_items_correct_answer", 0) or 0),
                    }
                )
        
        # Estimate token count (rough: 1 token ≈ 4 characters)
        num_tokens = len(narrative) // 4
        
        result = {
            "narrative": narrative,
            "chapters": chapters,
            "events": events,
            "qa_pairs": qa_pairs,
            "num_chapters": len(chapters),
            "num_chapters_requested": num_chapters,
            "num_tokens": num_tokens,
            "dataset_path": str(book_dir),
        }
        
        if self.verbose:
            print(f"Loaded dataset:")
            print(f"  Narrative length: {len(narrative):,} characters")
            print(f"  Number of chapters: {len(chapters)}")
            print(f"  Number of events: {len(events)}")
            print(f"  Number of QA pairs: {len(qa_pairs)}")
            print(f"  Estimated tokens: {num_tokens:,}")
        
        return result
