#!/usr/bin/env python3
"""
Script to create a synthetic dataset for faceting and filtering benchmarks.
This creates data once and stores it for repeated benchmark runs.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm

def create_synthetic_data(output_dir, num_passages=10000):
    """
    Create a synthetic dataset with metadata for faceting and filtering benchmarks.
    
    Args:
        output_dir: Directory to store the dataset
        num_passages: Number of passages to generate
    
    Returns:
        Path to the created collection file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    collection_path = os.path.join(output_dir, "synthetic_collection.jsonl")
    
    # Define metadata fields with different cardinalities
    # Low cardinality field (10 distinct values)
    category_values = ["science", "history", "politics", "art", "technology", 
                     "health", "business", "entertainment", "sports", "education"]
    
    # Medium cardinality field (100 distinct values)
    # Generate 100 different sources
    sources = [f"source_{i}" for i in range(100)]
    
    # High cardinality field (1000 distinct values)
    # Generate 1000 different author names
    authors = [f"author_{i}" for i in range(1000)]
    
    # Generate passage content
    # To make passages more realistic for testing, create them with varied length
    # and some topical words that can be used for searching
    topics = {
        "science": ["research", "experiment", "theory", "discovery", "scientist", "laboratory", 
                  "physics", "chemistry", "biology", "quantum", "molecular", "universe"],
        "history": ["ancient", "medieval", "century", "war", "civilization", "empire", 
                  "revolution", "dynasty", "archaeology", "historical", "era", "heritage"],
        "politics": ["government", "election", "democracy", "parliament", "president", "policy", 
                   "legislation", "campaign", "political", "debate", "vote", "reform"],
        "art": ["painting", "sculpture", "gallery", "exhibition", "artistic", "creative", 
              "canvas", "masterpiece", "portrait", "artist", "museum", "aesthetic"],
        "technology": ["innovation", "digital", "software", "hardware", "device", "internet", 
                     "computer", "algorithm", "engineering", "programming", "startup", "tech"],
        "health": ["medical", "disease", "treatment", "medicine", "doctor", "hospital", 
                 "patient", "therapy", "diagnosis", "wellness", "healthcare", "clinical"],
        "business": ["company", "market", "investment", "corporate", "financial", "economy", 
                   "industry", "startup", "entrepreneur", "profit", "commercial", "economic"],
        "entertainment": ["movie", "music", "concert", "celebrity", "film", "television", 
                        "performance", "actor", "director", "festival", "award", "artist"],
        "sports": ["athlete", "championship", "competition", "tournament", "player", "team", 
                 "stadium", "coach", "record", "medal", "league", "olympic"],
        "education": ["student", "teacher", "school", "university", "academic", "learning", 
                    "curriculum", "education", "college", "classroom", "professor", "study"]
    }
    
    common_words = ["the", "and", "of", "to", "in", "is", "that", "it", "with", "for", "as", 
                   "on", "by", "at", "from", "be", "this", "have", "or", "are", "an", "was"]
    
    print(f"Generating {num_passages} synthetic passages with metadata...")
    
    with open(collection_path, 'w') as f:
        for pid in tqdm(range(num_passages)):
            # Assign metadata with different distributions
            category = random.choice(category_values)
            source = random.choice(sources)
            author = random.choice(authors)
            
            # Year field for range filtering tests (2000-2023)
            year = random.randint(2000, 2023)
            
            # Generate passage text
            # Length between 50 and 200 words
            passage_length = random.randint(50, 200)
            
            # Generate content with topic-specific words
            topic_words = topics[category]
            
            # 70% common words, 30% topical words
            word_choices = []
            for i in range(passage_length):
                if random.random() < 0.7:
                    word_choices.append(random.choice(common_words))
                else:
                    word_choices.append(random.choice(topic_words))
            
            passage = " ".join(word_choices)
            
            # Write to JSONL file
            record = {
                "pid": pid,
                "passage": passage,
                "metadata": {
                    "category": category,
                    "source": source,
                    "author": author,
                    "year": year
                }
            }
            
            f.write(json.dumps(record) + '\n')
    
    # Write a metadata file about the dataset
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        metadata = {
            "num_passages": num_passages,
            "fields": {
                "category": {
                    "type": "categorical",
                    "cardinality": len(category_values),
                    "values": category_values
                },
                "source": {
                    "type": "categorical",
                    "cardinality": len(sources),
                    "values": ["source_0", "source_1", "..."]  # Just show a sample
                },
                "author": {
                    "type": "categorical",
                    "cardinality": len(authors),
                    "values": ["author_0", "author_1", "..."]  # Just show a sample
                },
                "year": {
                    "type": "numeric",
                    "min": 2000,
                    "max": 2023
                }
            }
        }
        json.dump(metadata, f, indent=2)
    
    print(f"Synthetic data created at: {collection_path}")
    print(f"Dataset metadata stored at: {metadata_path}")
    
    return collection_path, metadata_path

def main():
    parser = argparse.ArgumentParser(description="Create synthetic data for faceting benchmarks")
    parser.add_argument("--output-dir", type=str, default="tests/data/faceting_benchmark", 
                        help="Directory to store the dataset")
    parser.add_argument("--num-passages", type=int, default=10000,
                        help="Number of passages to generate")
    args = parser.parse_args()
    
    # Create the data
    collection_path, metadata_path = create_synthetic_data(
        args.output_dir, 
        num_passages=args.num_passages
    )
    
    print("\nTo run benchmarks with this data, use:")
    print(f"python -m pytest tests/test_faceting_benchmark.py --benchmark-data-dir={args.output_dir}")

if __name__ == "__main__":
    main()