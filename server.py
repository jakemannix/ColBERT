from flask import Flask, render_template, request, jsonify
from functools import lru_cache
import math
import json
import os
from dotenv import load_dotenv

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

load_dotenv()

INDEX_NAME = os.getenv("INDEX_NAME")
INDEX_ROOT = os.getenv("INDEX_ROOT")
app = Flask(__name__)

searcher = Searcher(index=INDEX_NAME, index_root=INDEX_ROOT)
counter = {"api" : 0}

@lru_cache(maxsize=1000000)
def api_search_query(query, k, facet_fields=None, facet_filters=None):
    """
    Search API with support for faceting.
    
    Args:
        query: The search query
        k: Number of results to return
        facet_fields: Comma-separated list of fields to compute facets for
        facet_filters: JSON string of field->value filters
    """
    print(f"Query={query}, Facets={facet_fields}, Filters={facet_filters}")
    
    # Process parameters
    if k is None: 
        k = 10
    k = min(int(k), 100)
    
    # Process facet fields
    if facet_fields:
        facet_fields = [field.strip() for field in facet_fields.split(',')]
    
    # Process facet filters
    filter_dict = None
    if facet_filters:
        try:
            filter_dict = json.loads(facet_filters)
        except json.JSONDecodeError:
            print(f"Invalid facet filter format: {facet_filters}")
    
    # Perform search with or without faceting
    if facet_fields:
        # Faceted search
        pids, ranks, scores, facets = searcher.search(
            query, 
            k=100, 
            facet_fields=facet_fields,
            facet_filters=filter_dict
        )
        pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    else:
        # Regular search without faceting
        pids, ranks, scores = searcher.search(
            query, 
            k=100,
            facet_filters=filter_dict
        )
        pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    
    # Process results
    passages = [searcher.collection[pid] for pid in pids]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs] if probs else []
    
    # Collect document data with metadata
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]
        d = {
            'text': text, 
            'pid': pid, 
            'rank': rank, 
            'score': score, 
            'prob': prob
        }
        
        # Add metadata if available
        metadata = searcher.collection.get_metadata(pid)
        if metadata:
            d['metadata'] = metadata
            
        topk.append(d)
    
    # Sort by score and create result
    topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
    result = {"query": query, "topk": topk}
    
    # Add facets if requested
    if facet_fields:
        result["facets"] = facets
        
    return result

@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        print("API request count:", counter["api"])
        return api_search_query(
            request.args.get("query"), 
            request.args.get("k"),
            request.args.get("facet_fields"),
            request.args.get("facet_filters")
        )
    else:
        return ('', 405)
        
@app.route("/api/facets", methods=["GET"])
def api_get_facets():
    """
    Get available facet fields (metadata schema).
    This is a simple API endpoint to discover what facets are available in the collection.
    """
    # Sample a few documents to discover metadata fields
    facet_fields = {}
    sample_size = min(100, len(searcher.collection))
    
    for pid in range(sample_size):
        metadata = searcher.collection.get_metadata(pid)
        for field, value in metadata.items():
            if field not in facet_fields:
                facet_fields[field] = {
                    "type": type(value).__name__,
                    "example": value
                }
    
    return jsonify({
        "facet_fields": facet_fields
    })

if __name__ == "__main__":
    app.run("0.0.0.0", int(os.getenv("PORT")))

