#!/usr/bin/env python3
"""
Enhanced Godot Documentation Search for AI Agent
Provides multiple search modes and intelligent filtering
"""

import weaviate
import weaviate.classes as wvc
import os
from typing import List, Dict, Optional

class EnhancedGodotDocsSearch:
    def __init__(self, weaviate_url: str, weaviate_api_key: str):
        # Ensure OpenAI API key is available for Weaviate
        openai_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_APIKEY')
        if openai_key:
            os.environ['OPENAI_APIKEY'] = openai_key
            os.environ['OPENAI_API_KEY'] = openai_key
        
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
            headers={'X-OpenAI-Api-Key': openai_key} if openai_key else {},
            skip_init_checks=True
        )
    
    def search_godot_docs_enhanced(self, query: str, mode: str = "hybrid", 
                                 section_filter: str = None, class_filter: str = None,
                                 difficulty: str = None, code_examples_only: bool = False,
                                 max_results: int = 5) -> Dict:
        """
        Enhanced documentation search with multiple modes and intelligent filtering
        """
        try:
            collection = self.client.collections.get("GodotDocs")
            
            # Auto-detect best search mode
            if mode == "auto":
                mode = self._detect_search_mode(query)
            
            # Build filters
            where_conditions = []
            
            if section_filter:
                where_conditions.append(
                    wvc.query.Filter.by_property("section").equal(section_filter)
                )
            
            if class_filter:
                where_conditions.append(
                    wvc.query.Filter.by_property("class_name").equal(class_filter)
                )
            
            if difficulty:
                where_conditions.append(
                    wvc.query.Filter.by_property("difficulty").equal(difficulty)
                )
            
            if code_examples_only:
                where_conditions.append(
                    wvc.query.Filter.by_property("code_examples").equal(True)
                )
            
            # Combine filters
            where_filter = None
            if where_conditions:
                if len(where_conditions) == 1:
                    where_filter = where_conditions[0]
                else:
                    where_filter = wvc.query.Filter.all_of(where_conditions)
            
            # Execute search based on mode (apply filters post-query due to API limitations)
            if mode == "semantic":
                results = collection.query.near_text(
                    query=query,
                    limit=max_results * 2,  # Get more to filter
                    return_metadata=["score"]
                )
            elif mode == "keyword":
                results = collection.query.bm25(
                    query=query,
                    limit=max_results * 2,  # Get more to filter
                    return_metadata=["score"]
                )
            else:  # hybrid - but skip if no objects
                # Check if we have objects first
                count_response = collection.aggregate.over_all(total_count=True)
                if count_response.total_count == 0:
                    # Fall back to keyword search if no embeddings
                    results = collection.query.bm25(
                        query=query,
                        limit=max_results,
                        where=where_filter,
                        return_metadata=["score"]
                    )
                else:
                    # Hybrid search doesn't support where filters in this version
                    results = collection.query.hybrid(
                        query=query,
                        limit=max_results,
                        return_metadata=["score"]
                    )
                    # Apply filters post-query if needed
                    if where_filter:
                        filtered_objects = []
                        for obj in results.objects:
                            props = obj.properties
                            include = True
                            if section_filter and props.get('section') != section_filter:
                                include = False
                            if class_filter and props.get('class_name') != class_filter:
                                include = False
                            if difficulty and props.get('difficulty') != difficulty:
                                include = False
                            if code_examples_only and not props.get('code_examples'):
                                include = False
                            if include:
                                filtered_objects.append(obj)
                        results.objects = filtered_objects[:max_results]
            
            # Apply filters to all search modes if needed
            if where_filter and mode in ["semantic", "keyword"]:
                filtered_objects = []
                for obj in results.objects:
                    props = obj.properties
                    include = True
                    if section_filter and props.get('section') != section_filter:
                        include = False
                    if class_filter and props.get('class_name') != class_filter:
                        include = False
                    if difficulty and props.get('difficulty') != difficulty:
                        include = False
                    if code_examples_only and not props.get('code_examples'):
                        include = False
                    if include:
                        filtered_objects.append(obj)
                results.objects = filtered_objects[:max_results]
            
            # Format results with intelligent ranking
            formatted_results = []
            for obj in results.objects:
                props = obj.properties
                score = obj.metadata.score if obj.metadata else 0.0
                
                # Use original score without hardcoded boosts
                enhanced_score = score
                
                formatted_results.append({
                    'title': props.get('title', ''),
                    'content': props.get('content', ''),
                    'snippet': props.get('content', '')[:300] + '...' if len(props.get('content', '')) > 300 else props.get('content', ''),
                    'doc_type': props.get('doc_type', ''),
                    'section': props.get('section', ''),
                    'class_name': props.get('class_name', ''),
                    'url': props.get('url', ''),
                    'similarity': enhanced_score,
                    'original_score': score,
                    'boost_applied': 0.0,
                    'keywords': props.get('keywords', []),
                    'code_examples': props.get('code_examples', False),
                    'difficulty': props.get('difficulty', 'intermediate')
                })
            
            # Sort by enhanced score
            formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                'success': True,
                'query': query,
                'search_mode': mode,
                'results': formatted_results,
                'total_found': len(formatted_results),
                'filters_applied': {
                    'section': section_filter,
                    'class': class_filter,
                    'difficulty': difficulty,
                    'code_examples_only': code_examples_only
                }
            }
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query
            }
    
    def _detect_search_mode(self, query: str) -> str:
        """Auto-detect best search mode based on query characteristics"""
        query_lower = query.lower()
        
        # Keyword indicators
        if any(indicator in query_lower for indicator in [
            'function', 'method', 'property', 'signal', 'class',
            'extends', 'var', 'func', '@export'
        ]):
            return "keyword"
        
        # Hybrid indicators (mix of concepts and specific terms)
        if any(concept in query_lower for concept in [
            'fps', 'platformer', 'movement', 'controller', 'setup', 'tutorial'
        ]) and any(term in query_lower for term in [
            'characterbody', 'camera', 'raycast', 'input', 'mouse'
        ]):
            return "hybrid"
        
        # Default to semantic for natural language queries
        return "semantic"
    
    def get_class_documentation(self, class_name: str) -> Dict:
        """Get complete documentation for a specific class"""
        try:
            collection = self.client.collections.get("GodotDocs")
            
            # Use BM25 search instead of fetch_objects with where filter
            results = collection.query.bm25(
                query=class_name,
                limit=50
            )
            
            # Filter results to only include the specific class
            filtered_objects = []
            for obj in results.objects:
                if obj.properties.get('class_name') == class_name:
                    filtered_objects.append(obj)
            results.objects = filtered_objects
            
            # Group by section
            sections = {
                'overview': [],
                'methods': [],
                'properties': [],
                'signals': []
            }
            
            for obj in results.objects:
                props = obj.properties
                section = props.get('section', 'overview')
                if section in sections:
                    sections[section].append(props)
            
            return {
                'success': True,
                'class_name': class_name,
                'sections': sections,
                'total_entries': sum(len(section) for section in sections.values())
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'class_name': class_name
            }
