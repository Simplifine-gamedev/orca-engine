#!/usr/bin/env python3
"""
Test what search results ACTUALLY contain
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import search_across_project_internal
import json

def test_search_enrichment():
    """Test what's actually in enriched search results"""
    
    user_id = "069ae9ad-bae1-4e0f-a74b-5d2b4a770b9d"
    project_id = "5695ea631a3d5da28464d01705efc939"
    project_root = "/Users/alikavoosi/Desktop/3d-design/GODOT/godot/backend/dodge_the_creeps"
    
    print("=" * 80)
    print("🔍 TESTING SEARCH RESULT ENRICHMENT")
    print("=" * 80)
    
    # Search for player damage/hit
    result = search_across_project_internal({
        'query': 'player damage hit collision',
        'max_results': 3,
        'include_graph': True,
        'project_root': project_root,
        'project_id': project_id,
        'search_mode': 'semantic'
    }, {'id': user_id})
    
    print(f"\n✅ Search Success: {result.get('success')}")
    print(f"📊 Files Found: {result.get('file_count', 0)}")
    
    if result.get('success') and result.get('results'):
        results_data = result['results']
        similar_files = results_data.get('similar_files', [])
        
        print(f"\n📁 Analyzing first result:")
        if similar_files:
            first = similar_files[0]
            print(f"   File: {first.get('file_path')}")
            print(f"   Similarity: {first.get('similarity', 0):.3f}")
            
            # CHECK: What context is included?
            print(f"\n📋 CONTEXT KEYS: {list(first.get('context', {}).keys())}")
            context = first.get('context', {})
            if context:
                print(f"   Functions: {len(context.get('functions', []))}")
                print(f"   Signals defined: {len(context.get('signals_defined', []))}")
                print(f"   Signals emitted: {len(context.get('signals_emitted', []))}")
                print(f"   Method calls: {len(context.get('method_calls', []))}")
                print(f"   Node accesses: {len(context.get('node_accesses', []))}")
                print(f"   Exports: {len(context.get('exports', []))}")
                print(f"   Groups: {len(context.get('groups', []))}")
                
                # Show actual signals_emitted data
                if context.get('signals_emitted'):
                    print(f"\n🔥 SIGNALS EMITTED (actual data):")
                    for sig in context.get('signals_emitted', [])[:3]:
                        print(f"      {sig}")
            
            # CHECK: What relationships are included?
            print(f"\n🔗 RELATIONSHIPS KEYS: {list(first.get('relationships', {}).keys())}")
            relationships = first.get('relationships', {})
            if relationships:
                print(f"   Signals emitted to: {len(relationships.get('signals_emitted_to', []))}")
                print(f"   Signals received from: {len(relationships.get('signals_received_from', []))}")
                print(f"   Scripts attached to: {len(relationships.get('attached_to_scenes', []))}")
                print(f"   Resources used: {len(relationships.get('resources_used', []))}")
                print(f"   Method calls to: {len(relationships.get('method_calls_to', []))}")
                
                # CHECK: Signal propagation tree
                if 'signal_propagation_tree' in relationships:
                    tree = relationships['signal_propagation_tree']
                    print(f"\n🌲 SIGNAL PROPAGATION TREE PRESENT!")
                    print(f"   Cascades: {len(tree.get('cascades', []))}")
                    print(f"   Max depth: {tree.get('total_depth', 0)}")
                else:
                    print(f"\n❌ signal_propagation_tree NOT in relationships")
                
                # CHECK: Scene composition
                if 'scene_composition' in relationships:
                    comp = relationships['scene_composition']
                    print(f"\n🏗️ SCENE COMPOSITION PRESENT!")
                    print(f"   Instantiates: {comp.get('instantiates', [])}")
                else:
                    print(f"\n❌ scene_composition NOT in relationships")
                
                # CHECK: Group interactions
                if 'group_interactions' in relationships:
                    groups = relationships['group_interactions']
                    print(f"\n👥 GROUP INTERACTIONS PRESENT!")
                    print(f"   Belongs to: {len(groups.get('belongs_to_groups', []))}")
                    print(f"   Calls on: {len(groups.get('calls_on_groups', []))}")
                else:
                    print(f"\n❌ group_interactions NOT in relationships")
            
            # CHECK: Usage summary
            print(f"\n📊 USAGE SUMMARY KEYS: {list(first.get('usage_summary', {}).keys())}")
            usage = first.get('usage_summary', {})
            if usage:
                print(f"   Architectural role: {usage.get('architectural_role')}")
                print(f"   Coupling score: {usage.get('coupling_score', 0):.2f}")
                print(f"   Is hub: {usage.get('is_hub')}")
        
        # Show ALL keys in the result to see what's available
        print(f"\n🔑 ALL TOP-LEVEL KEYS IN RESULT:")
        print(f"   {list(first.keys())}")
    
    print("\n" + "=" * 80)
    print("✅ ENRICHMENT ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_search_enrichment()


