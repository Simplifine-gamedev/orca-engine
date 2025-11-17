/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#ifndef AI_REASONING_PARSER_H
#define AI_REASONING_PARSER_H

#include "core/string/ustring.h"
#include "core/variant/dictionary.h"
#include "core/variant/array.h"

class Control;
class VBoxContainer;
class RichTextLabel;
class PanelContainer;
class Button;

class AIReasoningParser {
public:
	// Extract <reasoning> tags from text and return cleaned text + reasoning content
	static Dictionary parse_reasoning_tags(const String &p_text);
	
	// Parse into interleaved blocks preserving order: [{type: "text", content: "..."}, {type: "reasoning", content: "..."}]
	static Array parse_interleaved_blocks(const String &p_text);
	
	// Check if text contains <reasoning> tags
	static bool has_reasoning_tags(const String &p_text);
	
	// Create a thinking block UI element and add it to the container
	// Returns the created thinking label for updating content later
	// p_bbcode_content: Pre-converted BBCode content (if empty, uses raw content from block)
	static RichTextLabel *create_thinking_block_ui(VBoxContainer *p_container, const Dictionary &p_block, Control *p_theme_source, const String &p_bbcode_content = "");
	
	// Create a text block UI element and add it to the container
	// p_bbcode_content: Pre-converted BBCode content (if empty, uses raw p_content)
	static RichTextLabel *create_text_block_ui(VBoxContainer *p_container, const String &p_content, Control *p_theme_source, const String &p_bbcode_content = "");
	
	// Incrementally add new blocks to an existing reasoning container (for streaming)
	// Returns the number of new blocks added
	static int add_blocks_incremental(VBoxContainer *p_container, const Array &p_new_blocks, Control *p_theme_source, const String &p_markdown_to_bbcode_func_name = "");
};

#endif // AI_REASONING_PARSER_H

