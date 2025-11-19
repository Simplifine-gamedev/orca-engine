/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_reasoning_parser.h"
#include "core/variant/variant.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/style_box_flat.h"
#include "editor/themes/editor_theme_manager.h"
#include "core/object/callable_method_pointer.h"

static String _decode_reasoning_text(const String &p_text) {
	String decoded = p_text;
	// Decode common HTML entities that may be present when tags are escaped.
	decoded = decoded.replace("&amp;", "&");
	decoded = decoded.replace("&lt;", "<");
	decoded = decoded.replace("&gt;", ">");
	decoded = decoded.replace("&quot;", "\"");
	decoded = decoded.replace("&#39;", "'");
	return decoded;
}

void AIReasoningParser::_toggle_panel_visibility(Control *p_panel) {
	if (!p_panel) {
		return;
	}
	p_panel->set_visible(!p_panel->is_visible());
}

Dictionary AIReasoningParser::parse_reasoning_tags(const String &p_text) {
	String decoded_text = _decode_reasoning_text(p_text);
	Dictionary result;
	
	// Find all <reasoning> ... </reasoning> blocks
	String cleaned_text = decoded_text;
	String reasoning_content = "";
	
	int search_pos = 0;
	while (true) {
		int start_tag = cleaned_text.find("<reasoning>", search_pos);
		if (start_tag == -1) {
			break;
		}
		
		int content_start = start_tag + 11;
		int end_tag = cleaned_text.find("</reasoning>", content_start);
		
		if (end_tag == -1) {
			search_pos = start_tag + 1;
			continue;
		}
		
		String block_content = cleaned_text.substr(content_start, end_tag - content_start);
		if (!reasoning_content.is_empty()) {
			reasoning_content += "\n\n---\n\n";
		}
		reasoning_content += block_content.strip_edges();
		
		String before = cleaned_text.substr(0, start_tag);
		String after = cleaned_text.substr(end_tag + 12);
		cleaned_text = before + after;
		
		search_pos = start_tag;
	}
	
	result[Variant("text")] = Variant(cleaned_text.strip_edges());
	result[Variant("reasoning")] = Variant(reasoning_content);
	result[Variant("has_reasoning")] = Variant(!reasoning_content.is_empty());
	
	return result;
}

Array AIReasoningParser::parse_interleaved_blocks(const String &p_text) {
	String decoded_text = _decode_reasoning_text(p_text);
	Array blocks;
	
	int current_pos = 0;
	while (current_pos < decoded_text.length()) {
		// Find the next tag (reasoning or planning), whichever comes first
		int reasoning_start = decoded_text.find("<reasoning>", current_pos);
		int planning_start = decoded_text.find("<planning>", current_pos);
		
		// Determine which tag comes first (or if neither exists)
		int next_tag_start = -1;
		String tag_type;
		String open_tag;
		String close_tag;
		int tag_open_length;
		int tag_close_length;
		
		if (reasoning_start != -1 && planning_start != -1) {
			// Both exist - use whichever comes first
			if (reasoning_start < planning_start) {
				next_tag_start = reasoning_start;
				tag_type = "reasoning";
				open_tag = "<reasoning>";
				close_tag = "</reasoning>";
				tag_open_length = 11;
				tag_close_length = 12;
			} else {
				next_tag_start = planning_start;
				tag_type = "planning";
				open_tag = "<planning>";
				close_tag = "</planning>";
				tag_open_length = 10;
				tag_close_length = 11;
			}
		} else if (reasoning_start != -1) {
			next_tag_start = reasoning_start;
			tag_type = "reasoning";
			open_tag = "<reasoning>";
			close_tag = "</reasoning>";
			tag_open_length = 11;
			tag_close_length = 12;
		} else if (planning_start != -1) {
			next_tag_start = planning_start;
			tag_type = "planning";
			open_tag = "<planning>";
			close_tag = "</planning>";
			tag_open_length = 10;
			tag_close_length = 11;
		}
		
		// If no more tags, add remaining text as final block
		if (next_tag_start == -1) {
			String remaining = decoded_text.substr(current_pos);
			if (!remaining.strip_edges().is_empty()) {
				Dictionary text_block;
				text_block[Variant("type")] = Variant("text");
				text_block[Variant("content")] = Variant(remaining);
				blocks.push_back(text_block);
			}
			break;
		}
		
		// Add text before the tag
		if (next_tag_start > current_pos) {
			String text_before = decoded_text.substr(current_pos, next_tag_start - current_pos);
			if (!text_before.strip_edges().is_empty()) {
				Dictionary text_block;
				text_block[Variant("type")] = Variant("text");
				text_block[Variant("content")] = Variant(text_before);
				blocks.push_back(text_block);
			}
		}
		
		// Find end of the tag block
		int content_start = next_tag_start + tag_open_length;
		int tag_end = decoded_text.find(close_tag, content_start);
		
		if (tag_end == -1) {
			// Unclosed tag - during streaming, create the block anyway with incomplete content
			// This allows real-time rendering of planning/reasoning blocks as they stream in
			String incomplete_content = decoded_text.substr(content_start);
			if (!incomplete_content.strip_edges().is_empty() || decoded_text.length() > content_start) {
				// Create block with incomplete content - it will be updated as more content arrives
				Dictionary tag_block;
				tag_block[Variant("type")] = Variant(tag_type);
				tag_block[Variant("content")] = Variant(incomplete_content);
				blocks.push_back(tag_block);
			}
			// Don't break - check if there's more content after this incomplete tag
			// But for now, we've handled this tag, so move past it
			current_pos = decoded_text.length(); // Move to end to process remaining text
			break;
		}
		
		// Add the tag block (reasoning or planning)
		String tag_content = decoded_text.substr(content_start, tag_end - content_start);
		if (!tag_content.strip_edges().is_empty()) {
			Dictionary tag_block;
			tag_block[Variant("type")] = Variant(tag_type);
			tag_block[Variant("content")] = Variant(tag_content.strip_edges());
			blocks.push_back(tag_block);
		}
		
		// Move past this tag block
		current_pos = tag_end + tag_close_length;
	}
	
	return blocks;
}

bool AIReasoningParser::has_reasoning_tags(const String &p_text) {
	String decoded_text = _decode_reasoning_text(p_text);
	return decoded_text.find("<reasoning>") != -1;
}

bool AIReasoningParser::has_planning_tags(const String &p_text) {
	String decoded_text = _decode_reasoning_text(p_text);
	return decoded_text.find("<planning>") != -1;
}

RichTextLabel *AIReasoningParser::create_thinking_block_ui(VBoxContainer *p_container, const Dictionary &p_block, Control *p_theme_source, const String &p_bbcode_content) {
	if (!p_container || !p_theme_source) {
		return nullptr;
	}
	
	String content = p_block.get("content", p_block.get("thinking", ""));
	if (content.is_empty() && p_bbcode_content.is_empty()) {
		return nullptr;
	}
	
	// Create thinking block UI
	VBoxContainer *thinking_box = memnew(VBoxContainer);
	thinking_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	p_container->add_child(thinking_box);
	
	Button *toggle = memnew(Button);
	toggle->set_text("Thinking");
	toggle->set_flat(false);
	toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	toggle->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	toggle->add_theme_color_override("font_color", p_theme_source->get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	thinking_box->add_child(toggle);
	
	PanelContainer *content_panel = memnew(PanelContainer);
	content_panel->set_visible(true); // Show by default during streaming
	Ref<StyleBoxFlat> style = memnew(StyleBoxFlat);
	style->set_bg_color(p_theme_source->get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
	style->set_border_width_all(1);
	style->set_border_color(p_theme_source->get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
	style->set_content_margin_all(8);
	style->set_corner_radius_all(4);
	content_panel->add_theme_style_override("panel", style);
	thinking_box->add_child(content_panel);
	
	RichTextLabel *thinking_label = memnew(RichTextLabel);
	thinking_label->set_use_bbcode(true);
	thinking_label->set_fit_content(true);
	thinking_label->set_selection_enabled(true);
	thinking_label->set_context_menu_enabled(true);
	thinking_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	thinking_label->add_theme_color_override("default_color", p_theme_source->get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.9));
	thinking_label->set_text(p_bbcode_content.is_empty() ? content : p_bbcode_content);
	content_panel->add_child(thinking_label);
	
	// Store the content panel as metadata so toggle can work
	toggle->set_meta("content_panel", content_panel);
	
	// Connect toggle button to show/hide content panel
	toggle->connect("pressed", callable_mp_static(&AIReasoningParser::_toggle_panel_visibility).bind(content_panel));
	
	// Return the toggle button so caller can connect it if needed
	toggle->set_meta("thinking_label", thinking_label);
	
	return thinking_label;
}

RichTextLabel *AIReasoningParser::create_planning_block_ui(VBoxContainer *p_container, const Dictionary &p_block, Control *p_theme_source, const String &p_bbcode_content) {
	if (!p_container || !p_theme_source) {
		return nullptr;
	}
	
	String content = p_block.get("content", "");
	if (content.is_empty() && p_bbcode_content.is_empty()) {
		return nullptr;
	}
	
	// Create planning block UI (similar to thinking block but with "Planning" label)
	VBoxContainer *planning_box = memnew(VBoxContainer);
	planning_box->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	p_container->add_child(planning_box);
	
	Button *toggle = memnew(Button);
	toggle->set_text("Planning");
	toggle->set_flat(false);
	toggle->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	toggle->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	// Use a slightly different color for planning (e.g., a greenish accent)
	Color planning_color = p_theme_source->get_theme_color(SNAME("accent_color"), SNAME("Editor"));
	// Make it slightly more green/yellow for planning
	planning_color = Color(planning_color.r * 0.8, planning_color.g * 1.2, planning_color.b * 0.8, planning_color.a);
	toggle->add_theme_color_override("font_color", planning_color);
	planning_box->add_child(toggle);
	
	PanelContainer *content_panel = memnew(PanelContainer);
	content_panel->set_visible(true); // Show by default during streaming
	Ref<StyleBoxFlat> style = memnew(StyleBoxFlat);
	style->set_bg_color(p_theme_source->get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
	style->set_border_width_all(1);
	style->set_border_color(p_theme_source->get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
	style->set_content_margin_all(8);
	style->set_corner_radius_all(4);
	content_panel->add_theme_style_override("panel", style);
	planning_box->add_child(content_panel);
	
	RichTextLabel *planning_label = memnew(RichTextLabel);
	planning_label->set_use_bbcode(true);
	planning_label->set_fit_content(true);
	planning_label->set_selection_enabled(true);
	planning_label->set_context_menu_enabled(true);
	planning_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	planning_label->add_theme_color_override("default_color", p_theme_source->get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.9));
	planning_label->set_text(p_bbcode_content.is_empty() ? content : p_bbcode_content);
	content_panel->add_child(planning_label);
	
	// Store the content panel as metadata so toggle can work
	toggle->set_meta("content_panel", content_panel);
	
	// Connect toggle button to show/hide content panel
	toggle->connect("pressed", callable_mp_static(&AIReasoningParser::_toggle_panel_visibility).bind(content_panel));
	
	// Return the planning label so caller can update content later
	toggle->set_meta("planning_label", planning_label);
	
	return planning_label;
}

RichTextLabel *AIReasoningParser::create_text_block_ui(VBoxContainer *p_container, const String &p_content, Control *p_theme_source, const String &p_bbcode_content) {
	if (!p_container || !p_theme_source) {
		return nullptr;
	}
	
	if (p_content.is_empty() && p_bbcode_content.is_empty()) {
		return nullptr;
	}
	
	RichTextLabel *text_label = memnew(RichTextLabel);
	text_label->set_use_bbcode(true);
	text_label->set_fit_content(true);
	text_label->set_selection_enabled(true);
	text_label->set_context_menu_enabled(true);
	text_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	text_label->add_theme_color_override("default_color", p_theme_source->get_theme_color(SNAME("font_color"), SNAME("Editor")));
	text_label->set_text(p_bbcode_content.is_empty() ? p_content : p_bbcode_content);
	p_container->add_child(text_label);
	
	return text_label;
}
