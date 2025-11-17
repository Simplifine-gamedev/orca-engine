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

Dictionary AIReasoningParser::parse_reasoning_tags(const String &p_text) {
	Dictionary result;
	
	// Find all <reasoning> ... </reasoning> blocks
	String cleaned_text = p_text;
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
	Array blocks;
	
	int current_pos = 0;
	while (current_pos < p_text.length()) {
		int reasoning_start = p_text.find("<reasoning>", current_pos);
		
		// If no more reasoning tags, add remaining text as final block
		if (reasoning_start == -1) {
			String remaining = p_text.substr(current_pos);
			if (!remaining.strip_edges().is_empty()) {
				Dictionary text_block;
				text_block[Variant("type")] = Variant("text");
				text_block[Variant("content")] = Variant(remaining);
				blocks.push_back(text_block);
			}
			break;
		}
		
		// Add text before reasoning tag
		if (reasoning_start > current_pos) {
			String text_before = p_text.substr(current_pos, reasoning_start - current_pos);
			if (!text_before.strip_edges().is_empty()) {
				Dictionary text_block;
				text_block[Variant("type")] = Variant("text");
				text_block[Variant("content")] = Variant(text_before);
				blocks.push_back(text_block);
			}
		}
		
		// Find end of reasoning block
		int content_start = reasoning_start + 11; // Length of "<reasoning>"
		int reasoning_end = p_text.find("</reasoning>", content_start);
		
		if (reasoning_end == -1) {
			// Unclosed tag - treat rest as text
			String remaining = p_text.substr(reasoning_start);
			if (!remaining.strip_edges().is_empty()) {
				Dictionary text_block;
				text_block[Variant("type")] = Variant("text");
				text_block[Variant("content")] = Variant(remaining);
				blocks.push_back(text_block);
			}
			break;
		}
		
		// Add reasoning block
		String reasoning_content = p_text.substr(content_start, reasoning_end - content_start);
		if (!reasoning_content.strip_edges().is_empty()) {
			Dictionary reasoning_block;
			reasoning_block[Variant("type")] = Variant("reasoning");
			reasoning_block[Variant("content")] = Variant(reasoning_content.strip_edges());
			blocks.push_back(reasoning_block);
		}
		
		// Move past this reasoning block
		current_pos = reasoning_end + 12; // Length of "</reasoning>"
	}
	
	return blocks;
}

bool AIReasoningParser::has_reasoning_tags(const String &p_text) {
	return p_text.find("<reasoning>") != -1;
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
	
	// Return the toggle button so caller can connect it if needed
	toggle->set_meta("thinking_label", thinking_label);
	
	return thinking_label;
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
