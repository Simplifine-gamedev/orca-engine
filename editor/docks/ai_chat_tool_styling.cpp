/**************************************************************************/
/*  ai_chat_tool_styling.cpp                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             ORCA ENGINE                                */
/**************************************************************************/

#include "ai_chat_tool_styling.h"

void AIChatToolStyling::style_executing_tool_label(Label *p_label, Control *p_theme_source) {
	if (!p_label || !p_theme_source) return;
	
	// Monochromatic: gray text that blends with background
	// Use lower opacity to make it less prominent
	Color base_color = p_theme_source->get_theme_color(SNAME("font_color"), SNAME("Editor"));
	Color executing_color = base_color * Color(1.0, 1.0, 1.0, 0.5); // 50% opacity - blends into background
	
	p_label->add_theme_color_override("font_color", executing_color);
	p_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	p_label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_LEFT); // Left align
	
	// Make font slightly smaller
	Ref<Font> default_font = p_theme_source->get_theme_font(SNAME("font"), SNAME("Label"));
	int default_size = p_theme_source->get_theme_font_size(SNAME("font_size"), SNAME("Label"));
	p_label->add_theme_font_size_override("font_size", default_size - 1); // 1 size smaller
}

void AIChatToolStyling::style_tool_result_button(Button *p_button, bool p_success, Control *p_theme_source) {
	if (!p_button || !p_theme_source) return;
	
	// Monochromatic: No green/red colors
	// Use gray color that blends with background
	Color base_color = p_theme_source->get_theme_color(SNAME("font_color"), SNAME("Editor"));
	Color tool_color = base_color * Color(1.0, 1.0, 1.0, 0.5); // 50% opacity - blends into background
	
	p_button->add_theme_color_override("font_color", tool_color);
	p_button->set_flat(true); // Flat button - no background
	p_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	p_button->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT); // Left align
	p_button->set_clip_text(true);
	p_button->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	
	// Make font slightly smaller
	int default_size = p_theme_source->get_theme_font_size(SNAME("font_size"), SNAME("Button"));
	p_button->add_theme_font_size_override("font_size", default_size - 1); // 1 size smaller
	
	// Completely transparent style - no borders, no background for all states
	Ref<StyleBoxFlat> tool_button_style = memnew(StyleBoxFlat);
	tool_button_style->set_bg_color(Color(0, 0, 0, 0)); // Transparent background
	tool_button_style->set_border_width_all(0); // No border
	tool_button_style->set_border_color(Color(0, 0, 0, 0)); // Transparent border
	tool_button_style->set_corner_radius_all(0); // No rounded corners
	tool_button_style->set_content_margin_all(0); // No padding for left alignment
	tool_button_style->set_content_margin(SIDE_LEFT, 0); // Ensure left is flush
	p_button->add_theme_style_override("normal", tool_button_style);
	p_button->add_theme_style_override("hover", tool_button_style);
	p_button->add_theme_style_override("pressed", tool_button_style);
	p_button->add_theme_style_override("focus", tool_button_style); // No border when focused/clicked
	p_button->add_theme_style_override("disabled", tool_button_style);
	
	// Override font color for hover/pressed to make text white when clicked
	Color hover_color = Color(1, 1, 1, 1); // White text when hovering/clicking
	p_button->add_theme_color_override("font_hover_color", hover_color);
	p_button->add_theme_color_override("font_pressed_color", hover_color);
	p_button->add_theme_color_override("font_focus_color", hover_color);
}

void AIChatToolStyling::style_tool_placeholder_panel(PanelContainer *p_panel, Control *p_theme_source) {
	if (!p_panel || !p_theme_source) return;
	
	// Completely transparent panel - no background, no borders
	Ref<StyleBoxFlat> placeholder_style = memnew(StyleBoxFlat);
	placeholder_style->set_bg_color(Color(0, 0, 0, 0)); // Transparent background
	placeholder_style->set_content_margin_all(0); // Zero padding for alignment
	placeholder_style->set_border_width_all(0); // No border
	placeholder_style->set_border_color(Color(0, 0, 0, 0)); // Transparent border
	placeholder_style->set_corner_radius_all(0); // No rounded corners
	placeholder_style->set_draw_center(false); // Don't draw background at all
	p_panel->add_theme_style_override("panel", placeholder_style);
}

Color AIChatToolStyling::get_tool_status_color(bool p_success, Control *p_theme_source) {
	if (!p_theme_source) return Color(0.5, 0.5, 0.5, 1.0);
	
	// Monochromatic: Same gray color for both success and failure
	// Use lower opacity to blend with background
	Color base_color = p_theme_source->get_theme_color(SNAME("font_color"), SNAME("Editor"));
	return base_color * Color(1.0, 1.0, 1.0, 0.5);
}

