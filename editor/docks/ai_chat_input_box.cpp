/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_input_box.h"

#include "ai_chat_dock.h"
#include "ai_chat_mode_selector.h"
#include "core/io/image.h"
#include "scene/gui/button.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/text_edit.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/style_box.h"
#include "scene/resources/style_box_flat.h"
#include "editor/settings/editor_settings.h"

void AIChatInputBox::create_input_ui(AIChatDock *p_chat_dock, VBoxContainer *p_parent_container) {
	// Wrapper for input panel (adds margin from edges)
	MarginContainer *input_wrapper = memnew(MarginContainer);
	input_wrapper->add_theme_constant_override("margin_left", 8);
	input_wrapper->add_theme_constant_override("margin_right", 8);
	input_wrapper->add_theme_constant_override("margin_top", 8);
	input_wrapper->add_theme_constant_override("margin_bottom", 8);
	p_parent_container->add_child(input_wrapper);

	// Panel container for the input field + buttons
	PanelContainer *input_panel = memnew(PanelContainer);
	input_wrapper->add_child(input_panel);
	
	// Border style for the chatbox - subtle border when not focused
	Ref<StyleBoxFlat> panel_style = memnew(StyleBoxFlat);
	panel_style->set_bg_color(p_chat_dock->get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
	panel_style->set_border_width_all(1);
	panel_style->set_border_color(p_chat_dock->get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.3)); // More subtle border
	panel_style->set_corner_radius_all(8);
	input_panel->add_theme_style_override("panel", panel_style);

	// Container inside the panel for all content
	VBoxContainer *input_content = memnew(VBoxContainer);
	input_content->add_theme_constant_override("separation", 0); // No spacing between elements
	input_panel->add_child(input_content);

	// --- TOP TOOLBAR (@ Attachment button) ---
	MarginContainer *top_toolbar_margin = memnew(MarginContainer);
	top_toolbar_margin->add_theme_constant_override("margin_left", 8);
	top_toolbar_margin->add_theme_constant_override("margin_right", 8);
	top_toolbar_margin->add_theme_constant_override("margin_top", 8);
	top_toolbar_margin->add_theme_constant_override("margin_bottom", 4);
	input_content->add_child(top_toolbar_margin);

	HBoxContainer *top_toolbar = memnew(HBoxContainer);
	top_toolbar->add_theme_constant_override("separation", 4);
	top_toolbar_margin->add_child(top_toolbar);

	// Attachment button (MenuButton) - displays @ symbol for Cursor-like appearance
	p_chat_dock->attach_button = memnew(MenuButton);
	p_chat_dock->attach_button->set_text("@"); // Display @ symbol
	p_chat_dock->attach_button->set_tooltip_text("Attach files, nodes, scripts, or resources");
	p_chat_dock->attach_button->set_custom_minimum_size(Size2(32, 28)); // Compact square button
	
	// Set up the popup menu
	PopupMenu *popup = p_chat_dock->attach_button->get_popup();
	popup->add_item("Files", 0);
	popup->set_item_icon(0, p_chat_dock->get_theme_icon(SNAME("FileList"), SNAME("EditorIcons")));
	popup->add_item("Scene Nodes", 1);
	popup->set_item_icon(1, p_chat_dock->get_theme_icon(SNAME("SceneTree"), SNAME("EditorIcons")));
	popup->add_item("Current Script", 2);
	popup->set_item_icon(2, p_chat_dock->get_theme_icon(SNAME("Script"), SNAME("EditorIcons")));
	popup->add_item("Resources", 3);
	popup->set_item_icon(3, p_chat_dock->get_theme_icon(SNAME("ResourcePreloader"), SNAME("EditorIcons")));
	popup->add_separator();
	popup->add_item("Re-index Project", 4);
	popup->set_item_icon(4, p_chat_dock->get_theme_icon(SNAME("Reload"), SNAME("EditorIcons")));
	popup->connect("id_pressed", callable_mp(p_chat_dock, &AIChatDock::_on_attachment_menu_item_pressed));
	
	top_toolbar->add_child(p_chat_dock->attach_button);

	// Spacer to push buttons to left
	Control *top_spacer = memnew(Control);
	top_spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	top_toolbar->add_child(top_spacer);

	// --- TEXT INPUT FIELD ---
	p_chat_dock->input_field = memnew(TextEdit);
			
	// Enable word wrapping and disable auto-height expansion
	p_chat_dock->input_field->set_line_wrapping_mode(TextEdit::LINE_WRAPPING_BOUNDARY);
	p_chat_dock->input_field->set_fit_content_height_enabled(false);
	
	// Remove border from input field since panel provides it
	Ref<StyleBoxFlat> input_style = memnew(StyleBoxFlat);
	input_style->set_bg_color(Color(0, 0, 0, 0)); // Transparent background
	input_style->set_border_width_all(0);
	input_style->set_corner_radius_all(0);
	input_style->set_content_margin(SIDE_LEFT, 24); // Double the padding from left edge
	input_style->set_content_margin(SIDE_RIGHT, 16); // Double the padding
	input_style->set_content_margin(SIDE_TOP, 16); // Double the padding
	input_style->set_content_margin(SIDE_BOTTOM, 16); // Double the padding
	p_chat_dock->input_field->add_theme_style_override("normal", input_style);
	p_chat_dock->input_field->add_theme_style_override("focus", input_style);
	p_chat_dock->input_field->add_theme_style_override("read_only", input_style); // Keep consistent during streaming
	
	p_chat_dock->input_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	p_chat_dock->input_field->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	p_chat_dock->input_field->set_placeholder("Ask, learn, brainstorm");
	p_chat_dock->input_field->set_custom_minimum_size(Size2(0, 80)); // Reduced from 150 to be more compact
	
	// Enable caret blinking for better responsiveness
	p_chat_dock->input_field->set_caret_blink_enabled(true);
	p_chat_dock->input_field->set_caret_blink_interval(0.5); // Blink every 0.5 seconds
	
	p_chat_dock->input_field->connect("text_changed", callable_mp(p_chat_dock, &AIChatDock::_on_input_text_changed));
	p_chat_dock->input_field->connect("gui_input", callable_mp(p_chat_dock, &AIChatDock::_on_input_field_gui_input));
	input_content->add_child(p_chat_dock->input_field);

	// --- BOTTOM TOOLBAR (Model selector and Send/Stop buttons) ---
	MarginContainer *bottom_toolbar_margin = memnew(MarginContainer);
	bottom_toolbar_margin->add_theme_constant_override("margin_left", 8);
	bottom_toolbar_margin->add_theme_constant_override("margin_right", 8);
	bottom_toolbar_margin->add_theme_constant_override("margin_top", 4);
	bottom_toolbar_margin->add_theme_constant_override("margin_bottom", 8);
	input_content->add_child(bottom_toolbar_margin);

	HBoxContainer *bottom_toolbar = memnew(HBoxContainer);
	bottom_toolbar->add_theme_constant_override("separation", 8);
	bottom_toolbar_margin->add_child(bottom_toolbar);

    // Inline container to hold model selector and mode selector side-by-side
    HBoxContainer *model_mode_box = memnew(HBoxContainer);
    model_mode_box->add_theme_constant_override("separation", 10);

    // Model dropdown (styled as text)
    p_chat_dock->model_dropdown = memnew(OptionButton);
    p_chat_dock->model_dropdown->set_flat(true);
    p_chat_dock->model_dropdown->set_clip_text(false);
    p_chat_dock->model_dropdown->set_custom_minimum_size(Size2(0, 28));
    p_chat_dock->model_dropdown->connect("item_selected", callable_mp(p_chat_dock, &AIChatDock::_on_model_selected));

    Ref<StyleBoxEmpty> empty_style = memnew(StyleBoxEmpty);
    p_chat_dock->model_dropdown->add_theme_style_override("normal", empty_style);
    p_chat_dock->model_dropdown->add_theme_style_override("hover", empty_style);
    p_chat_dock->model_dropdown->add_theme_style_override("pressed", empty_style);
    p_chat_dock->model_dropdown->add_theme_style_override("focus", empty_style);

    p_chat_dock->model_dropdown->add_theme_constant_override("arrow_margin", 0);
    p_chat_dock->model_dropdown->add_theme_constant_override("modulate_arrow", 1);
    p_chat_dock->model_dropdown->add_theme_constant_override("h_separation", 0);
    Ref<ImageTexture> transparent_icon = memnew(ImageTexture);
    p_chat_dock->model_dropdown->add_theme_icon_override("arrow", transparent_icon);

    Color text_color = p_chat_dock->get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.6);
    Color hover_color = p_chat_dock->get_theme_color(SNAME("font_color"), SNAME("Editor"));
    p_chat_dock->model_dropdown->add_theme_color_override("font_color", text_color);
    p_chat_dock->model_dropdown->add_theme_color_override("font_hover_color", hover_color);
    p_chat_dock->model_dropdown->add_theme_color_override("font_pressed_color", hover_color);
    p_chat_dock->model_dropdown->add_theme_font_size_override("font_size", 22);
    p_chat_dock->model_dropdown->set_fit_to_longest_item(false);

    model_mode_box->add_child(p_chat_dock->model_dropdown);

    // Mode selector (Ask/Agent)
    AIChatModeSelector *mode_selector = memnew(AIChatModeSelector);
    mode_selector->setup(p_chat_dock);
    mode_selector->connect("mode_changed", callable_mp(p_chat_dock, &AIChatDock::_on_mode_changed));
    model_mode_box->add_child(mode_selector);

    bottom_toolbar->add_child(model_mode_box);

	// Spacer to push send button to right
	Control *bottom_spacer = memnew(Control);
	bottom_spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	bottom_toolbar->add_child(bottom_spacer);

	// Send button - circular icon-only button
	p_chat_dock->send_button = memnew(Button);
	p_chat_dock->send_button->set_text(""); // Icon only, no text
	p_chat_dock->send_button->set_disabled(true);
	p_chat_dock->send_button->add_theme_icon_override("icon", p_chat_dock->get_theme_icon(SNAME("Play"), SNAME("EditorIcons")));
	p_chat_dock->send_button->set_custom_minimum_size(Size2(36, 36)); // Larger for circular button
	p_chat_dock->send_button->set_tooltip_text("Send (Enter)");
	
	style_send_button(p_chat_dock->send_button, p_chat_dock);
	
	p_chat_dock->send_button->connect("pressed", callable_mp(p_chat_dock, &AIChatDock::_on_send_button_pressed));
	bottom_toolbar->add_child(p_chat_dock->send_button);

	// Stop button - circular icon-only button (initially hidden)
	p_chat_dock->stop_button = memnew(Button);
	p_chat_dock->stop_button->set_text(""); // Icon only, no text
	p_chat_dock->stop_button->set_visible(false);
	p_chat_dock->stop_button->add_theme_icon_override("icon", p_chat_dock->get_theme_icon(SNAME("Stop"), SNAME("EditorIcons")));
	p_chat_dock->stop_button->set_custom_minimum_size(Size2(36, 36)); // Larger for circular button
	p_chat_dock->stop_button->set_tooltip_text("Stop");
	
	style_stop_button(p_chat_dock->stop_button, p_chat_dock, false);

	p_chat_dock->stop_button->connect("pressed", callable_mp(p_chat_dock, &AIChatDock::_on_stop_button_pressed));
	bottom_toolbar->add_child(p_chat_dock->stop_button);
}

void AIChatInputBox::style_send_button(Button *p_send_button, AIChatDock *p_chat_dock) {
	// Normal style - circular button
	Ref<StyleBoxFlat> button_style = memnew(StyleBoxFlat);
	button_style->set_bg_color(p_chat_dock->get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	button_style->set_corner_radius_all(18); // Fully circular (half of 36px)
	button_style->set_content_margin_all(6);
	p_send_button->add_theme_style_override("normal", button_style);
	
	// Hover style - circular
	Ref<StyleBoxFlat> button_hover_style = memnew(StyleBoxFlat);
	button_hover_style->set_bg_color(p_chat_dock->get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1.15, 1.15, 1.15));
	button_hover_style->set_corner_radius_all(18); // Fully circular
	button_hover_style->set_content_margin_all(6);
	p_send_button->add_theme_style_override("hover", button_hover_style);
	
	// Disabled style - circular
	Ref<StyleBoxFlat> button_disabled_style = memnew(StyleBoxFlat);
	button_disabled_style->set_bg_color(p_chat_dock->get_theme_color(SNAME("disabled_bg_color"), SNAME("Editor")));
	button_disabled_style->set_corner_radius_all(18); // Fully circular
	button_disabled_style->set_content_margin_all(6);
	p_send_button->add_theme_style_override("disabled", button_disabled_style);
}

void AIChatInputBox::style_stop_button(Button *p_stop_button, AIChatDock *p_chat_dock, bool p_enabled) {
	// White background - circular button
	Ref<StyleBoxFlat> stop_button_style = memnew(StyleBoxFlat);
	stop_button_style->set_bg_color(Color(1.0, 1.0, 1.0)); // White
	stop_button_style->set_corner_radius_all(18); // Fully circular (half of 36px)
	stop_button_style->set_content_margin_all(6);
	p_stop_button->add_theme_style_override("normal", stop_button_style);
	
	// Hover style - slightly darker white, circular
	Ref<StyleBoxFlat> stop_button_hover_style = memnew(StyleBoxFlat);
	stop_button_hover_style->set_bg_color(Color(0.95, 0.95, 0.95));
	stop_button_hover_style->set_corner_radius_all(18); // Fully circular
	stop_button_hover_style->set_content_margin_all(6);
	p_stop_button->add_theme_style_override("hover", stop_button_hover_style);
	
	// Dark gray when disabled - circular
	Ref<StyleBoxFlat> stop_button_disabled_style = memnew(StyleBoxFlat);
	stop_button_disabled_style->set_bg_color(Color(0.3, 0.3, 0.3)); // Dark gray
	stop_button_disabled_style->set_corner_radius_all(18); // Fully circular
	stop_button_disabled_style->set_content_margin_all(6);
	p_stop_button->add_theme_style_override("disabled", stop_button_disabled_style);
}

