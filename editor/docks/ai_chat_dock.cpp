/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */
#include "ai_chat_dock.h"
#include "core/io/config_file.h"
#include "core/io/json.h"
#include "core/os/time.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource.h"
#include "core/io/dir_access.h"
#include "core/io/image.h"
#include "core/string/ustring.h"
#include "core/io/marshalls.h"
#include "core/core_bind.h"
#include "core/config/project_settings.h"
#include "scene/resources/image_texture.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_paths.h"
#include "editor/editor_interface.h"
#include "editor/editor_string_names.h"
#include "editor/settings/editor_settings.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/script/script_text_editor.h"
#include "modules/gdscript/gdscript.h"
#include "modules/gdscript/gdscript_cache.h"
#include "editor/gui/editor_file_dialog.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/flow_container.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/rich_text_label.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/separator.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/tree.h"
#include "scene/main/scene_tree.h"
void AIChatDock::_on_tool_result_retry_timeout(const String &p_tool_call_id, const String &p_tool_name, const String &p_content, const Array &p_tool_results) {
    if (tool_calls_ui_applied.has(p_tool_call_id)) {
        return;
    }
    _apply_tool_result_deferred(p_tool_call_id, p_tool_name, p_content, p_tool_results);
}
#include "scene/gui/texture_rect.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/popup_menu.h"
#include "scene/main/http_request.h"
#include "scene/resources/style_box_flat.h"
#include "ai_auth_helper.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/node_3d.h"
#include "core/string/string_name.h"
#include "editor/file_system/editor_file_system.h"
#include "scene/gui/popup.h"
#include "scene/gui/control.h"

#include "../ai/editor_tools.h"
#include "diff_viewer.h"

void AIChatDock::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_send_button_pressed"), &AIChatDock::_on_send_button_pressed);
	ClassDB::bind_method(D_METHOD("_on_stop_button_pressed"), &AIChatDock::_on_stop_button_pressed);
	ClassDB::bind_method(D_METHOD("_on_stop_request_completed"), &AIChatDock::_on_stop_request_completed);
	ClassDB::bind_method(D_METHOD("_on_edit_message_pressed"), &AIChatDock::_on_edit_message_pressed);
	ClassDB::bind_method(D_METHOD("_on_edit_message_send_pressed"), &AIChatDock::_on_edit_message_send_pressed);
	ClassDB::bind_method(D_METHOD("_on_edit_send_button_pressed"), &AIChatDock::_on_edit_send_button_pressed);
	ClassDB::bind_method(D_METHOD("_on_edit_message_cancel_pressed"), &AIChatDock::_on_edit_message_cancel_pressed);
	ClassDB::bind_method(D_METHOD("_on_edit_field_gui_input"), &AIChatDock::_on_edit_field_gui_input);
	ClassDB::bind_method(D_METHOD("_process_send_request_async"), &AIChatDock::_process_send_request_async);
	ClassDB::bind_method(D_METHOD("_save_conversations_async"), &AIChatDock::_save_conversations_async);
	ClassDB::bind_method(D_METHOD("_on_input_text_changed"), &AIChatDock::_on_input_text_changed);
	ClassDB::bind_method(D_METHOD("_update_at_mention_popup"), &AIChatDock::_update_at_mention_popup);
	ClassDB::bind_method(D_METHOD("_populate_at_mention_tree"), &AIChatDock::_populate_at_mention_tree);
	ClassDB::bind_method(D_METHOD("_populate_tree_recursive"), &AIChatDock::_populate_tree_recursive);
	ClassDB::bind_method(D_METHOD("_on_at_mention_item_selected"), &AIChatDock::_on_at_mention_item_selected);
	ClassDB::bind_method(D_METHOD("_on_input_field_gui_input"), &AIChatDock::_on_input_field_gui_input);
	ClassDB::bind_method(D_METHOD("_on_model_selected"), &AIChatDock::_on_model_selected);
	ClassDB::bind_method(D_METHOD("_populate_cerebras_models"), &AIChatDock::_populate_cerebras_models);
	ClassDB::bind_method(D_METHOD("_on_models_request_completed"), &AIChatDock::_on_models_request_completed);
	ClassDB::bind_method(D_METHOD("_on_index_button_pressed"), &AIChatDock::_on_index_button_pressed);
	ClassDB::bind_method(D_METHOD("_on_tool_output_toggled"), &AIChatDock::_on_tool_output_toggled);
	ClassDB::bind_method(D_METHOD("_on_tool_file_link_pressed", "path"), &AIChatDock::_on_tool_file_link_pressed);
	ClassDB::bind_method(D_METHOD("_on_chat_scroll_changed", "value"), &AIChatDock::_on_chat_scroll_changed);

	ClassDB::bind_method(D_METHOD("_on_attachment_menu_item_pressed"), &AIChatDock::_on_attachment_menu_item_pressed);
	ClassDB::bind_method(D_METHOD("_on_attach_files_pressed"), &AIChatDock::_on_attach_files_pressed);
	ClassDB::bind_method(D_METHOD("_on_attach_scene_nodes_pressed"), &AIChatDock::_on_attach_scene_nodes_pressed);
	ClassDB::bind_method(D_METHOD("_on_attach_current_script_pressed"), &AIChatDock::_on_attach_current_script_pressed);
	ClassDB::bind_method(D_METHOD("_on_attach_resources_pressed"), &AIChatDock::_on_attach_resources_pressed);
	ClassDB::bind_method(D_METHOD("_on_reindex_project_pressed"), &AIChatDock::_on_reindex_project_pressed);
	ClassDB::bind_method(D_METHOD("_perform_project_reindex"), &AIChatDock::_perform_project_reindex);
	ClassDB::bind_method(D_METHOD("_on_reindex_response"), &AIChatDock::_on_reindex_response);
	ClassDB::bind_method(D_METHOD("_on_scene_tree_node_selected"), &AIChatDock::_on_scene_tree_node_selected);
	ClassDB::bind_method(D_METHOD("_on_files_selected"), &AIChatDock::_on_files_selected);
	ClassDB::bind_method(D_METHOD("_on_remove_attachment", "path"), &AIChatDock::_on_remove_attachment);
	ClassDB::bind_method(D_METHOD("_on_conversation_selected"), &AIChatDock::_on_conversation_selected);
	ClassDB::bind_method(D_METHOD("_on_new_conversation_pressed"), &AIChatDock::_on_new_conversation_pressed);
	ClassDB::bind_method(D_METHOD("_on_save_image_pressed", "base64_data", "format"), &AIChatDock::_on_save_image_pressed);
	ClassDB::bind_method(D_METHOD("_on_save_image_location_selected", "file_path"), &AIChatDock::_on_save_image_location_selected);

	ClassDB::bind_method(D_METHOD("_save_conversations_to_disk"), &AIChatDock::_save_conversations_to_disk);
	ClassDB::bind_method(D_METHOD("_process_image_attachment_async"), &AIChatDock::_process_image_attachment_async);
	ClassDB::bind_method(D_METHOD("_send_chat_request_chunked"), &AIChatDock::_send_chat_request_chunked);
	ClassDB::bind_method(D_METHOD("_apply_tool_result_deferred"), &AIChatDock::_apply_tool_result_deferred);

	ClassDB::bind_method(D_METHOD("_finalize_chat_request"), &AIChatDock::_finalize_chat_request);
	ClassDB::bind_method(D_METHOD("_save_conversations_chunked"), &AIChatDock::_save_conversations_chunked);
	ClassDB::bind_method(D_METHOD("_finalize_conversations_save"), &AIChatDock::_finalize_conversations_save);
	ClassDB::bind_method(D_METHOD("_execute_delayed_save"), &AIChatDock::_execute_delayed_save);
	ClassDB::bind_method(D_METHOD("_display_generated_image_deferred", "base64_data", "id"), &AIChatDock::_display_generated_image_deferred);
    ClassDB::bind_method(D_METHOD("_on_apply_edit_thread_done"), &AIChatDock::_on_apply_edit_thread_done);
	
	// Embedding system methods
	ClassDB::bind_method(D_METHOD("_initialize_embedding_system"), &AIChatDock::_initialize_embedding_system);
	ClassDB::bind_method(D_METHOD("_perform_initial_indexing"), &AIChatDock::_perform_initial_indexing);
	ClassDB::bind_method(D_METHOD("_check_index_status_and_start_if_needed"), &AIChatDock::_check_index_status_and_start_if_needed);
	ClassDB::bind_method(D_METHOD("_on_index_status_response"), &AIChatDock::_on_index_status_response);
	ClassDB::bind_method(D_METHOD("_scan_and_index_project_files"), &AIChatDock::_scan_and_index_project_files);
	ClassDB::bind_method(D_METHOD("_send_file_batch"), &AIChatDock::_send_file_batch);
	ClassDB::bind_method(D_METHOD("_on_embedding_request_completed"), &AIChatDock::_on_embedding_request_completed);
    ClassDB::bind_method(D_METHOD("_on_filesystem_debounced_scan", "scheduled_at"), &AIChatDock::_on_filesystem_debounced_scan);
    ClassDB::bind_method(D_METHOD("_perform_filesystem_scan_changes"), &AIChatDock::_perform_filesystem_scan_changes);
	ClassDB::bind_method(D_METHOD("_on_embedding_status_tick"), &AIChatDock::_on_embedding_status_tick);
	ClassDB::bind_method(D_METHOD("_show_diff_in_script_editor_deferred"), &AIChatDock::_show_diff_in_script_editor_deferred);

	ClassDB::bind_method(D_METHOD("_scroll_to_bottom"), &AIChatDock::_scroll_to_bottom);
	ClassDB::bind_method(D_METHOD("_perform_scroll"), &AIChatDock::_perform_scroll);
	ClassDB::bind_method(D_METHOD("_is_at_bottom"), &AIChatDock::_is_at_bottom);
	ClassDB::bind_method(D_METHOD("_on_chat_scroll_changed"), &AIChatDock::_on_chat_scroll_changed);
	ClassDB::bind_method(D_METHOD("_on_chat_content_min_size_changed"), &AIChatDock::_on_chat_content_min_size_changed);
	ClassDB::bind_method(D_METHOD("_scroll_to_bottom_smooth"), &AIChatDock::_scroll_to_bottom_smooth);

	ADD_SIGNAL(MethodInfo("chat_gui_input", PropertyInfo(Variant::OBJECT, "event", PROPERTY_HINT_RESOURCE_TYPE, "InputEvent")));
}

void AIChatDock::attach_external_file(const String &p_file_path) {
    if (p_file_path.is_empty()) {
        return;
    }
    Vector<String> files;
    files.push_back(p_file_path);
    // Reuse existing external attachment flow to handle images/text and UI updates.
    _attach_external_files(files);
}

void AIChatDock::_notification(int p_notification) {
	switch (p_notification) {
		case NOTIFICATION_POST_ENTER_TREE: {
			// UI setup moved from constructor to here to ensure theme is available.
			
			// Conversation history row
			HBoxContainer *history_container = memnew(HBoxContainer);
			add_child(history_container);

			Label *history_label = memnew(Label);
			history_label->set_text("Conversation:");
			history_container->add_child(history_label);

			conversation_history_dropdown = memnew(OptionButton);
			conversation_history_dropdown->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			conversation_history_dropdown->connect("item_selected", callable_mp(this, &AIChatDock::_on_conversation_selected));
			history_container->add_child(conversation_history_dropdown);

			new_conversation_button = memnew(Button);
			new_conversation_button->set_text("New");
			new_conversation_button->set_tooltip_text("Start a new conversation");
			new_conversation_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Add"), SNAME("EditorIcons")));
			new_conversation_button->connect("pressed", callable_mp(this, &AIChatDock::_on_new_conversation_pressed));
			history_container->add_child(new_conversation_button);

            // Model selection row
            HBoxContainer *top_container = memnew(HBoxContainer);
			add_child(top_container);

			Label *model_label = memnew(Label);
			model_label->set_text("Model:");
			top_container->add_child(model_label);

            model_dropdown = memnew(OptionButton);
            // Add multi-provider models
            model_dropdown->add_item("gpt-5");
			model_dropdown->add_item("gpt-4o");
			model_dropdown->add_item("claude-4");
			model_dropdown->add_item("gemini-2.5");
			// Add Cerebras models (high-speed inference) - will be populated dynamically
			// Using [FAST] prefix to avoid emoji encoding issues
			_populate_cerebras_models();
			model_dropdown->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			model_dropdown->connect("item_selected", callable_mp(this, &AIChatDock::_on_model_selected));
			top_container->add_child(model_dropdown);

            // Index button removed; indexing will be managed automatically

            // Embedding status label
            embedding_status_label = memnew(Label);
            embedding_status_label->set_text("");
            embedding_status_label->set_modulate(Color(0.8, 0.8, 0.8));
            top_container->add_child(embedding_status_label);
			
			// Setup authentication UI
			_setup_authentication_ui();

			// Container for attached files (initially hidden) - will be added to bottom panel later
			attached_files_container = memnew(HFlowContainer);
			attached_files_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			attached_files_container->add_theme_constant_override("h_separation", 6); // Horizontal spacing between attachment tabs
			attached_files_container->add_theme_constant_override("v_separation", 4); // Vertical spacing if wrapping
			attached_files_container->set_visible(false);

					// Create file dialog
		file_dialog = memnew(EditorFileDialog);
		file_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
		file_dialog->set_access(EditorFileDialog::ACCESS_RESOURCES);
		file_dialog->connect("files_selected", callable_mp(this, &AIChatDock::_on_files_selected));
		
		// Add support for common file types including images
		file_dialog->add_filter("*.png, *.jpg, *.jpeg, *.gif, *.bmp, *.webp, *.svg", "Image Files");
		file_dialog->add_filter("*.gd, *.cs, *.cpp, *.h, *.py, *.js, *.json, *.xml, *.txt, *.md", "Code & Text Files");
		file_dialog->add_filter("*", "All Files");
		
		add_child(file_dialog);

		// Create image warning dialog
		image_warning_dialog = memnew(AcceptDialog);
		image_warning_dialog->set_title("Image Downsampled");
		add_child(image_warning_dialog);

		// Create save image dialog
		save_image_dialog = memnew(EditorFileDialog);
		save_image_dialog->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
		save_image_dialog->set_access(EditorFileDialog::ACCESS_RESOURCES);
		save_image_dialog->add_filter("*.png", "PNG Images");
		save_image_dialog->add_filter("*.jpg", "JPEG Images");
		save_image_dialog->connect("file_selected", callable_mp(this, &AIChatDock::_on_save_image_location_selected));
		add_child(save_image_dialog);

		// --- At-Mention Popup ---
		at_mention_popup = memnew(PopupPanel);
		at_mention_popup->set_name("at_mention_popup");
		at_mention_popup->set_size(Size2i(300, 400));
		
		VBoxContainer *at_mention_vbox = memnew(VBoxContainer);
		at_mention_popup->add_child(at_mention_vbox);
		
		at_mention_tree = memnew(Tree);
		at_mention_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		at_mention_tree->connect("item_activated", callable_mp(this, &AIChatDock::_on_at_mention_item_selected));
		at_mention_vbox->add_child(at_mention_tree);
		
		add_child(at_mention_popup);



		// --- Scene Tree Popup ---
		scene_tree_popup = memnew(PopupPanel);
		scene_tree_popup->set_name("scene_tree_popup");
		scene_tree_popup->set_size(Size2i(400, 500));
		
		VBoxContainer *scene_tree_vbox = memnew(VBoxContainer);
		scene_tree_popup->add_child(scene_tree_vbox);
		
		Label *scene_tree_label = memnew(Label);
		scene_tree_label->set_text("Select Scene Nodes to Attach:");
		scene_tree_vbox->add_child(scene_tree_label);
		
		scene_tree = memnew(Tree);
		scene_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
		scene_tree->set_select_mode(Tree::SELECT_MULTI);
		scene_tree->connect("item_activated", callable_mp(this, &AIChatDock::_on_scene_tree_node_selected));
		scene_tree_vbox->add_child(scene_tree);
		
		HBoxContainer *scene_tree_buttons = memnew(HBoxContainer);
		scene_tree_vbox->add_child(scene_tree_buttons);
		
		Button *attach_selected_button = memnew(Button);
		attach_selected_button->set_text("Attach Selected");
		attach_selected_button->connect("pressed", callable_mp(this, &AIChatDock::_on_scene_tree_node_selected));
		scene_tree_buttons->add_child(attach_selected_button);
		
		Control *spacer_control = memnew(Control); // Spacer
		spacer_control->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		scene_tree_buttons->add_child(spacer_control);
		
		Button *scene_tree_cancel_button = memnew(Button);
		scene_tree_cancel_button->set_text("Cancel");
		scene_tree_cancel_button->connect("pressed", Callable(scene_tree_popup, "hide"));
		scene_tree_buttons->add_child(scene_tree_cancel_button);
		
		add_child(scene_tree_popup);

		// --- Resource Dialog ---
		resource_dialog = memnew(EditorFileDialog);
		resource_dialog->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
		resource_dialog->set_access(EditorFileDialog::ACCESS_RESOURCES);
		resource_dialog->add_filter("*.tres, *.res", "Resources");
		resource_dialog->add_filter("*.tscn, *.scn", "Scenes");
		resource_dialog->add_filter("*.png, *.jpg, *.jpeg, *.svg", "Textures");
		resource_dialog->add_filter("*.ogg, *.wav, *.mp3", "Audio");
		resource_dialog->add_filter("*", "All Files");
		resource_dialog->connect("files_selected", callable_mp(this, &AIChatDock::_on_files_selected));
		add_child(resource_dialog);

			// Chat history area - expand to fill available space
			chat_scroll = memnew(ScrollContainer);
			chat_scroll->set_v_size_flags(Control::SIZE_EXPAND_FILL);
			chat_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
			add_child(chat_scroll);
			
			// Connect scroll change event
			chat_scroll->get_v_scroll_bar()->connect("value_changed", callable_mp(this, &AIChatDock::_on_chat_scroll_changed), CONNECT_DEFERRED);
			// Connect content size changes to auto-scroll when near bottom
			// We connect after creating chat_container below

			chat_container = memnew(VBoxContainer);
			chat_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			chat_scroll->add_child(chat_container);
			chat_container->connect("minimum_size_changed", callable_mp(this, &AIChatDock::_on_chat_content_min_size_changed), CONNECT_DEFERRED);

			// Add a container for attachments just above the input field
			VBoxContainer *bottom_panel = memnew(VBoxContainer);
			add_child(bottom_panel);
			
			// Add attachments container to the bottom panel (positioned above input)
			bottom_panel->add_child(attached_files_container);
			
			// Attach files button row (above input)
			HBoxContainer *attach_container = memnew(HBoxContainer);
			bottom_panel->add_child(attach_container);

			// Add spacer to push button to the right
			Control *spacer = memnew(Control);
			spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			attach_container->add_child(spacer);

			// Attach files button
			attach_button = memnew(MenuButton);
			attach_button->set_text("Attach");
			attach_button->set_tooltip_text("Attach project files to your message");
			attach_button->add_theme_icon_override("icon", get_theme_icon(SNAME("FileList"), SNAME("EditorIcons")));
			attach_button->set_custom_minimum_size(Size2(80, 32));
			
			// Set up the popup menu
			PopupMenu *popup = attach_button->get_popup();
			popup->add_item("Files", 0);
			popup->set_item_icon(0, get_theme_icon(SNAME("FileList"), SNAME("EditorIcons")));
			popup->add_item("Scene Nodes", 1);
			popup->set_item_icon(1, get_theme_icon(SNAME("SceneTree"), SNAME("EditorIcons")));
			popup->add_item("Current Script", 2);
			popup->set_item_icon(2, get_theme_icon(SNAME("Script"), SNAME("EditorIcons")));
			popup->add_item("Resources", 3);
			popup->set_item_icon(3, get_theme_icon(SNAME("ResourcePreloader"), SNAME("EditorIcons")));
			popup->add_separator();
			popup->add_item("Re-index Project", 4);
			popup->set_item_icon(4, get_theme_icon(SNAME("Reload"), SNAME("EditorIcons")));
			popup->connect("id_pressed", callable_mp(this, &AIChatDock::_on_attachment_menu_item_pressed));
			
			attach_container->add_child(attach_button);

			// Add some spacing before input area
			Control *input_spacer = memnew(Control);
			input_spacer->set_custom_minimum_size(Size2(0, 4));
			add_child(input_spacer);

			// Input area at the bottom
			HBoxContainer *input_container = memnew(HBoxContainer);
			input_container->add_theme_constant_override("separation", 8); // Gap between input and send button
			add_child(input_container);

			input_field = memnew(TextEdit);
					
		// Enable word wrapping and disable horizontal scrolling
		input_field->set_line_wrapping_mode(TextEdit::LINE_WRAPPING_BOUNDARY);
		input_field->set_fit_content_height_enabled(true);
					
			Ref<StyleBoxFlat> input_style = memnew(StyleBoxFlat);
			input_style->set_bg_color(get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
			input_style->set_border_width_all(2);
			input_style->set_border_color(get_theme_color(SNAME("dark_color_3"), SNAME("Editor")));
			input_style->set_corner_radius_all(8); // Modern rounded input
			input_style->set_content_margin_all(8); // Better inner padding
			input_field->add_theme_style_override("normal", input_style);
			
			// Focus style for modern interaction
			Ref<StyleBoxFlat> input_focus_style = memnew(StyleBoxFlat);
			input_focus_style->set_bg_color(get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
			input_focus_style->set_border_width_all(2);
			input_focus_style->set_border_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.6));
			input_focus_style->set_corner_radius_all(8);
			input_focus_style->set_content_margin_all(8);
			input_field->add_theme_style_override("focus", input_focus_style);
			
			input_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			input_field->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
			input_field->set_placeholder("Ask me anything about Orca...");
			input_field->set_custom_minimum_size(Size2(0, 100)); // Minimum height for the input box
			input_field->connect("text_changed", callable_mp(this, &AIChatDock::_on_input_text_changed));
			input_field->connect("gui_input", callable_mp(this, &AIChatDock::_on_input_field_gui_input));
			input_container->add_child(input_field);

			send_button = memnew(Button);
			send_button->set_text("Send");
			send_button->set_disabled(true);
			send_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Play"), SNAME("EditorIcons")));
			send_button->set_custom_minimum_size(Size2(80, 40));
			
			// Modern button styling
			Ref<StyleBoxFlat> button_style = memnew(StyleBoxFlat);
			button_style->set_bg_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")));
			button_style->set_corner_radius_all(6);
			button_style->set_content_margin_all(8);
			send_button->add_theme_style_override("normal", button_style);
			
			// Hover style
			Ref<StyleBoxFlat> button_hover_style = memnew(StyleBoxFlat);
			button_hover_style->set_bg_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1.1, 1.1, 1.1));
			button_hover_style->set_corner_radius_all(6);
			button_hover_style->set_content_margin_all(8);
			send_button->add_theme_style_override("hover", button_hover_style);
			
			send_button->connect("pressed", callable_mp(this, &AIChatDock::_on_send_button_pressed));
			input_container->add_child(send_button);

			// Stop button (initially hidden)
			stop_button = memnew(Button);
			stop_button->set_text("Stop");
			stop_button->set_visible(false); // Hidden by default
			stop_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Stop"), SNAME("EditorIcons")));
			stop_button->set_custom_minimum_size(Size2(80, 40));
			
			// Stop button styling (red theme)
			Ref<StyleBoxFlat> stop_button_style = memnew(StyleBoxFlat);
			stop_button_style->set_bg_color(Color(0.8, 0.2, 0.2)); // Red color
			stop_button_style->set_corner_radius_all(6);
			stop_button_style->set_content_margin_all(8);
			stop_button->add_theme_style_override("normal", stop_button_style);
			
			// Stop button hover style
			Ref<StyleBoxFlat> stop_button_hover_style = memnew(StyleBoxFlat);
			stop_button_hover_style->set_bg_color(Color(0.9, 0.3, 0.3)); // Lighter red on hover
			stop_button_hover_style->set_corner_radius_all(6);
			stop_button_hover_style->set_content_margin_all(8);
			stop_button->add_theme_style_override("hover", stop_button_hover_style);
			
			stop_button->connect("pressed", callable_mp(this, &AIChatDock::_on_stop_button_pressed));
			input_container->add_child(stop_button);

			// Load saved model from settings, now that UI is ready. Restrict to allowed models.
			if (EditorSettings::get_singleton()->has_setting("ai_chat/model")) {
				String saved_model = EditorSettings::get_singleton()->get_setting("ai_chat/model");
				// Basic validation - allow base models and any [FAST] models
				bool is_valid = (saved_model == "gpt-5" || saved_model == "gpt-4o" || 
								saved_model == "claude-4" || saved_model == "gemini-2.5" ||
								saved_model.begins_with("[FAST] "));
				if (!is_valid) {
					saved_model = "gpt-5"; // Fallback to default model
				}
				model = saved_model;
				// Set the dropdown to the saved model
				for (int i = 0; i < model_dropdown->get_item_count(); i++) {
					String dropdown_text = model_dropdown->get_item_text(i);
					if (dropdown_text == saved_model) {
						model_dropdown->select(i);
						break;
					}
				}
			}

            // Initialize conversation system after UI is ready
            // Prefer storing in project settings dir to avoid per-app-name drift and ensure portability with the project
            if (EditorPaths::get_singleton() && EditorPaths::get_singleton()->are_paths_valid()) {
                String new_path = EditorPaths::get_singleton()->get_project_settings_dir().path_join("ai_chat_conversations.simplifine");
                // Migrate from legacy location if needed
                String legacy_path = OS::get_singleton()->get_user_data_dir().path_join("ai_chat_conversations.simplifine");
                if (FileAccess::exists(legacy_path) && !FileAccess::exists(new_path)) {
                    Ref<DirAccess> da = DirAccess::create_for_path(new_path.get_base_dir());
                    if (da.is_valid() && !da->dir_exists(new_path.get_base_dir())) {
                        da->make_dir_recursive(new_path.get_base_dir());
                    }
                    Error copy_err;
                    PackedByteArray bytes = FileAccess::get_file_as_bytes(legacy_path, &copy_err);
                    if (copy_err == OK) {
                        Error write_err;
                        Ref<FileAccess> out = FileAccess::open(new_path, FileAccess::WRITE, &write_err);
                        if (write_err == OK) {
                            out->store_buffer(bytes);
                            out->close();
                            // Optionally delete legacy file
                            Ref<DirAccess> da_old = DirAccess::open(legacy_path.get_base_dir());
                            if (da_old.is_valid()) {
                                da_old->remove(legacy_path.get_file());
                            }
                        }
                    }
                }
                conversations_file_path = new_path;
            } else {
                conversations_file_path = OS::get_singleton()->get_user_data_dir().path_join("ai_chat_conversations.simplifine");
            }
            _load_conversations();

            // Ensure indexing is set up immediately on startup (guest if needed)
            _ensure_project_indexing();
			
			// Create first conversation if none exist
            if (conversations.is_empty()) {
                _create_new_conversation();
                _update_conversation_dropdown();
                // Ensure first auto-created conversation is persisted
				_queue_delayed_save();
				_execute_delayed_save(); // start background save immediately
            } else {
				// Switch to the most recent conversation
				_switch_to_conversation(conversations.size() - 1);
			}
			
			_update_conversation_dropdown();
		} break;
		case NOTIFICATION_PROCESS: {
			if (http_client.is_valid()) {
				http_client->poll();
				HTTPClient::Status client_status = http_client->get_status();
				if (http_status == STATUS_CONNECTING && client_status == HTTPClient::STATUS_CONNECTED) {
					http_status = STATUS_REQUESTING;
					Error err = http_client->request(HTTPClient::METHOD_POST, pending_request_path, pending_request_headers, pending_request_body.ptr(), pending_request_body.size());
					if (err != OK) {
						_add_message_to_chat("system", "Failed to send request to backend.");
						is_waiting_for_response = false;
						_update_ui_state();
						http_status = STATUS_DONE;
						return;
					}
				} else if (http_status == STATUS_REQUESTING && client_status == HTTPClient::STATUS_BODY) {
					http_status = STATUS_BODY;
				}

				if (client_status == HTTPClient::STATUS_BODY) {
					PackedByteArray chunk = http_client->read_response_body_chunk();
					if (chunk.size() > 0) {
						_handle_response_chunk(chunk);
					}
				}

                if (client_status == HTTPClient::STATUS_DISCONNECTED || client_status == HTTPClient::STATUS_CONNECTION_ERROR || client_status == HTTPClient::STATUS_CANT_CONNECT) {
                    if (stream_completed_successfully) {
                        print_line("AI Chat: Stream completed successfully, server closed connection");
                    } else {
                        print_line("AI Chat: HTTP connection failed with status: " + String::num_int64(client_status));
                        _add_message_to_chat("system", "Connection lost or failed (Status: " + String::num_int64(client_status) + ") - Try again please.");
                    }

                    // If we still have async tool tasks running (e.g., apply_edit), keep the UI in 'waiting' state
                    bool has_async_work = pending_tool_tasks > 0;
                    is_waiting_for_response = has_async_work ? true : false;
                    // Clean up stop mechanism state
                    stop_requested = false;
                    current_request_id = "";
                    _update_ui_state();
                    http_status = STATUS_DONE;
                    current_assistant_message_label = nullptr;
                    // Stop HTTP polling; async tasks will resume the chat when done
                    set_process(false);

                    // CRITICAL: Save conversation when request completes to ensure no data loss
                    if (current_conversation_index >= 0) {
                        conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
                        _queue_delayed_save();
                    }
                }
			}
		} break;
		case NOTIFICATION_ENTER_TREE: {
    // Load API key from editor settings
			if (EditorSettings::get_singleton()->has_setting("ai_chat/api_key")) {
				api_key = EditorSettings::get_singleton()->get_setting("ai_chat/api_key");
			}
    // Resolve API endpoint based on environment
    String is_dev = OS::get_singleton()->get_environment("IS_DEV");
    if (is_dev.is_empty()) {
        // Backward-compat with DEV_MODE
        is_dev = OS::get_singleton()->get_environment("DEV_MODE");
    }
    if (!is_dev.is_empty() && is_dev.to_lower() == "true") {
        api_endpoint = "http://127.0.0.1:8000/chat";
    } else {
        api_endpoint = "https://gamechat.simplifine.com/chat";
    }
		} break;
			case NOTIFICATION_READY: {
			// Auto-verify saved authentication when everything is fully ready
			_auto_verify_saved_credentials();
			// Debug: Log environment mode and resolved endpoint
			String is_dev_env = OS::get_singleton()->get_environment("IS_DEV");
			if (is_dev_env.is_empty()) {
				is_dev_env = OS::get_singleton()->get_environment("DEV_MODE");
			}
			String mode = (!is_dev_env.is_empty() && is_dev_env.to_lower() == "true") ? String("DEV") : String("PROD");
			print_line("AI Chat: Launch mode=" + mode + ", endpoint=" + api_endpoint);
		} break;
			case NOTIFICATION_EXIT_TREE: {
				// Final flush save to ensure no data loss on exit
				if (save_timer) {
					save_timer->stop();
				}
				if (save_thread_busy && save_thread) {
					save_thread->wait_to_finish();
					memdelete(save_thread);
					save_thread = nullptr;
					save_thread_busy = false;
				}
				// Always perform a final synchronous save
				_save_conversations();
			} break;
		case NOTIFICATION_THEME_CHANGED: {
			// Theme updates handled automatically
		} break;
	}
}
void AIChatDock::_on_send_button_pressed() {
	String message = input_field->get_text().strip_edges();
	if (message.is_empty() || is_waiting_for_response) {
		return;
	}

	// Auto-suggest relevant files based on message content
	if (embedding_system_initialized && initial_indexing_done) {
		_auto_attach_relevant_context();
	}

	// PERFORMANCE OPTIMIZATION: Instant UI feedback with deferred heavy processing
	// This prevents UI freezing by splitting operations across frames
	input_field->set_text("");
	is_waiting_for_response = true;
	_update_ui_state();
	
	// Force auto-scroll to re-engage when user sends a message.
	auto_scroll_at_bottom = true;

	// Create message with attached files (lightweight operation)
	AIChatDock::ChatMessage msg;
	msg.role = "user";
	msg.content = message;
	msg.timestamp = _get_timestamp();
	msg.attached_files = current_attached_files;
	
	Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
	chat_history.push_back(msg);
	_create_message_bubble(msg, chat_history.size() - 1);
	call_deferred("_scroll_to_bottom");

	// Clear attachments immediately for instant feedback
	_clear_attachments();
	
	// Reset stream completion flag for new request
	stream_completed_successfully = false;
	
	// Defer heavy processing to next frame to keep UI responsive
	call_deferred("_process_send_request_async");
	input_field->grab_focus();
}
void AIChatDock::_on_stop_button_pressed() {
	print_line("AI Chat: Stop button pressed! is_waiting_for_response=" + String(is_waiting_for_response ? "true" : "false") + ", current_request_id='" + current_request_id + "'");

	// Always honor stop: cancel backend stream if we have a request_id,
	// and locally stop any ongoing HTTP streaming and async tool tasks.
	stop_requested = true;
	
	// Best-effort backend stop when possible.
	if (!current_request_id.is_empty()) {
		print_line("AI Chat: Stop button pressed, sending stop request for: " + current_request_id);
		_send_stop_request();
	}

	// Immediately abort local HTTP streaming to unblock UI.
	if (http_client.is_valid()) {
		HTTPClient::Status st = http_client->get_status();
		if (st != HTTPClient::STATUS_DISCONNECTED) {
			print_line("AI Chat: Aborting HTTP stream locally");
			http_client->close();
		}
	}
	http_status = STATUS_DONE;
	set_process(false);

	// Drop any pending assistant label reference to avoid UI deadlocks.
	current_assistant_message_label = nullptr;

	// Reflect immediately in UI.
	is_waiting_for_response = false;
	_update_ui_state();
}
void AIChatDock::_send_stop_request() {
	if (current_request_id.is_empty()) {
		print_line("AI Chat: Cannot send stop request - no current request ID");
		return;
	}
	
	// Prepare stop request data
	Dictionary request_data;
	request_data["request_id"] = current_request_id;
	
	String request_body = JSON::stringify(request_data);
	
    PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	// Extract stop endpoint from main API endpoint
	String stop_endpoint = api_endpoint;
	if (stop_endpoint.ends_with("/chat")) {
		stop_endpoint = stop_endpoint.trim_suffix("/chat") + "/stop";
	} else {
		// Fallback - just add /stop to the base
		stop_endpoint += "/stop";
	}
	
	print_line("AI Chat: Sending stop request to: " + stop_endpoint);
	stop_http_request->request(stop_endpoint, headers, HTTPClient::METHOD_POST, request_body);
}
void AIChatDock::_on_stop_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	String response_body = String::utf8((const char *)p_body.ptr(), p_body.size());
	print_line("AI Chat: Stop request completed - Result: " + String::num(p_result) + ", Code: " + String::num(p_code) + ", Body: " + response_body);
	
	if (p_code == 200) {
		print_line("AI Chat: Stop request successful");
	} else {
		print_line("AI Chat: Stop request failed with code: " + String::num(p_code));
	}
}

void AIChatDock::_on_edit_message_pressed(int p_message_index) {
    print_line("AI Chat: Edit button pressed for message index: " + String::num(p_message_index));

    Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
    if (p_message_index < 0 || p_message_index >= chat_history.size()) {
        print_line("AI Chat: Invalid message index for editing: " + String::num(p_message_index));
        return;
    }

    ChatMessage &message = chat_history.write[p_message_index];
    if (message.role != "user") {
        print_line("AI Chat: Can only edit user messages, but got role: " + message.role);
        return;
    }

    // Replace only the selected message panel to avoid O(n) rebuilds and extra allocations.
    if (!chat_container) {
        return;
    }
    Node *node = chat_container->find_child("message_panel_" + String::num_int64(p_message_index), true, false);
    PanelContainer *old_panel = Object::cast_to<PanelContainer>(node);
    if (!old_panel) {
        // Fallback to rebuild if we cannot find the targeted panel, but keep proper panel naming and structure.
        for (int i = 0; i < chat_container->get_child_count(); i++) {
            Node *child = chat_container->get_child(i);
            child->queue_free();
        }
        for (int i = 0; i < chat_history.size(); i++) {
            if (i == p_message_index) {
                PanelContainer *panel = _build_edit_message_panel(chat_history[i], i);
                chat_container->add_child(panel);
            } else {
                _create_message_bubble(chat_history[i], i);
            }
        }
        call_deferred("_scroll_to_bottom");
        return;
    }

    // Build the edit panel and swap it in-place preserving index order.
    PanelContainer *edit_panel = _build_edit_message_panel(message, p_message_index);
    if (!edit_panel) {
        return;
    }
    Node *parent = old_panel->get_parent();
    int old_index = old_panel->get_index();
    parent->remove_child(old_panel);
    old_panel->queue_free();
    parent->add_child(edit_panel);
    parent->move_child(edit_panel, old_index);

    call_deferred("_scroll_to_bottom");
}
void AIChatDock::_create_edit_message_bubble(const AIChatDock::ChatMessage &p_message, int p_message_index) {
	if (chat_container == nullptr) {
		return;
	}
	
	PanelContainer *message_panel = memnew(PanelContainer);
	
	// Add spacing before each message for cleaner layout
	if (chat_container->get_child_count() > 0) {
		Control *spacer = memnew(Control);
		spacer->set_custom_minimum_size(Size2(0, 8)); // 8px gap between messages
		chat_container->add_child(spacer);
	}
	
	chat_container->add_child(message_panel);
	message_panel->set_visible(true);

	// User message styling (editing mode)
	Ref<StyleBoxFlat> panel_style = memnew(StyleBoxFlat);
	panel_style->set_content_margin_all(12);
	panel_style->set_corner_radius_all(8);
	panel_style->set_bg_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.15)); // Brighter to indicate edit mode
	panel_style->set_border_width_all(2);
	panel_style->set_border_color(get_theme_color(SNAME("accent_color"), SNAME("Editor"))); // Accent border for edit mode
	message_panel->add_theme_style_override("panel", panel_style);

	// Content
	VBoxContainer *message_vbox = memnew(VBoxContainer);
	message_panel->add_child(message_vbox);

	// Role label
	Label *role_label = memnew(Label);
	role_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
	role_label->set_text("User (Editing)");
	role_label->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	message_vbox->add_child(role_label);

	// Editable text field
	TextEdit *edit_field = memnew(TextEdit);
	edit_field->set_text(p_message.content);
	edit_field->set_custom_minimum_size(Size2(0, 100));
	edit_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	// TextEdit wrapping might have different method name in Godot 4
	message_vbox->add_child(edit_field);

	// Buttons
	HBoxContainer *button_container = memnew(HBoxContainer);
	message_vbox->add_child(button_container);

	Button *send_button = memnew(Button);
	send_button->set_text("Send");
	send_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Play"), SNAME("EditorIcons")));
	// Store the edit field reference in the button's metadata so we can get the text when pressed
	send_button->set_meta("edit_field", edit_field);
	send_button->set_meta("message_index", p_message_index);
	send_button->connect("pressed", callable_mp(this, &AIChatDock::_on_edit_send_button_pressed).bind(send_button));
	button_container->add_child(send_button);
	
	// Connect keyboard input for Enter key support
	edit_field->connect("gui_input", callable_mp(this, &AIChatDock::_on_edit_field_gui_input).bind(send_button));

	Button *cancel_button = memnew(Button);
	cancel_button->set_text("Cancel");
	cancel_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Stop"), SNAME("EditorIcons")));
	cancel_button->connect("pressed", callable_mp(this, &AIChatDock::_on_edit_message_cancel_pressed).bind(p_message_index));
	button_container->add_child(cancel_button);
	
	// Focus the edit field
	edit_field->grab_focus();
}
PanelContainer *AIChatDock::_build_edit_message_panel(const AIChatDock::ChatMessage &p_message, int p_message_index) {
    PanelContainer *message_panel = memnew(PanelContainer);
    if (p_message_index >= 0) {
        message_panel->set_name("message_panel_" + String::num_int64(p_message_index));
    }

    Ref<StyleBoxFlat> panel_style = memnew(StyleBoxFlat);
    panel_style->set_content_margin_all(12);
    panel_style->set_corner_radius_all(8);
    panel_style->set_bg_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.15));
    panel_style->set_border_width_all(2);
    panel_style->set_border_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")));
    message_panel->add_theme_style_override("panel", panel_style);

    VBoxContainer *message_vbox = memnew(VBoxContainer);
    message_panel->add_child(message_vbox);

    Label *role_label = memnew(Label);
    role_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
    role_label->set_text("User (Editing)");
    role_label->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
    message_vbox->add_child(role_label);

    TextEdit *edit_field = memnew(TextEdit);
    edit_field->set_text(p_message.content);
    edit_field->set_custom_minimum_size(Size2(0, 100));
    edit_field->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    message_vbox->add_child(edit_field);

    HBoxContainer *button_container = memnew(HBoxContainer);
    message_vbox->add_child(button_container);

    Button *send_button = memnew(Button);
    send_button->set_text("Send");
    send_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Play"), SNAME("EditorIcons")));
    send_button->set_meta("edit_field", edit_field);
    send_button->set_meta("message_index", p_message_index);
    send_button->connect("pressed", callable_mp(this, &AIChatDock::_on_edit_send_button_pressed).bind(send_button));
    button_container->add_child(send_button);

    edit_field->connect("gui_input", callable_mp(this, &AIChatDock::_on_edit_field_gui_input).bind(send_button));

    Button *cancel_button = memnew(Button);
    cancel_button->set_text("Cancel");
    cancel_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Stop"), SNAME("EditorIcons")));
    cancel_button->connect("pressed", callable_mp(this, &AIChatDock::_on_edit_message_cancel_pressed).bind(p_message_index));
    button_container->add_child(cancel_button);

    edit_field->grab_focus();
    return message_panel;
}

void AIChatDock::_on_edit_send_button_pressed(Button *p_button) {
	// Get the edit field and message index from the button's metadata
	TextEdit *edit_field = Object::cast_to<TextEdit>(p_button->get_meta("edit_field"));
	int message_index = p_button->get_meta("message_index");
	
	if (!edit_field) {
		print_line("AI Chat: Error - could not find edit field from button metadata");
		return;
	}
	
	String new_content = edit_field->get_text();
	_on_edit_message_send_pressed(message_index, new_content);
}

void AIChatDock::_on_edit_message_send_pressed(int p_message_index, const String &p_new_content) {
	Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
	if (p_message_index < 0 || p_message_index >= chat_history.size()) {
		print_line("AI Chat: Invalid message index for sending edit: " + String::num(p_message_index));
		return;
	}
	
	String trimmed_content = p_new_content.strip_edges();
	if (trimmed_content.is_empty()) {
		print_line("AI Chat: Cannot send empty message");
		return;
	}
	
	print_line("AI Chat: Sending edited message at index " + String::num(p_message_index) + ": " + trimmed_content);
	
	// Update the message content
	chat_history.write[p_message_index].content = trimmed_content;
	chat_history.write[p_message_index].timestamp = _get_timestamp();
	
	// Truncate conversation at this point (remove everything after this message)
	chat_history.resize(p_message_index + 1);
	
	// Update conversation timestamp
	if (current_conversation_index >= 0) {
		conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
	}
	
	// Remove all messages after the edited one since we truncated the conversation
	if (chat_container) {
		// First, find the index of the edited message panel
		int edited_panel_ui_index = -1;
		for (int i = 0; i < chat_container->get_child_count(); i++) {
			Node *child = chat_container->get_child(i);
			PanelContainer *panel = Object::cast_to<PanelContainer>(child);
			if (panel) {
				String panel_name = panel->get_name();
				if (panel_name == "message_panel_" + String::num_int64(p_message_index)) {
					edited_panel_ui_index = i;
					break;
				}
			}
		}
		
		// Remove all nodes after the edited message (including spacers)
		if (edited_panel_ui_index >= 0) {
			for (int i = chat_container->get_child_count() - 1; i > edited_panel_ui_index; i--) {
				Node *child = chat_container->get_child(i);
				chat_container->remove_child(child);
				child->queue_free();
			}
		}
		
		// Now replace the edit panel with a normal message bubble
		Node *existing = chat_container->find_child("message_panel_" + String::num_int64(p_message_index), true, false);
		PanelContainer *old_panel = Object::cast_to<PanelContainer>(existing);
		if (old_panel) {
			// Build a new message panel without using _create_message_bubble (which adds spacers)
			PanelContainer *new_panel = memnew(PanelContainer);
			new_panel->set_name("message_panel_" + String::num_int64(p_message_index));
			
			// Copy the structure from _create_message_bubble but without spacer logic
			_build_message_content(new_panel, chat_history[p_message_index], p_message_index);
			
			// Replace the old panel with the new one at the same position
			Node *parent = old_panel->get_parent();
			int old_index = old_panel->get_index();
			parent->remove_child(old_panel);
			old_panel->queue_free();
			parent->add_child(new_panel);
			parent->move_child(new_panel, old_index);
		}
	}
	
	// Reset the assistant message label so streaming can create a new one
	current_assistant_message_label = nullptr;
	
	// Queue a delayed save instead of immediate save
	_queue_delayed_save();
	
	// Send the request as if it's a new message
	is_waiting_for_response = true;
	_update_ui_state();
	call_deferred("_process_send_request_async");
}

void AIChatDock::_on_edit_field_gui_input(const Ref<InputEvent> &p_event, Button *p_send_button) {
	// Handle Enter key to send the edited message
	Ref<InputEventKey> key_event = p_event;
	if (key_event.is_valid() && key_event->is_pressed()) {
		if (key_event->get_keycode() == Key::ENTER || key_event->get_keycode() == Key::KP_ENTER) {
			// Check if Shift is not pressed (Shift+Enter should add new line)
			if (!key_event->is_shift_pressed()) {
				print_line("AI Chat: Enter key pressed in edit field, triggering send");
				// Trigger the send button
				p_send_button->emit_signal("pressed");
				// Accept the event to prevent further processing
				get_viewport()->set_input_as_handled();
			}
		} else if (key_event->get_keycode() == Key::ESCAPE) {
			// Escape key cancels editing
			print_line("AI Chat: Escape key pressed in edit field, cancelling edit");
			// Find the message index from the send button's metadata
			int message_index = p_send_button->get_meta("message_index");
			_on_edit_message_cancel_pressed(message_index);
			get_viewport()->set_input_as_handled();
		}
	}
}

void AIChatDock::_on_edit_message_cancel_pressed(int p_message_index) {
	print_line("AI Chat: Cancelled editing message at index " + String::num(p_message_index));
	
	// Replace edit panel back to normal message bubble, avoid full rebuild
	Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
	if (chat_container) {
		Node *existing = chat_container->find_child("message_panel_" + String::num_int64(p_message_index), true, false);
		PanelContainer *old_panel = Object::cast_to<PanelContainer>(existing);
		if (old_panel) {
			// Build a new message panel without using _create_message_bubble (which adds spacers)
			PanelContainer *new_panel = memnew(PanelContainer);
			new_panel->set_name("message_panel_" + String::num_int64(p_message_index));
			
			// Copy the structure from _create_message_bubble but without spacer logic
			_build_message_content(new_panel, chat_history[p_message_index], p_message_index);
			
			// Replace the old panel with the new one at the same position
			Node *parent = old_panel->get_parent();
			int old_index = old_panel->get_index();
			parent->remove_child(old_panel);
			old_panel->queue_free();
			parent->add_child(new_panel);
			parent->move_child(new_panel, old_index);
		} else {
			for (int i = 0; i < chat_container->get_child_count(); i++) {
				Node *child = chat_container->get_child(i);
				child->queue_free();
			}
			_rebuild_conversation_ui(chat_history);
		}
	}
}

// Embedding system code moved to ai_chat_dock_embeddings.cpp

// Authentication Implementation

void AIChatDock::_setup_authentication_ui() {
	// User authentication row
	HBoxContainer *auth_container = memnew(HBoxContainer);
	add_child(auth_container);
	
	Label *auth_label = memnew(Label);
	auth_label->set_text("User:");
	auth_container->add_child(auth_label);
	
	user_status_label = memnew(Label);
	user_status_label->set_text("Not logged in");
	user_status_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	auth_container->add_child(user_status_label);
	
	login_button = memnew(Button);
	login_button->set_text("Login");
	login_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Key"), SNAME("EditorIcons")));
	login_button->connect("pressed", callable_mp(this, &AIChatDock::_on_login_button_pressed));
	auth_container->add_child(login_button);
	
	// Create HTTP request for authentication
	auth_request = memnew(HTTPRequest);
	add_child(auth_request);
	auth_request->connect("request_completed", callable_mp(this, &AIChatDock::_on_auth_request_completed));

	// Create HTTP request for providers discovery
	auth_providers_request = memnew(HTTPRequest);
	add_child(auth_providers_request);
	auth_providers_request->connect("request_completed", callable_mp(this, &AIChatDock::_on_auth_providers_request_completed));
}

void AIChatDock::_on_login_button_pressed() {
	if (_is_user_authenticated()) {
		// User is logged in, offer logout
		_logout_user();
		return;
	}
	
    // Show a menu of providers, anchored to the login button
    // Fetch provider availability from backend to hide unavailable options
    String providers_url = api_endpoint.replace("/chat", "/auth/providers");
    Error perr = auth_providers_request->request(providers_url);
    if (perr != OK) {
        print_line("AI Chat: Failed to fetch providers; defaulting to all options");
    }
    PopupMenu *providers = memnew(PopupMenu);
    add_child(providers);
    int id_google = 1;
    int id_guest = 2;
    providers->add_icon_item(get_theme_icon(SNAME("Key"), SNAME("EditorIcons")), "Sign in with Google", id_google);
    providers->add_separator();
    providers->add_icon_item(get_theme_icon(SNAME("Key"), SNAME("EditorIcons")), "Continue as Guest", id_guest);
    providers->set_name("auth_provider_menu");
    providers->connect("id_pressed", callable_mp(this, &AIChatDock::_on_auth_provider_selected));
    // Popup directly under the login button using parent-anchored rect
    Rect2i parent_rect(Point2i((int)login_button->get_global_position().x, (int)login_button->get_global_position().y),
                       Size2i((int)login_button->get_size().x, (int)login_button->get_size().y));
    parent_rect.position.y += parent_rect.size.y;
    providers->popup_on_parent(parent_rect);
}
void AIChatDock::_on_auth_providers_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
    if (p_code != 200) {
        return;
    }
    String text = String::utf8((const char *)p_body.ptr(), p_body.size());
    JSON json;
    if (json.parse(text) != OK) {
        return;
    }
    Dictionary resp = json.get_data();
    if (!resp.get("success", false)) {
        return;
    }
    Dictionary providers = resp.get("providers", Dictionary());
    bool has_google = providers.get("google", false);
    bool has_github = providers.get("github", false);
    bool has_ms = providers.get("microsoft", false);
    // Suppress unused variable warnings - these may be used in future functionality
    (void)has_google;
    (void)has_github;
    // If Microsoft is not configured, avoid letting user select it by clearing pending_provider when selected
    if (pending_login_provider == "microsoft" && !has_ms) {
        // Nothing to do here now; selection guarded at build time of menu ideally
    }
}

void AIChatDock::_on_auth_dialog_action(const StringName &p_action) {
	// This method is kept for compatibility but no longer used
	// Login polling handles authentication automatically
}

void AIChatDock::_on_auth_provider_selected(int p_id) {
    String provider;
    if (p_id == 1) {
        provider = "google";
    } else if (p_id == 2) {
        provider = "guest";
    } else {
        return;
    }
    pending_login_provider = provider;
    if (provider == "guest") {
        // Call backend to create/get guest session, then reflect in UI
        current_user_id = "guest:" + get_machine_id();
        current_user_name = "Guest";
        auth_token = ""; // Guest flow in backend does not need token in headers
        _update_user_status();
        print_line("AI Chat: Using guest session");
        return;
    } else {
        String machine_id = OS::get_singleton()->get_unique_id();
        String url = AILoginHelper::build_auth_login_url(api_endpoint, machine_id, provider);
        OS::get_singleton()->shell_open(url);
    }

    // Start polling silently; if Microsoft was chosen, require provider and disallow guest
    user_status_label->set_text("Waiting for login...");
    _start_login_polling();
}

void AIChatDock::_check_authentication_status() {
	// Check with backend if user is authenticated
	String auth_check_url = api_endpoint.replace("/chat", "/auth/status");
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	Dictionary data;
	data["machine_id"] = OS::get_singleton()->get_unique_id();
    if (!pending_login_provider.is_empty()) {
        data["require_provider"] = pending_login_provider;
        // While polling for a specific provider, do not accept guest fallback
        data["allow_guest"] = false;
    }
	String json_data = JSON::stringify(data);
	
	Error err = auth_request->request(auth_check_url, headers, HTTPClient::METHOD_POST, json_data);
	if (err != OK) {
		print_line("AI Chat: Failed to check authentication status: " + String::num_int64(err));
	}
}
void AIChatDock::_on_auth_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
	
	print_line("AI Chat: 📡 Auth request completed - Result: " + String::num_int64(p_result) + ", Code: " + String::num_int64(p_code));
	
	if (p_code != 200) {
		print_line("AI Chat: ❌ Authentication request failed with code " + String::num_int64(p_code) + ": " + response_text);
		// If it's a connection error, the backend might not be running
		if (p_code == 0) {
			print_line("AI Chat: 🔌 Backend server might not be running. Please start the backend server.");
		}
		return;
	}
	
	JSON json;
	Error parse_err = json.parse(response_text);
	if (parse_err != OK) {
		print_line("AI Chat: Failed to parse authentication response: " + response_text);
		return;
	}
	
	Dictionary response = json.get_data();
	bool success = response.get("success", false);
	
	if (success && response.has("user")) {
		Dictionary user_data = response.get("user", Dictionary());
		current_user_id = user_data.get("id", "");
		current_user_name = user_data.get("name", "");
		auth_token = response.get("token", "");
		
		// Save authentication
		EditorSettings::get_singleton()->set_setting("ai_chat/auth_token", auth_token);
		EditorSettings::get_singleton()->set_setting("ai_chat/user_id", current_user_id);
		EditorSettings::get_singleton()->set_setting("ai_chat/user_name", current_user_name);
		
		print_line("AI Chat: ✅ User authenticated successfully: " + current_user_name);
		
        // Stop login polling if it was running
        _stop_login_polling();
        pending_login_provider = "";
		
		// Force UI update
        _update_user_status();
		
		// Force comprehensive project indexing
		_ensure_project_indexing();
	} else {
		// Authentication failed - clear any invalid saved credentials
		current_user_id = "";
		current_user_name = "";
		auth_token = "";
		
		// Remove invalid credentials from settings
		if (EditorSettings::get_singleton()->has_setting("ai_chat/auth_token")) {
			EditorSettings::get_singleton()->erase("ai_chat/auth_token");
			EditorSettings::get_singleton()->erase("ai_chat/user_id");
			EditorSettings::get_singleton()->erase("ai_chat/user_name");
			print_line("AI Chat: Cleared invalid saved credentials");
		}
		
		_update_user_status();
		print_line("AI Chat: Authentication failed: " + String(response.get("error", "Unknown error")));
	}
}
void AIChatDock::_update_user_status() {
	if (_is_user_authenticated()) {
		user_status_label->set_text(current_user_name);
		login_button->set_text("Logout");
		login_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Unlock"), SNAME("EditorIcons")));
	} else {
		user_status_label->set_text("Not logged in");
		login_button->set_text("Login");
		login_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Key"), SNAME("EditorIcons")));
	}
}

void AIChatDock::_logout_user() {
	// Clear authentication data
	current_user_id = "";
	current_user_name = "";
	auth_token = "";
	
	// Reset embedding system state
	embedding_system_initialized = false;
	initial_indexing_done = false;
	
	// Remove from settings
	EditorSettings::get_singleton()->erase("ai_chat/auth_token");
	EditorSettings::get_singleton()->erase("ai_chat/user_id");
	EditorSettings::get_singleton()->erase("ai_chat/user_name");
	
	_update_user_status();
	print_line("AI Chat: User logged out - embedding system reset");
}
bool AIChatDock::_is_user_authenticated() const {
	// Treat guest sessions as authenticated for indexing and chat.
	if (current_user_id.begins_with("guest:")) {
		return true;
	}
	return !current_user_id.is_empty() && !auth_token.is_empty();
}

void AIChatDock::_auto_verify_saved_credentials() {
	print_line("AI Chat: 🔍 Checking for saved authentication credentials...");
	
	// Check if we have saved authentication credentials
	if (EditorSettings::get_singleton()->has_setting("ai_chat/auth_token")) {
		String saved_token = EditorSettings::get_singleton()->get_setting("ai_chat/auth_token");
		String saved_user_id = EditorSettings::get_singleton()->get_setting("ai_chat/user_id");
		String saved_user_name = EditorSettings::get_singleton()->get_setting("ai_chat/user_name");
		
		if (!saved_token.is_empty() && !saved_user_id.is_empty()) {
			// Set credentials for verification
			auth_token = saved_token;
			current_user_id = saved_user_id;
			current_user_name = saved_user_name;
			
			print_line("AI Chat: 🔐 Auto-verifying saved authentication for user: " + saved_user_name);
			
			// Show temporary status while verifying
			user_status_label->set_text("Verifying login...");
			
			// Verify with backend
			_check_authentication_status();
		} else {
			print_line("AI Chat: Saved credentials found but incomplete");
		}
	} else {
		print_line("AI Chat: No saved authentication credentials found");
	}
}

void AIChatDock::_start_login_polling() {
	login_poll_attempts = 0;
	login_poll_max_attempts = 30; // Poll for 30 seconds (every 1 second)
	
	// Create timer if it doesn't exist
	if (!login_poll_timer) {
		login_poll_timer = memnew(Timer);
		add_child(login_poll_timer);
		login_poll_timer->connect("timeout", callable_mp(this, &AIChatDock::_poll_login_status));
	}
	
	// Start polling every 1 second
	login_poll_timer->set_wait_time(1.0);
	login_poll_timer->set_one_shot(false);
	login_poll_timer->start();
	
	print_line("AI Chat: 🔄 Started automatic login polling");
}

void AIChatDock::_poll_login_status() {
	login_poll_attempts++;
	
	if (login_poll_attempts > login_poll_max_attempts) {
		// Stop polling after max attempts
		login_poll_timer->stop();
		user_status_label->set_text("Login timeout - try again");
		print_line("AI Chat: ⏰ Login polling timed out");
		return;
	}
	
	// Check authentication status
	_check_authentication_status();
}

void AIChatDock::_stop_login_polling() {
	if (login_poll_timer && login_poll_timer->is_connected("timeout", callable_mp(this, &AIChatDock::_poll_login_status))) {
		login_poll_timer->stop();
		print_line("AI Chat: ✅ Stopped login polling");
	}
}

void AIChatDock::_on_index_button_pressed() {
    print_line("AI Chat: ▶️ Index button pressed");
    _ensure_project_indexing();
}
void AIChatDock::_ensure_project_indexing() {
	print_line("AI Chat: 🔄 Ensuring project indexing starts...");
	print_line("AI Chat: 🔍 DEBUG - embedding_system_initialized: " + String(embedding_system_initialized ? "true" : "false"));
	print_line("AI Chat: 🔍 DEBUG - initial_indexing_done: " + String(initial_indexing_done ? "true" : "false"));
	
    // If not authenticated, fallback to guest automatically
    if (!_is_user_authenticated()) {
        current_user_id = "guest:" + get_machine_id();
        current_user_name = "Guest";
        auth_token = "";
        _update_user_status();
        print_line("AI Chat: ℹ️ No auth detected; indexing as guest");
    }
	
	// Initialize embedding system if needed
	if (!embedding_system_initialized) {
		print_line("AI Chat: 📝 Initializing embedding system...");
		_initialize_embedding_system();
	} else if (!initial_indexing_done) {
		print_line("AI Chat: 📝 Embedding system initialized but indexing not done. Checking index status...");
		_check_index_status_and_start_if_needed();
	} else {
		print_line("AI Chat: ✅ Embedding system already initialized and indexing complete. Skipping unnecessary re-index.");
		return;
	}
}

void AIChatDock::_process_send_request_async() {
	// This runs in the next frame, keeping UI responsive
	
	// Update conversation timestamp and title (heavier operations)
	if (current_conversation_index >= 0) {
		conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
		// Update title if it's still "New Conversation"
		if (conversations[current_conversation_index].title == "New Conversation") {
			conversations.write[current_conversation_index].title = _generate_conversation_title(_get_current_chat_history());
		}
		_update_conversation_dropdown();
	}
	
	// Now send the actual request (this is the heavy operation)
	_send_chat_request();
}
void AIChatDock::_save_conversations_async() {
	// Start the async save to prevent UI blocking during file operations
	call_deferred("_save_conversations_chunked", 0);
}

void AIChatDock::_save_conversations_to_disk(const String &p_json_data) {
	// This runs in the next frame, preventing blocking of current frame
    Error err;
    // Safe write via temp + rename to avoid corruption
    String final_path = conversations_file_path;
    String base_dir = final_path.get_base_dir();
    String final_name = final_path.get_file();
    String temp_name = final_name + ".tmp";
    String temp_path = base_dir.path_join(temp_name);

    // Ensure destination directory exists
    {
        Ref<DirAccess> da_mk = DirAccess::create_for_path(base_dir);
        if (da_mk.is_valid() && !da_mk->dir_exists(base_dir)) {
            da_mk->make_dir_recursive(base_dir);
        }
    }
    // Write to temp
    {
        Ref<FileAccess> tmp = FileAccess::open(temp_path, FileAccess::WRITE, &err);
        if (err != OK) {
            print_line("AI Chat: Failed to open temp file for save: " + String::num_int64(err));
            return;
        }
        tmp->store_string(p_json_data);
        tmp->close();
    }

    // Rename temp to final
    Ref<DirAccess> da = DirAccess::open(base_dir);
    if (da.is_valid()) {
        if (da->file_exists(final_name)) {
            da->remove(final_name);
        }
        da->rename(temp_name, final_name);
    } else {
        // Fallback: write directly if Directory open failed
        Ref<FileAccess> file = FileAccess::open(final_path, FileAccess::WRITE, &err);
        if (err != OK) {
            print_line("AI Chat: Failed to save conversations file: " + String::num_int64(err));
            return;
        }
        file->store_string(p_json_data);
        file->close();
    }
	// print_line("AI Chat: Conversations saved successfully");
}

void AIChatDock::_save_conversations_chunked(int p_start_index) {
	// Process conversations in chunks of 5 to avoid blocking UI
	const int CHUNK_SIZE = 5;
	int end_index = MIN(p_start_index + CHUNK_SIZE, conversations.size());
	
	// If this is the first chunk, initialize the array
	if (p_start_index == 0) {
		_chunked_conversations_array.clear();
	}
	
    // Process this chunk
	for (int i = p_start_index; i < end_index; i++) {
		const AIChatDock::Conversation &conv = conversations[i];
		Dictionary conv_dict;
		conv_dict["id"] = conv.id;
		conv_dict["title"] = conv.title;
		conv_dict["created_timestamp"] = conv.created_timestamp;
		conv_dict["last_modified_timestamp"] = conv.last_modified_timestamp;
		
		Array messages_array;
        for (int j = 0; j < conv.messages.size(); j++) {
            const ChatMessage &msg = conv.messages[j];
            Dictionary msg_dict;
            msg_dict["role"] = msg.role;
            msg_dict["content"] = msg.content;
            msg_dict["timestamp"] = msg.timestamp;
            msg_dict["tool_calls"] = msg.tool_calls;
            msg_dict["tool_call_id"] = msg.tool_call_id;
            msg_dict["name"] = msg.name;
            msg_dict["tool_results"] = msg.tool_results;

            // Attached files
            Array attached_files_array;
            for (int k = 0; k < msg.attached_files.size(); k++) {
                const AttachedFile &file = msg.attached_files[k];
                Dictionary file_dict;
                file_dict["path"] = file.path;
                file_dict["name"] = file.name;
                file_dict["content"] = file.content;
                file_dict["is_image"] = file.is_image;
                file_dict["mime_type"] = file.mime_type;
                file_dict["base64_data"] = file.base64_data;
                // v2 array sizes
                Array os;
                os.push_back(file.original_size.x);
                os.push_back(file.original_size.y);
                file_dict["original_size"] = os;
                Array ds;
                ds.push_back(file.display_size.x);
                ds.push_back(file.display_size.y);
                file_dict["display_size"] = ds;
                // v1 compatibility
                file_dict["original_size_x"] = file.original_size.x;
                file_dict["original_size_y"] = file.original_size.y;
                file_dict["display_size_x"] = file.display_size.x;
                file_dict["display_size_y"] = file.display_size.y;
                file_dict["was_downsampled"] = file.was_downsampled;
                file_dict["is_node"] = file.is_node;
                file_dict["node_path"] = file.node_path;
                file_dict["node_type"] = file.node_type;
                attached_files_array.push_back(file_dict);
            }
            msg_dict["attached_files"] = attached_files_array;

            messages_array.push_back(msg_dict);
        }
		conv_dict["messages"] = messages_array;
		_chunked_conversations_array.push_back(conv_dict);
	}
	
	print_line("AI Chat: Processed conversation save chunk " + String::num_int64(p_start_index) + "-" + String::num_int64(end_index-1) + " of " + String::num_int64(conversations.size()));
	
	// If more chunks to process, defer the next chunk
	if (end_index < conversations.size()) {
		call_deferred("_save_conversations_chunked", end_index);
		return;
	}
	
	// All chunks processed, now finalize the save
	call_deferred("_finalize_conversations_save");
}

void AIChatDock::_finalize_conversations_save() {
    Dictionary data;
    data["version"] = 2;
	data["conversations"] = _chunked_conversations_array;
	
	Ref<JSON> json;
	json.instantiate();
	String json_string = json->stringify(data, "  ");
	
	_save_conversations_to_disk(json_string);
	
	// Clear the chunked array to free memory
	_chunked_conversations_array.clear();
}
void AIChatDock::_queue_delayed_save() {
	// If a save is already scheduled, do nothing
	if (save_pending) {
		return;
	}
	
	save_pending = true;
	// Start (or restart) the save timer for 3 seconds
	if (save_timer) {
		save_timer->stop();
		save_timer->start(3.0); // wait 3 seconds of idle time before saving
	}
}

void AIChatDock::_execute_delayed_save() {
	if (!save_pending) {
		return; // No save pending
	}
	
	if (save_thread_busy) {
        // A save is already running. Reschedule shortly to coalesce further changes
        if (save_timer) {
            save_timer->stop();
            save_timer->start(0.5); // retry soon after current save finishes
        }
        return;
	}
	
	save_pending = false;
	save_thread_busy = true;
	
	// Create a data structure to pass both the snapshot and the instance
	struct SaveData {
		Vector<Conversation> *snapshot;
		AIChatDock *instance;
		String file_path;
	};
	
	SaveData *save_data = memnew(SaveData);
	save_data->snapshot = memnew(Vector<Conversation>(conversations));
	save_data->instance = this;
	save_data->file_path = conversations_file_path;
	
	save_thread = memnew(Thread);
	save_thread->start(_background_save, save_data);
	
	print_line("AI Chat: Started background conversation save");
}
void AIChatDock::_background_save(void *p_data_ptr) {
	struct SaveData {
		Vector<Conversation> *snapshot;
		AIChatDock *instance;
		String file_path;
	};
	
	SaveData *save_data = reinterpret_cast<SaveData*>(p_data_ptr);
	Vector<Conversation> &snapshot = *save_data->snapshot;
	
    // Build the data structure (heavy JSON stringify)
    Dictionary data;
    data["version"] = 2; // schema version for future migrations
	Array conv_array;
	
	for (const Conversation &conv : snapshot) {
		Dictionary conv_dict;
		conv_dict["id"] = conv.id;
		conv_dict["title"] = conv.title;
		conv_dict["created_timestamp"] = conv.created_timestamp;
		conv_dict["last_modified_timestamp"] = conv.last_modified_timestamp;
		
		Array messages_array;
		for (const ChatMessage &msg : conv.messages) {
			Dictionary msg_dict;
			msg_dict["role"] = msg.role;
			msg_dict["content"] = msg.content;
			msg_dict["timestamp"] = msg.timestamp;
			msg_dict["tool_calls"] = msg.tool_calls;
			msg_dict["tool_call_id"] = msg.tool_call_id;
			msg_dict["name"] = msg.name;
            msg_dict["tool_results"] = msg.tool_results;
			
            // Handle attached files
			Array files_array;
			for (const AttachedFile &file : msg.attached_files) {
				Dictionary file_dict;
				file_dict["path"] = file.path;
				file_dict["name"] = file.name;
				file_dict["content"] = file.content;
				file_dict["is_image"] = file.is_image;
				file_dict["mime_type"] = file.mime_type;
				file_dict["base64_data"] = file.base64_data;
                // Use array-based sizes for v2, loader supports both v1 (x/y) and v2 (arrays)
                Array original_size_array;
                original_size_array.push_back(file.original_size.x);
                original_size_array.push_back(file.original_size.y);
                file_dict["original_size"] = original_size_array;
                Array display_size_array;
                display_size_array.push_back(file.display_size.x);
                display_size_array.push_back(file.display_size.y);
                file_dict["display_size"] = display_size_array;
                // Also include x/y keys for backward compatibility with older loaders
                file_dict["original_size_x"] = file.original_size.x;
                file_dict["original_size_y"] = file.original_size.y;
                file_dict["display_size_x"] = file.display_size.x;
                file_dict["display_size_y"] = file.display_size.y;
				file_dict["was_downsampled"] = file.was_downsampled;
				file_dict["is_node"] = file.is_node;
				file_dict["node_path"] = file.node_path;
				file_dict["node_type"] = file.node_type;
				files_array.push_back(file_dict);
			}
			msg_dict["attached_files"] = files_array;
			
			messages_array.push_back(msg_dict);
		}
		conv_dict["messages"] = messages_array;
		conv_array.push_back(conv_dict);
	}
	data["conversations"] = conv_array;
	
	// JSON stringify (heavy operation)
	Ref<JSON> json;
	json.instantiate();
	String json_string = json->stringify(data, "  ");
	
    // File I/O (heavy operation) with safe write via temp + rename
    Error err;
    String final_path = save_data->file_path;
    String base_dir = final_path.get_base_dir();
    String final_name = final_path.get_file();
    String temp_name = final_name + ".tmp";
    String temp_path = base_dir.path_join(temp_name);

    // Ensure destination directory exists
    {
        Ref<DirAccess> da_mk = DirAccess::create_for_path(base_dir);
        if (da_mk.is_valid() && !da_mk->dir_exists(base_dir)) {
            da_mk->make_dir_recursive(base_dir);
        }
    }
    // Write to temp
    {
        Ref<FileAccess> tmp = FileAccess::open(temp_path, FileAccess::WRITE, &err);
        if (err == OK) {
            tmp->store_string(json_string);
            tmp->close();
        }
    }
    if (err == OK) {
        // Rename temp to final (best-effort atomic within same dir)
        Ref<DirAccess> da = DirAccess::open(base_dir);
        if (da.is_valid()) {
            // Remove existing file if present (rename should overwrite but be safe cross-platform)
            if (da->file_exists(final_name)) {
                da->remove(final_name);
            }
            da->rename(temp_name, final_name);
        } else {
            // Fallback: write directly if Directory open failed
            Ref<FileAccess> file = FileAccess::open(final_path, FileAccess::WRITE, &err);
            if (err == OK) {
                file->store_string(json_string);
                file->close();
            }
        }
    }
	
	// Signal back to main thread
	AIChatDock *instance = save_data->instance;
	
	// Clean up
	memdelete(save_data->snapshot);
	memdelete(save_data);
	
	// Signal completion to main thread
	if (instance) {
		instance->call_deferred("_on_background_save_finished");
	}
}

void AIChatDock::_on_background_save_finished() {
	if (save_thread) {
		save_thread->wait_to_finish();
		memdelete(save_thread);
		save_thread = nullptr;
	}
	save_thread_busy = false;
	print_line("AI Chat: Background save finished");
    // If changes accumulated during the background save, schedule another save soon
    if (save_pending && save_timer) {
        save_timer->stop();
        save_timer->start(0.25);
    }
}

void AIChatDock::_process_image_attachment_async(const String &p_file_path, const String &p_name, const String &p_mime_type) {
	// This runs in the next frame to prevent UI blocking during image processing
	Ref<Image> image = Image::load_from_file(p_file_path);
	if (image.is_null() || image->is_empty()) {
		print_line("AI Chat: Failed to load image: " + p_file_path);
		return;
	}

	AIChatDock::AttachedFile attached_file;
	attached_file.path = p_file_path;
	// Create unique image ID: timestamp + clean filename
	String clean_name = p_name.get_basename(); // Remove extension
	clean_name = clean_name.to_lower().replace(" ", "_");
	attached_file.name = "img_" + String::num_int64(OS::get_singleton()->get_ticks_msec()) + "_" + clean_name;
	attached_file.is_image = true;
	attached_file.mime_type = p_mime_type;

	Vector2i original_size = Vector2i(image->get_width(), image->get_height());
	attached_file.original_size = original_size;
	
	// Check if image needs to be downsampled (max 1024px on any side)
	const int MAX_DIMENSION = 1024;
	Vector2i target_size = _calculate_downsampled_size(original_size, MAX_DIMENSION);
	
	if (target_size != original_size) {
		attached_file.was_downsampled = true;
		image->resize(target_size.x, target_size.y, Image::INTERPOLATE_LANCZOS);
		
		// Show warning dialog - defer this to ensure UI is responsive
		call_deferred("_show_image_warning_dialog", attached_file.name, original_size, target_size);
	}
	
	attached_file.display_size = target_size;
	
	// Convert to base64 for API transmission
	Vector<uint8_t> buffer;
	if (attached_file.mime_type == "image/jpeg" || attached_file.mime_type == "image/jpg") {
		buffer = image->save_jpg_to_buffer(0.85f); // Good quality JPEG
	} else {
		buffer = image->save_png_to_buffer();
	}
	
	if (buffer.size() == 0) {
		print_line("AI Chat: Failed to encode image: " + p_file_path);
		return;
	}
	
	// Encode to base64
	attached_file.base64_data = CoreBind::Marshalls::get_singleton()->raw_to_base64(buffer);
	
	// Add to attached files and update display
	current_attached_files.push_back(attached_file);
	_update_attached_files_display();
	
	print_line("AI Chat: Successfully processed image: " + p_name + " -> ID: " + attached_file.name);
}

void AIChatDock::_on_input_text_changed() {
	send_button->set_disabled(input_field->get_text().strip_edges().is_empty() || is_waiting_for_response);
}
// --- At-Mention Implementation ---
// This is actually not working rn! :/ 

void AIChatDock::_update_at_mention_popup() {
	String text = input_field->get_text();
	int cursor_pos = input_field->get_caret_column();
	
	int at_pos = text.rfind("@", cursor_pos);
	if (at_pos == -1) {
		at_mention_popup->hide();
		return;
	}
	
	String query = text.substr(at_pos + 1, cursor_pos - at_pos - 1);
	
	// Check for spaces after '@' which would invalidate the mention
	if (query.find(" ") != -1) {
		at_mention_popup->hide();
		return;
	}
	
	// Populate and show the popup
	_populate_at_mention_tree(query);
	
	Point2i popup_pos = input_field->get_screen_position() + Point2i(0, -at_mention_popup->get_size().y);
	at_mention_popup->set_position(popup_pos);
	at_mention_popup->popup();
}

void AIChatDock::_populate_at_mention_tree(const String &p_filter) {
	at_mention_tree->clear();
	TreeItem *root = at_mention_tree->create_item();
	at_mention_tree->set_hide_root(true);
	
	// Use EditorFileSystem to get project files
	EditorFileSystem *fs = EditorFileSystem::get_singleton();
	
	// Recursive function to populate the tree
	_populate_tree_recursive(fs->get_filesystem(), root, p_filter);
}

void AIChatDock::_populate_tree_recursive(EditorFileSystemDirectory *p_dir, TreeItem *p_parent, const String &p_filter) {
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		TreeItem *dir_item = at_mention_tree->create_item(p_parent);
		dir_item->set_text(0, p_dir->get_subdir(i)->get_name());
		dir_item->set_icon(0, get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")));
		_populate_tree_recursive(p_dir->get_subdir(i), dir_item, p_filter);
	}
	
	for (int i = 0; i < p_dir->get_file_count(); i++) {
		String file_name = p_dir->get_file(i);
		if (p_filter.is_empty() || file_name.findn(p_filter) != -1) {
			// Exclude image files
			String ext = file_name.get_extension().to_lower();
			if (ext != "png" && ext != "jpg" && ext != "jpeg" && ext != "gif" && ext != "bmp" && ext != "webp" && ext != "svg") {
				TreeItem *file_item = at_mention_tree->create_item(p_parent);
				file_item->set_text(0, file_name);
				file_item->set_metadata(0, p_dir->get_file_path(i));
				file_item->set_icon(0, get_theme_icon(SNAME("File"), SNAME("EditorIcons")));
			}
		}
	}
}
void AIChatDock::_on_at_mention_item_selected() {
	TreeItem *selected = at_mention_tree->get_selected();
	if (!selected || selected->get_metadata(0).is_null()) {
		return; // It's a directory
	}
	
	String file_path = selected->get_metadata(0);
	String file_name = selected->get_text(0);
	
	// Add the file to attachments
	Vector<String> files_to_add;
	files_to_add.push_back(file_path);
	_on_files_selected(files_to_add);
	
	// Replace the @mention with the file name
	String text = input_field->get_text();
	int cursor_pos = input_field->get_caret_column();
	int at_pos = text.rfind("@", cursor_pos);
	
	String before = text.substr(0, at_pos);
	String after = text.substr(cursor_pos);
	
	input_field->set_text(before + file_name + " " + after);
	input_field->set_caret_column(at_pos + file_name.length() + 1);
	
	at_mention_popup->hide();
}

void AIChatDock::_on_input_field_gui_input(const Ref<InputEvent> &p_event) {
	Ref<InputEventKey> key_event = p_event;
	if (key_event.is_valid() && key_event->is_pressed() && !key_event->is_echo()) {
		if (key_event->get_keycode() == Key::ENTER && !key_event->is_shift_pressed()) {
			_on_send_button_pressed();
			get_viewport()->set_input_as_handled(); // Consume the event to prevent the default newline behavior
		}
	}
}

void AIChatDock::_on_model_selected(int p_index) {
	if (model_dropdown) {
		String selected_model = model_dropdown->get_item_text(p_index);
		// Keep the full model name including [FAST] prefix for backend communication
		model = selected_model;
		print_line("AI Chat: Model selected: '" + selected_model + "' -> internal: '" + model + "'");
		// Save the selected model to editor settings
		EditorSettings::get_singleton()->set_setting("ai_chat/model", model);
	}
}
void AIChatDock::_on_attachment_menu_item_pressed(int p_id) {
	switch (p_id) {
		case 0: // Files
			_on_attach_files_pressed();
			break;
		case 1: // Scene Nodes
			_on_attach_scene_nodes_pressed();
			break;
		case 2: // Current Script
			_on_attach_current_script_pressed();
			break;
		case 3: // Resources
			_on_attach_resources_pressed();
			break;
		case 4: // Re-index Project
			_on_reindex_project_pressed();
			break;
	}
}

void AIChatDock::_on_attach_files_pressed() {
	if (file_dialog) {
		file_dialog->popup_file_dialog();
	}
}

void AIChatDock::_on_attach_scene_nodes_pressed() {
	if (scene_tree_popup && scene_tree) {
		// Populate the scene tree with current scene nodes
		scene_tree->clear();
		Node *current_scene = EditorNode::get_singleton()->get_edited_scene();
		if (current_scene) {
			TreeItem *root = scene_tree->create_item();
			root->set_text(0, current_scene->get_name());
			root->set_icon(0, get_theme_icon(SNAME("PackedScene"), SNAME("EditorIcons")));
			root->set_metadata(0, current_scene->get_path());
			_populate_scene_tree_recursive(current_scene, root);
			root->set_collapsed(false);
		} else {
			TreeItem *root = scene_tree->create_item();
			root->set_text(0, "No scene open");
			root->set_selectable(0, false);
		}
		
		// Position and show the popup
		Rect2 button_rect = attach_button->get_global_rect();
		Vector2 popup_pos = Vector2(button_rect.position.x, button_rect.position.y + button_rect.size.y + 5);
		scene_tree_popup->set_position(popup_pos);
		scene_tree_popup->popup();
	}
}

void AIChatDock::_on_attach_current_script_pressed() {
	_attach_current_script();
}

void AIChatDock::_on_attach_resources_pressed() {
	if (resource_dialog) {
		resource_dialog->popup_file_dialog();
	}
}

void AIChatDock::_on_reindex_project_pressed() {
	print_line("AI Chat: 🔄 Re-index Project requested");
	
	// Show confirmation dialog
	AcceptDialog *confirm_dialog = memnew(AcceptDialog);
	confirm_dialog->set_title("Re-index Project");
	confirm_dialog->set_text("This will clear the existing project index and rebuild it from scratch.\n\nThis is useful after:\n• Updating the AI system\n• Making major project changes\n• Experiencing search issues\n\nProceed with re-indexing?");
	
	// Add custom buttons
	confirm_dialog->add_cancel_button("Cancel");
	Button *confirm_button = confirm_dialog->get_ok_button();
	confirm_button->set_text("Re-index");
	
	confirm_dialog->connect("confirmed", callable_mp(this, &AIChatDock::_perform_project_reindex));
	get_parent()->add_child(confirm_dialog);
	confirm_dialog->popup_centered(Size2(400, 200));
}

void AIChatDock::_perform_project_reindex() {
	print_line("AI Chat: 🗑️ Starting project re-index...");
	
	// Ensure we're authenticated
	if (!_is_user_authenticated()) {
		current_user_id = "guest:" + get_machine_id();
		current_user_name = "Guest";
		auth_token = "";
		_update_user_status();
		print_line("AI Chat: ℹ️ Re-indexing as guest");
	}
	
	_set_embedding_status("Clearing old index", true);
	
	// Call backend reindex endpoint - construct server URL
	String base_url;
	String is_dev = OS::get_singleton()->get_environment("IS_DEV");
	if (is_dev.is_empty()) {
		// Backward-compat with DEV_MODE
		is_dev = OS::get_singleton()->get_environment("DEV_MODE");
	}
	if (!is_dev.is_empty() && is_dev.to_lower() == "true") {
		base_url = "http://127.0.0.1:8000";
	} else {
		base_url = "https://gamechat.simplifine.com";
	}
	
	// Allow override via editor settings or environment variable
	if (EditorSettings::get_singleton() && EditorSettings::get_singleton()->has_setting("ai_chat/base_url")) {
		String override_url = EditorSettings::get_singleton()->get_setting("ai_chat/base_url");
		if (!override_url.is_empty()) {
			base_url = override_url;
		}
	} else if (!OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL").is_empty()) {
		base_url = OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL");
	}
	
	HTTPRequest *reindex_request = memnew(HTTPRequest);
	get_parent()->add_child(reindex_request);
	
	// Set headers
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	headers.push_back("Authorization: Bearer " + auth_token);
	headers.push_back("X-Machine-ID: " + get_machine_id());
	headers.push_back("X-Project-Root: " + _get_project_root_path());
	headers.push_back("X-User-ID: " + current_user_id);
	
	// Send request
	Dictionary request_data;
	request_data["project_root"] = _get_project_root_path();
	
	String json_string = JSON::stringify(request_data);
	Error err = reindex_request->request(base_url + "/reindex_project", headers, HTTPClient::METHOD_POST, json_string);
	
	if (err != OK) {
		print_line("AI Chat: ❌ Failed to start reindex request: " + String::num_int64(err));
		_set_embedding_status("Reindex failed", false);
		reindex_request->queue_free();
		return;
	}
	
	// Connect response handler
	reindex_request->connect("request_completed", callable_mp(this, &AIChatDock::_on_reindex_response));
}

void AIChatDock::_on_reindex_response(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	print_line("AI Chat: 📨 Re-index response received - Result: " + String::num_int64(p_result) + ", Code: " + String::num_int64(p_response_code));
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_response_code != 200) {
		print_line("AI Chat: ❌ Reindex request failed. Code: " + String::num_int64(p_response_code));
		_set_embedding_status("Reindex failed", false);
		return;
	}
	
	// Parse response  
	String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
	print_line("AI Chat: 📄 Reindex response body: " + response_text);
	
	JSON json_parser;
	Error parse_err = json_parser.parse(response_text);
	
	if (parse_err != OK) {
		print_line("AI Chat: ❌ Failed to parse reindex response JSON: " + response_text);
		_set_embedding_status("Reindex failed", false);
		return;
	}
	
	Dictionary response = json_parser.get_data();
	print_line("AI Chat: 📊 Parsed response data, success: " + String(response.get("success", false)));
	
	if (!response.get("success", false)) {
		String error_msg = response.get("error", "Unknown error");
		print_line("AI Chat: ❌ Reindex failed: " + error_msg);
		_set_embedding_status("Reindex failed", false);
		return;
	}
	
	print_line("AI Chat: ✅ Project cleared successfully. Starting fresh indexing...");
	// Do not mark busy here; busy=true would block _perform_initial_indexing()
	_set_embedding_status("Starting reindex", false);
	
	// Reset indexing progression only; keep embedding system initialized
	initial_indexing_done = false;
	
	print_line("AI Chat: ▶️ Triggering _perform_initial_indexing directly...");
	// Start fresh indexing directly
	call_deferred("_perform_initial_indexing");
}

void AIChatDock::_check_index_status_and_start_if_needed() {
	print_line("AI Chat: 🔍 Checking if project is already indexed...");
	
	// Construct server URL
	String base_url = _get_embed_base_url();
	
	HTTPRequest *status_request = memnew(HTTPRequest);
	get_parent()->add_child(status_request);
	
	// Set headers
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	headers.push_back("Authorization: Bearer " + auth_token);
	headers.push_back("X-Machine-ID: " + get_machine_id());
	headers.push_back("X-Project-Root: " + _get_project_root_path());
	headers.push_back("X-User-ID: " + current_user_id);
	
	// Send request
	Dictionary request_data;
	request_data["project_root"] = _get_project_root_path();
	
	String json_string = JSON::stringify(request_data);
	Error err = status_request->request(base_url + "/index_status", headers, HTTPClient::METHOD_POST, json_string);
	
	if (err != OK) {
		print_line("AI Chat: ❌ Failed to check index status. Starting fresh indexing as fallback...");
		status_request->queue_free();
		call_deferred("_perform_initial_indexing");
		return;
	}
	
	// Connect response handler
	status_request->connect("request_completed", callable_mp(this, &AIChatDock::_on_index_status_response));
}

void AIChatDock::_on_index_status_response(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	print_line("AI Chat: 📨 Index status response - Result: " + String::num_int64(p_result) + ", Code: " + String::num_int64(p_response_code));
	
	HTTPRequest *request_node = Object::cast_to<HTTPRequest>(get_children()[-1]);
	if (request_node) {
		request_node->queue_free();
	}
	
	bool should_index = true; // Default to indexing if we can't determine status
	
	if (p_result == HTTPRequest::RESULT_SUCCESS && p_response_code == 200) {
		String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
		
		JSON json_parser;
		Error parse_err = json_parser.parse(response_text);
		
		if (parse_err == OK) {
			Dictionary response = json_parser.get_data();
			bool indexed = response.get("indexed", false);
			
			if (indexed) {
				Dictionary stats = response.get("stats", Dictionary());
				int total_files = stats.get("total_files", 0);
				print_line("AI Chat: ✅ Project already indexed with " + String::num_int64(total_files) + " files. Skipping re-index.");
				
				_set_embedding_status(String::num_int64(total_files) + " files already indexed", false);
				initial_indexing_done = true;
				should_index = false;
			} else {
				print_line("AI Chat: 📝 Project not indexed yet. Starting fresh indexing...");
			}
		} else {
			print_line("AI Chat: ⚠️ Could not parse index status response. Starting indexing as fallback.");
		}
	} else {
		print_line("AI Chat: ⚠️ Index status check failed. Starting indexing as fallback.");
	}
	
	if (should_index) {
		call_deferred("_perform_initial_indexing");
	}
}

void AIChatDock::_on_scene_tree_node_selected() {
	if (!scene_tree) {
		return;
	}
	
	// Get all selected items
	TreeItem *selected = scene_tree->get_selected();
	if (selected) {
		NodePath node_path = selected->get_metadata(0);
		Node *current_scene = EditorNode::get_singleton()->get_edited_scene();
		if (current_scene) {
			Node *node = current_scene->get_node_or_null(node_path);
			if (node) {
				_attach_scene_node(node);
			}
		}
	}
	
	// Hide the popup after selection
	scene_tree_popup->hide();
}
void AIChatDock::_on_files_selected(const Vector<String> &p_files) {
	for (int i = 0; i < p_files.size(); i++) {
		String file_path = p_files[i];
		
		// Check if file is already attached
		bool already_attached = false;
		for (const AttachedFile &existing_file : current_attached_files) {
			if (existing_file.path == file_path) {
				already_attached = true;
				break;
			}
		}
		
		if (!already_attached) {
			AIChatDock::AttachedFile attached_file;
			attached_file.path = file_path;
			if (_is_image_file(file_path)) {
				// Create unique image ID for images
				String clean_name = file_path.get_file().get_basename().to_lower().replace(" ", "_");
				attached_file.name = "img_" + String::num_int64(OS::get_singleton()->get_ticks_msec()) + "_" + clean_name;
			} else {
				attached_file.name = file_path.get_file();
			}
			attached_file.is_image = _is_image_file(file_path);
			attached_file.mime_type = _get_mime_type_from_extension(file_path);
			
			if (attached_file.is_image) {
				// Process image asynchronously to prevent UI blocking
				call_deferred("_process_image_attachment_async", file_path, attached_file.name, attached_file.mime_type);
			} else {
				// Read text file content with safety limits to avoid blowing context
				Error err = OK;
				Ref<FileAccess> f = FileAccess::open(file_path, FileAccess::READ, &err);
				if (err == OK && f.is_valid()) {
					int64_t to_read = MIN(f->get_length(), (int64_t)MAX_TEXT_ATTACHMENT_PREVIEW_BYTES);
					PackedByteArray bytes;
					bytes.resize(to_read);
					int64_t read = f->get_buffer(bytes.ptrw(), to_read);
					f->close();
					String content = String::utf8((const char *)bytes.ptr(), (int)read);
					attached_file.content = _truncate_text_for_context(content);
					bool truncated = f->get_length() > to_read || content.length() > attached_file.content.length();
					if (truncated) {
						attached_file.content += "\n\n…\n[Truncated preview of large file. Only first " + String::num_int64(attached_file.content.length()) + " chars shown.]";
					}
					current_attached_files.push_back(attached_file);
				} else {
					print_line("AI Chat: Failed to read file: " + file_path);
				}
			}
		}
	}
	
	_update_attached_files_display();
}

void AIChatDock::_on_remove_attachment(const String &p_path) {
	for (int i = 0; i < current_attached_files.size(); i++) {
		if (current_attached_files[i].path == p_path) {
			current_attached_files.remove_at(i);
			break;
		}
	}
	_update_attached_files_display();
}

// this now fixed
void AIChatDock::_update_attached_files_display() {
	// Clear existing file display
	for (int i = attached_files_container->get_child_count() - 1; i >= 0; i--) {
		Node *child = attached_files_container->get_child(i);
		child->queue_free();
	}
	
	// Show container only if there are attached files
	attached_files_container->set_visible(current_attached_files.size() > 0);
	
	// Add file chips for each attached file
	for (const AttachedFile &file : current_attached_files) {
		PanelContainer *file_chip = memnew(PanelContainer);
		attached_files_container->add_child(file_chip);
		
		// Modern tab-like styling
		Ref<StyleBoxFlat> chip_style = memnew(StyleBoxFlat);
		// Subtle background with modern styling
		chip_style->set_bg_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
		chip_style->set_border_width_all(1);
		chip_style->set_border_color(get_theme_color(SNAME("dark_color_3"), SNAME("Editor")));
		chip_style->set_corner_radius_all(6); // More modern rounded corners
		chip_style->set_content_margin(SIDE_TOP, 2);
		chip_style->set_content_margin(SIDE_RIGHT, 6);
		chip_style->set_content_margin(SIDE_BOTTOM, 2);
		chip_style->set_content_margin(SIDE_LEFT, 6);
		chip_style->set_shadow_color(Color(0, 0, 0, 0.15));
		chip_style->set_shadow_size(1);
		file_chip->add_theme_style_override("panel", chip_style);
		
		// Create compact tab-like attachment for both images and files
		HBoxContainer *chip_container = memnew(HBoxContainer);
		chip_container->set_custom_minimum_size(Size2(0, 32)); // Fixed height for consistent tabs
		file_chip->add_child(chip_container);
		
		// File type icon (different icons for different file types)
		Label *file_icon = memnew(Label);
		String icon_name = _get_file_type_icon(file);
		file_icon->add_theme_icon_override("icon", get_theme_icon(icon_name, SNAME("EditorIcons")));
		chip_container->add_child(file_icon);
		
		// File name label (truncated if too long)
		Label *file_label = memnew(Label);
		String display_name = file.name;
		if (display_name.length() > 20) {
			display_name = display_name.substr(0, 17) + "...";
		}
		file_label->set_text(display_name);
		file_label->add_theme_font_size_override("font_size", 12);
		file_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")));
		file_label->set_clip_contents(true);
		file_label->set_tooltip_text(file.name);
		chip_container->add_child(file_label);
		
		// Small spacer
		Control *spacer = memnew(Control);
		spacer->set_custom_minimum_size(Size2(4, 0));
		chip_container->add_child(spacer);
		
		// Remove button
		Button *remove_button = memnew(Button);
		remove_button->set_flat(true);
		remove_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Close"), SNAME("EditorIcons")));
		remove_button->add_theme_color_override("icon_normal_color", get_theme_color(SNAME("error_color"), SNAME("Editor")) * Color(1, 1, 1, 0.6));
		remove_button->add_theme_color_override("icon_hover_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
		remove_button->set_tooltip_text("Remove " + file.name);
		remove_button->set_custom_minimum_size(Size2(20, 20));
		remove_button->connect("pressed", callable_mp(this, &AIChatDock::_on_remove_attachment).bind(file.path));
		chip_container->add_child(remove_button);
	}
}

void AIChatDock::_clear_attachments() {
	current_attached_files.clear();
	_update_attached_files_display();
}

// Standard drag and drop support for file attachments and scene nodes
bool AIChatDock::can_drop_data(const Point2 &p_point, const Variant &p_data) const {
	Dictionary drag_data = p_data;
	
	// Debug: Print drag data to understand what we're receiving
	print_line("AI Chat: Drag data received:");
	print_line("  Type: " + Variant::get_type_name(p_data.get_type()));
	if (drag_data.has("type")) {
		print_line("  Drag type: " + String(drag_data["type"]));
	}
	if (drag_data.has("files")) {
		Array files = drag_data["files"];
		print_line("  Files count: " + String::num_int64(files.size()));
		for (int i = 0; i < files.size() && i < 3; i++) {
			print_line("    File " + String::num_int64(i) + ": " + String(files[i]));
		}
	}
	if (drag_data.has("nodes")) {
		Array nodes = drag_data["nodes"];
		print_line("  Nodes count: " + String::num_int64(nodes.size()));
		for (int i = 0; i < nodes.size() && i < 3; i++) {
			print_line("    Node " + String::num_int64(i) + ": " + String(nodes[i]));
		}
	}
	// Print all keys in drag_data
	Array keys = drag_data.keys();
	print_line("  All keys: " + String(Variant(keys)));
	print_line("---");
	
	// Support for drag and drop from filesystem and external sources
	
	// Check if this is a file/directory drag operation from FileSystem dock
	if (drag_data.has("type") && drag_data.has("files")) {
		String type = drag_data["type"];
		if (type == "files" || type == "files_and_dirs") {
			Array files = drag_data["files"];
			// Only accept if there are files to drop (not directories)
			return files.size() > 0;
		}
	}
	
	// Check if this is an external file drop (from OS file manager)
	if (drag_data.has("type")) {
		String type = drag_data["type"];
		if (type == "files_and_dirs_external" || type == "files_external") {
			if (drag_data.has("files")) {
				Array files = drag_data["files"];
				return files.size() > 0;
			}
		}
	}
	
	// Check for direct file paths (alternative external file format)
	if (drag_data.has("files") && !drag_data.has("type")) {
		Array files = drag_data["files"];
		return files.size() > 0;
	}
	
	// Check if this is a scene node drag operation
	if (drag_data.has("type") && drag_data.has("nodes")) {
		String type = drag_data["type"];
		if (type == "nodes") {
			Array nodes = drag_data["nodes"];
			return nodes.size() > 0;
		}
	}
	
	return false;
}
void AIChatDock::drop_data(const Point2 &p_point, const Variant &p_data) {
	if (!can_drop_data(p_point, p_data)) {
		return;
	}
	
	Dictionary drag_data = p_data;
	String type = "";
	if (drag_data.has("type")) {
		type = drag_data["type"];
	}
	
	// Handle file drops (internal FileSystem dock files)
	if (type == "files" || type == "files_and_dirs") {
		Array files = drag_data["files"];
		
		Vector<String> file_paths;
		for (int i = 0; i < files.size(); i++) {
			String path = files[i];
			// Skip directories for now (may add support later)
			if (!path.ends_with("/")) {
				file_paths.push_back(path);
			}
		}
		
		if (file_paths.size() > 0) {
			_attach_dragged_files(file_paths);
		}
	}
	// Handle external file drops (from OS file manager)
	else if (type == "files_and_dirs_external" || type == "files_external" || 
			 (drag_data.has("files") && !drag_data.has("type"))) {
		Array files = drag_data["files"];
		
		Vector<String> file_paths;
		for (int i = 0; i < files.size(); i++) {
			String path = files[i];
			// Skip directories for external files too
			if (!path.ends_with("/") && !path.ends_with("\\")) {
				file_paths.push_back(path);
			}
		}
		
		if (file_paths.size() > 0) {
			_attach_external_files(file_paths);
		}
	}
	// Handle scene node drops
	else if (type == "nodes") {
		Array nodes = drag_data["nodes"];
		_attach_dragged_nodes(nodes);
	}
}

// Forwarded drag and drop support (kept for compatibility)
bool AIChatDock::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	return can_drop_data(p_point, p_data);
}

void AIChatDock::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	drop_data(p_point, p_data);
}

void AIChatDock::_attach_dragged_files(const Vector<String> &p_files) {
	// Use the existing attachment logic for internal project files
	_on_files_selected(p_files);
}
void AIChatDock::_attach_external_files(const Vector<String> &p_files) {
	for (int i = 0; i < p_files.size(); i++) {
		String file_path = p_files[i];
		
		// Check if file exists
		if (!FileAccess::exists(file_path)) {
			print_line("AI Chat: External file does not exist: " + file_path);
			continue;
		}
		
		// Check if file is already attached
		bool already_attached = false;
		for (const AttachedFile &existing_file : current_attached_files) {
			if (existing_file.path == file_path) {
				already_attached = true;
				break;
			}
		}
		
		if (!already_attached) {
			AIChatDock::AttachedFile attached_file;
			attached_file.path = file_path;
			if (_is_image_file(file_path)) {
				// Create unique image ID for images
				String clean_name = file_path.get_file().get_basename().to_lower().replace(" ", "_");
				attached_file.name = "img_" + String::num_int64(OS::get_singleton()->get_ticks_msec()) + "_" + clean_name;
			} else {
				attached_file.name = file_path.get_file();
			}
			attached_file.is_image = _is_image_file(file_path);
			attached_file.mime_type = _get_mime_type_from_extension(file_path);
			
			if (attached_file.is_image) {
				// Process image (resize if needed, convert to base64)
				if (_process_image_attachment(attached_file)) {
					current_attached_files.push_back(attached_file);
				} else {
					print_line("AI Chat: Failed to process external image: " + file_path);
				}
			} else {
				// Read text file content with safety limits to avoid blowing context
				Error err = OK;
				Ref<FileAccess> f = FileAccess::open(file_path, FileAccess::READ, &err);
				if (err == OK && f.is_valid()) {
					int64_t to_read = MIN(f->get_length(), (int64_t)MAX_TEXT_ATTACHMENT_PREVIEW_BYTES);
					PackedByteArray bytes;
					bytes.resize(to_read);
					int64_t read = f->get_buffer(bytes.ptrw(), to_read);
					f->close();
					String content = String::utf8((const char *)bytes.ptr(), (int)read);
					attached_file.content = _truncate_text_for_context(content);
					bool truncated = f->get_length() > to_read || content.length() > attached_file.content.length();
					if (truncated) {
						attached_file.content += "\n\n…\n[Truncated preview of large file. Only first " + String::num_int64(attached_file.content.length()) + " chars shown.]";
					}
					current_attached_files.push_back(attached_file);
				} else {
					print_line("AI Chat: Failed to read external file: " + file_path + " (Error: " + String::num_int64(err) + ")");
				}
			}
		}
	}
	
	_update_attached_files_display();
}
void AIChatDock::_attach_dragged_nodes(const Array &p_nodes) {
	for (int i = 0; i < p_nodes.size(); i++) {
		NodePath node_path = p_nodes[i];
		
		// Get the node to check if it exists and get info
		Node *node = get_node_or_null(node_path);
		if (!node) {
			continue; // Skip invalid nodes
		}
		
		// Check if node is already attached
		bool already_attached = false;
		for (const AttachedFile &existing_file : current_attached_files) {
			if (existing_file.is_node && existing_file.node_path == node_path) {
				already_attached = true;
				break;
			}
		}
		
		if (!already_attached) {
			AIChatDock::AttachedFile attached_node;
			attached_node.is_node = true;
			attached_node.node_path = node_path;
			attached_node.name = node->get_name();
			attached_node.node_type = node->get_class_name();
			attached_node.path = String(node_path); // Store as string for display
			
			// Create descriptive content for the node
			String node_content = "Node: " + String(node_path) + "\n";
			node_content += "Type: " + attached_node.node_type + "\n";
			node_content += "Name: " + attached_node.name + "\n";
			
			// Add additional properties if script is attached
			Ref<Script> script = node->get_script();
			if (script.is_valid()) {
				node_content += "Script: " + script->get_path().get_file() + "\n";
			}
			
			// Add position for Node2D/Node3D
			Node2D *node2d = Object::cast_to<Node2D>(node);
			if (node2d) {
				node_content += "Position: " + String(node2d->get_position()) + "\n";
			}
			Node3D *node3d = Object::cast_to<Node3D>(node);
			if (node3d) {
				node_content += "Position: " + String(node3d->get_position()) + "\n";
			}
			
			attached_node.content = node_content;
			current_attached_files.push_back(attached_node);
		}
	}
	
	_update_attached_files_display();
}
String AIChatDock::_get_file_type_icon(const AttachedFile &p_file) {
	// Check if this is a node attachment
	if (p_file.is_node) {
		// Return icon based on node type
		if (p_file.node_type == "Node2D") {
			return "Node2D";
		} else if (p_file.node_type == "Node3D") {
			return "Node3D";
		} else if (p_file.node_type == "Control") {
			return "Control";
		} else if (p_file.node_type == "Label") {
			return "Label";
		} else if (p_file.node_type == "Button") {
			return "Button";
		} else {
			return "Node";
		}
	}
	
	String extension = p_file.path.get_extension().to_lower();
	
	// Images
	if (extension == "png" || extension == "jpg" || extension == "jpeg" || 
		extension == "gif" || extension == "bmp" || extension == "svg" || 
		extension == "webp" || extension == "tga" || extension == "exr" || extension == "hdr") {
		return "ImageTexture";
	}
	
	// Scripts
	if (extension == "gd" || extension == "cs") {
		return "Script";
	}
	
	// Scenes
	if (extension == "tscn" || extension == "scn") {
		return "PackedScene";
	}
	
	// Resources
	if (extension == "tres" || extension == "res") {
		return "Object";
	}
	
	// Shaders
	if (extension == "gdshader" || extension == "glsl") {
		return "Shader";
	}
	
	// Audio
	if (extension == "ogg" || extension == "wav" || extension == "mp3") {
		return "AudioStreamOggVorbis";
	}
	
	// Models
	if (extension == "gltf" || extension == "glb" || extension == "obj" || 
		extension == "fbx" || extension == "dae") {
		return "MeshInstance3D";
	}
	
	// Text/Code files
	if (extension == "txt" || extension == "md" || extension == "json" || 
		extension == "xml" || extension == "yaml" || extension == "yml" ||
		extension == "csv" || extension == "cfg" || extension == "ini") {
		return "TextFile";
	}
	
	// Default file icon
	return "File";
}

void AIChatDock::_handle_response_chunk(const PackedByteArray &p_chunk) {
	response_buffer += String::utf8((const char *)p_chunk.ptr(), p_chunk.size());

	int newline_pos;
	while ((newline_pos = response_buffer.find("\n")) != -1) {
		String line = response_buffer.substr(0, newline_pos);
		response_buffer = response_buffer.substr(newline_pos + 1);
		if (line.strip_edges().is_empty()) {
			continue;
		}
		_process_ndjson_line(line);
	}
}
void AIChatDock::_process_ndjson_line(const String &p_line) {
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(p_line);
	if (err != OK) {
		_add_message_to_chat("system", "Error parsing streaming response: " + p_line);
		return;
	}

	Dictionary data = json->get_data();

	if (data.has("error")) {
		_add_message_to_chat("system", "Backend error: " + String(data["error"]));
		return;
	}

    if (data.has("status") && (data["status"] == "finished" || data["status"] == "completed" || data["status"] == "awaiting_frontend_action")) {
        String status = data["status"];
        print_line("AI Chat: Stream status: " + status);

        if (status == "awaiting_frontend_action") {
            // The backend has sent tool calls for us to execute.
            // The stream is technically "done" from the backend's perspective for this request,
            // but the overall operation is not complete. We keep the UI in a busy state.
            http_status = STATUS_DONE;
            // We don't call _request_completed() here because we are waiting for tool results.
            // The UI should remain in a "streaming" state with the stop button visible.
            print_line("AI Chat: Awaiting frontend tool execution.");
            return; // Exit without calling _request_completed()
        }

        // For "completed" or "finished", we finalize the request.
        _request_completed();
        http_status = STATUS_DONE;
        return;
    }

	// Handle request_id from backend (first message)
	if (data.has("request_id") && data.has("status") && data["status"] == "started") {
		current_request_id = data["request_id"];
		print_line("AI Chat: Received request ID: " + current_request_id);
		// Update UI state to enable the stop button now that we have a request ID
		_update_ui_state();
		return;
	}

	// Handle stop status from backend
	if (data.has("status") && data["status"] == "stopped") {
		stream_completed_successfully = true; // Mark as successful completion to avoid connection error
		String message = data.get("message", "Request stopped");
		print_line("AI Chat: " + message);
		
		// IMPORTANT: Save conversation when request is stopped to preserve partial content
		if (current_conversation_index >= 0) {
			conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
			_queue_delayed_save();
		}
		
		// Reset state silently (no system message shown to user)
		is_waiting_for_response = false;
		stop_requested = false;
		current_request_id = "";
		_update_ui_state();
		return;
	}

	// Handle tool starting messages for immediate feedback
	if (data.has("status") && data["status"] == "tool_starting") {
		String tool_name = data.get("tool_starting", "unknown_tool");
		String tool_id = data.get("tool_id", "");
		
		// Show immediate feedback to user that tool is starting
		print_line("AI Chat: Tool starting - " + tool_name + " (ID: " + tool_id + ")");
		
		// Backend-only tools (image_operation, search_across_project) emit tool_starting
		// without an accompanying "executing_tools" message. Ensure a placeholder exists
		// so the subsequent tool_completed can update UI. This won't duplicate frontend
		// tool UI because frontend path never sends tool_starting.
		if (chat_container) {
			PanelContainer *existing = Object::cast_to<PanelContainer>(chat_container->find_child("tool_placeholder_" + tool_id, true, false));
			if (!existing && !tool_id.is_empty()) {
				_create_backend_tool_placeholder(tool_id, tool_name);
			}
		}
		
		return; // Stop further processing for this line
	}

	if (data.has("status") && data["status"] == "executing_tools") {
		if (data.has("assistant_message")) {
			Dictionary assistant_message = data["assistant_message"];
			Array tool_calls = assistant_message.get("tool_calls", Array());

			// Mark tool call ids as executing to debounce UI creation across streams
			for (int i = 0; i < tool_calls.size(); i++) {
				Dictionary tc = tool_calls[i];
				String tcid = tc.get("id", "");
				if (!tcid.is_empty()) {
					tool_calls_executed_locally.insert(tcid);
				}
			}
			
			// Always create a new consolidated assistant message with tool calls
			// This creates one cohesive message blob containing both text and tool calls
			String assistant_content = assistant_message.get("content", "");
			if (assistant_content.is_empty() && tool_calls.size() > 0) {
				assistant_content = "Running " + String::num_int64(tool_calls.size()) + " tool" + (tool_calls.size() > 1 ? "s" : "") + "...";
			}
			
			bool attached_to_existing = false;
			// Prefer augmenting the current streaming assistant message instead of creating a new one
			if (current_assistant_message_label != nullptr) {
				Vector<AIChatDock::ChatMessage> &history = _get_current_chat_history();
				if (!history.is_empty()) {
					ChatMessage &last = history.write[history.size() - 1];
					if (last.role == "assistant") {
						// If the backend provided content here and the current message has none, adopt it
						if (!assistant_content.is_empty() && last.content.is_empty()) {
							last.content = assistant_content;
							String bbcode = _markdown_to_bbcode(assistant_content);
							if (!bbcode.is_empty()) {
								current_assistant_message_label->set_text(bbcode);
							}
						}
						// Attach tool calls to the existing assistant message
						last.tool_calls = tool_calls;
						_create_tool_call_bubbles(tool_calls);
						attached_to_existing = true;
					}
				}
			}

			if (!attached_to_existing) {
				_add_message_to_chat("assistant", assistant_content, tool_calls);
			}

			if (assistant_message.has("tool_calls")) {
				_execute_tool_calls(assistant_message.get("tool_calls", Array()));
			}
		}
		return; // Stop further processing for this line
	}
	
    // Handle completed tool execution results
    if (data.has("status") && data["status"] == "tool_completed") {
        String tool_executed = data.get("tool_executed", "");
        Dictionary tool_result = data.get("tool_result", Dictionary());
        print_line("AI Chat: Tool completed: " + tool_executed + " (success: " + String(tool_result.get("success", false) ? "true" : "false") + ")");

        String tool_call_id = data.get("tool_call_id", "");

        // Ensure conversation history contains the required assistant message with tool_calls
        // so that subsequent API requests won't fail schema validation.
        if (!tool_call_id.is_empty()) {
            AIChatDock::ChatMessage assistant_tool_call_msg;
            assistant_tool_call_msg.role = "assistant";
            assistant_tool_call_msg.content = String();
            Array tool_calls_arr;
            Dictionary tool_call_dict;
            tool_call_dict["id"] = tool_call_id;
            tool_call_dict["type"] = "function";
            Dictionary function_dict;
            function_dict["name"] = tool_executed;
            function_dict["arguments"] = "{}"; // Minimal placeholder; backend already executed the real args
            tool_call_dict["function"] = function_dict;
            tool_calls_arr.push_back(tool_call_dict);
            assistant_tool_call_msg.tool_calls = tool_calls_arr;
            Vector<AIChatDock::ChatMessage> &history = _get_current_chat_history();
            history.push_back(assistant_tool_call_msg);
        }

        // Special-case for image results when no tool_call_id is present (backend not providing it)
        if (tool_executed == "image_operation" && tool_result.get("success", false)) {
            String image_data = tool_result.get("image_data", "");
            if (!image_data.is_empty()) {
                if (tool_call_id.is_empty()) {
                    // Fallback: render image immediately if we can't update a placeholder
                    _handle_generated_image(image_data, "generated_" + String::num_int64(OS::get_singleton()->get_ticks_msec()));
                    return;
                }
                // If we have a tool_call_id, fall through to unified UI for consistency
            }
        }

        // Unified tool UI handling (updates placeholder and records in history)
        if (!tool_call_id.is_empty()) {
            Dictionary args; // Optionally populate with inputs
            _add_tool_response_to_chat(tool_call_id, tool_executed, args, tool_result);
        } else {
            print_line("AI Chat: Warning - tool_completed missing tool_call_id, cannot update placeholder");
        }
        return; // Stop further processing for this line
    }

	// Handle image generation results (legacy - keeping for backward compatibility)
	if (data.has("status") && data["status"] == "image_generated") {
		if (data.has("image_generated")) {
			Dictionary image_data = data["image_generated"];
			_handle_generated_image(image_data.get("base64_data", ""), image_data.get("id", ""));
		}
		return; // Stop further processing for this line
	}

	// This handles streaming content deltas
	if (data.has("content_delta")) {
		RichTextLabel *label = _get_or_create_current_assistant_message_label();
		
		// Safety check: ensure label is valid
		if (!label) {
			print_line("AI Chat: Warning - invalid label in content_delta handler");
			return;
		}
		
		String delta = data["content_delta"];
		Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
		if (!chat_history.is_empty()) {
			ChatMessage &last_msg = chat_history.write[chat_history.size() - 1];
			if (last_msg.role == "assistant") {
				last_msg.content += delta;
				
				// Update conversation timestamp for streaming content (but don't save yet)
				if (current_conversation_index >= 0) {
					conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
				}
				
				// Safety check: ensure content is valid before processing
				if (!last_msg.content.is_empty()) {
					String bbcode_content = _markdown_to_bbcode(last_msg.content);
					// Additional safety check for the converted content
					if (!bbcode_content.is_empty()) {
						label->set_text(bbcode_content);
					}
				}
				
				// Note: We don't save during streaming to avoid performance issues
				// Saving happens when the message is complete or stopped
			}
		}
	}

	// This handles the final message of a stream.
	if (data.has("assistant_message")) {
		Dictionary assistant_message = data["assistant_message"];

		// If this final message has tool calls, we shouldn't be processing it here.
		// It should have been handled by the "executing_tools" status.
		// Skip to avoid duplicate messages.
		if (assistant_message.has("tool_calls")) {
			print_line("AI_CHAT_DOCK: Skipping duplicate assistant message with tool calls");
			return;
		}

		// Check if we already have an assistant message with content from tool execution
		Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
		if (!chat_history.is_empty()) {
			ChatMessage &last_msg = chat_history.write[chat_history.size() - 1];
			if (last_msg.role == "assistant" && !last_msg.content.is_empty()) {
				// We already have assistant content, likely from tool execution
				// Update it with the final content instead of creating a new message
				String final_content = assistant_message.get("content", "");
				if (final_content != "<null>" && !final_content.is_empty()) {
					last_msg.content = final_content;
					
					// Update the existing label if available
					if (current_assistant_message_label) {
						String bbcode_content = _markdown_to_bbcode(final_content);
						if (!bbcode_content.is_empty()) {
							current_assistant_message_label->set_text(bbcode_content);
						}
					}
				}
				
				// Update conversation timestamp
				if (current_conversation_index >= 0) {
					conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
				}
				
				// CRITICAL: Save final assistant message content to prevent loss
				_queue_delayed_save();
				return;
			}
		}

		RichTextLabel *label = _get_or_create_current_assistant_message_label();
		
		// Safety check: ensure label is valid
		if (!label) {
			print_line("AI Chat: Warning - invalid label in assistant_message handler");
			return;
		}
		
		String final_content = assistant_message.get("content", "");
		if (final_content == "<null>") {
			final_content = "";
		}
		if (!chat_history.is_empty()) {
			ChatMessage &last_msg = chat_history.write[chat_history.size() - 1];
			if (last_msg.role == "assistant") {
				// Update the content in history and then render from history.
				last_msg.content = final_content;
				
				// Update conversation timestamp for final content
				if (current_conversation_index >= 0) {
					conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
				}
				
				if (!last_msg.content.is_empty()) {
					String bbcode_content = _markdown_to_bbcode(last_msg.content);
					if (!bbcode_content.is_empty()) {
						label->set_text(bbcode_content);
					}
				}
				
				// CRITICAL: Save final assistant message content to prevent loss
				_queue_delayed_save();
			}
		} else {
			// This case should ideally not happen if the stream started correctly.
			if (!final_content.is_empty()) {
				String bbcode_content = _markdown_to_bbcode(final_content);
				if (!bbcode_content.is_empty()) {
					label->set_text(bbcode_content);
				}
			}
		}
	}
}

RichTextLabel *AIChatDock::_get_or_create_current_assistant_message_label() {
	// If a label is already assigned, validate it and use it.
	if (current_assistant_message_label != nullptr) {
		// First check if the label itself is still valid
		bool label_is_valid = false;
		if (chat_container != nullptr) {
			// Check if the label is still a child of some node in the chat container
			for (int i = 0; i < chat_container->get_child_count(); i++) {
				Node *child = chat_container->get_child(i);
				if (_is_label_descendant_of_node(current_assistant_message_label, child)) {
					label_is_valid = true;
					break;
				}
			}
		}
		
		if (label_is_valid) {
			// Make sure its parent hierarchy is visible with safe checks
			Node *parent = current_assistant_message_label->get_parent();
			if (parent != nullptr) {
				Node *grandparent = parent->get_parent();
				if (grandparent != nullptr) {
					Control *bubble_panel = Object::cast_to<Control>(grandparent);
					if (bubble_panel && !bubble_panel->is_visible()) {
						bubble_panel->set_visible(true);
					}
				}
			}
			return current_assistant_message_label;
		} else {
			// Label is invalid, clear it
			current_assistant_message_label = nullptr;
		}
	}



	// If the last message was an assistant message (likely with tool calls but no content yet),
	// find its label and reuse it.
	// First ensure we have a valid conversation
	if (current_conversation_index < 0 || current_conversation_index >= conversations.size()) {
		// No valid conversation, create a new one
		_create_new_conversation();
	}
	
	Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
	if (!chat_history.is_empty()) {
		const AIChatDock::ChatMessage &last_msg = chat_history[chat_history.size() - 1];
		if (last_msg.role == "assistant") {
			// Find the last message bubble created.
			if (chat_container != nullptr) {
				for (int i = chat_container->get_child_count() - 1; i >= 0; i--) {
					PanelContainer *panel = Object::cast_to<PanelContainer>(chat_container->get_child(i));
					if (panel) {
						RichTextLabel *rt = find_rich_text_label_in_children(panel);
						if (rt) {
							current_assistant_message_label = rt;
							panel->set_visible(true); // Ensure it's visible.
							return current_assistant_message_label;
						}
					}
				}
			}
		}
	}

	// Otherwise, create a new message bubble.
	_add_message_to_chat("assistant", "");
	return current_assistant_message_label; // _add_message_to_chat sets this.
}
void AIChatDock::_execute_tool_calls(const Array &p_tool_calls) {
    // Ensure counter is sane at the start of a batch
    if (pending_tool_tasks < 0) pending_tool_tasks = 0;
	for (int i = 0; i < p_tool_calls.size(); i++) {
		Dictionary tool_call = p_tool_calls[i];
		String tool_call_id = tool_call.get("id", "");
		Dictionary function_dict = tool_call.get("function", Dictionary());
		String function_name = function_dict.get("name", "");
		String arguments_str = function_dict.get("arguments", "{}");

		// Show immediate visual feedback that tool is starting
		print_line("AI Chat: 🔧 Executing tool: " + function_name);
		
		// Update placeholder to show execution status
		_update_tool_placeholder_status(tool_call_id, function_name, "starting");

		Ref<JSON> json;
		json.instantiate();
		Error err = json->parse(arguments_str);
		Dictionary args;
		if (err == OK) {
			args = json->get_data();
		}

		Dictionary result;

		// Execute the tool
		if (function_name == "get_scene_info") {
			result = EditorTools::get_scene_info(args);
		} else if (function_name == "get_all_nodes") {
			result = EditorTools::get_all_nodes(args);
		} else if (function_name == "search_nodes_by_type") {
			result = EditorTools::search_nodes_by_type(args);
		} else if (function_name == "get_editor_selection") {
			result = EditorTools::get_editor_selection(args);
		} else if (function_name == "get_node_properties") {
			result = EditorTools::get_node_properties(args);
        } else if (function_name == "save_scene") {
            // Tool removed for safety - return explicit error so models stop calling it
            result["success"] = false;
            result["message"] = "Tool 'save_scene' has been removed. Please save scenes manually.";
		} else if (function_name == "get_available_classes") {
			result = EditorTools::get_available_classes(args);
		} else if (function_name == "create_node") {
			result = EditorTools::create_node(args);
		} else if (function_name == "delete_node") {
			result = EditorTools::delete_node(args);
		} else if (function_name == "set_node_property") {
			result = EditorTools::set_node_property(args);
		} else if (function_name == "batch_set_node_properties") {
			result = EditorTools::batch_set_node_properties(args);
		} else if (function_name == "move_node") {
			result = EditorTools::move_node(args);
		} else if (function_name == "call_node_method") {
			result = EditorTools::call_node_method(args);
		} else if (function_name == "get_node_script") {
			result = EditorTools::get_node_script(args);
		} else if (function_name == "attach_script") {
			result = EditorTools::attach_script(args);
		} else if (function_name == "manage_scene") {
			result = EditorTools::manage_scene(args);
        } else if (function_name == "add_collision_shape") {
			result = EditorTools::add_collision_shape(args);
		} else if (function_name == "list_project_files") {
			result = EditorTools::list_project_files(args);
        } else if (function_name == "read_file") {
            result = EditorTools::read_file(args);
		} else if (function_name == "search_project_files") {
			// search_project_files was renamed to list_project_files
			result = EditorTools::list_project_files(args);
		} else if (function_name == "read_file_content") {
            // Deprecated: route to unified read
            result = EditorTools::read_file(args);
        } else if (function_name == "read_file_advanced") {
            // Deprecated: route to unified read
            result = EditorTools::read_file(args);
        } else if (function_name == "apply_edit") {
            // Run apply_edit asynchronously to avoid blocking the UI
            // Skip compilation checking by default for better performance
            if (!args.has("skip_compilation_check")) {
                args["skip_compilation_check"] = true;
            }
            pending_tool_tasks++;
            _update_tool_placeholder_status(tool_call_id, function_name, "running");
            _execute_apply_edit_async(tool_call_id, args);
            // Skip immediate result handling; it will be added when the thread completes
            continue;
		} else if (function_name == "check_compilation_errors") {
			result = EditorTools::check_compilation_errors(args);
		} else if (function_name == "run_scene") {
			result = EditorTools::run_scene(args);
		} else if (function_name == "get_scene_tree_hierarchy") {
			result = EditorTools::get_scene_tree_hierarchy(args);
		} else if (function_name == "inspect_physics_body") {
			result = EditorTools::inspect_physics_body(args);
		} else if (function_name == "get_camera_info") {
			result = EditorTools::get_camera_info(args);
		} else if (function_name == "take_screenshot") {
			result = EditorTools::take_screenshot(args);
		} else if (function_name == "check_node_in_scene_tree") {
			result = EditorTools::check_node_in_scene_tree(args);
		} else if (function_name == "inspect_animation_state") {
			result = EditorTools::inspect_animation_state(args);
		} else if (function_name == "get_layers_and_zindex") {
			result = EditorTools::get_layers_and_zindex(args);
        } else if (function_name == "create_script_file") {
            // Map deprecated create_script_file to apply_edit for forward compatibility
            // Expect args: { path, description, script_type?, node_type? }
            String target_path = String(args.get("path", ""));
            String description = String(args.get("description", ""));
            String script_type = String(args.get("script_type", ""));
            String node_type = String(args.get("node_type", ""));

            Dictionary apply_args;
            apply_args["path"] = target_path;
            // Craft a prompt for file creation when using apply_edit
            String composed_prompt;
            composed_prompt += "Create or overwrite this file with a valid Godot 4.x script.\n";
            if (!script_type.is_empty()) composed_prompt += "Script type: " + script_type + "\n";
            if (!node_type.is_empty()) composed_prompt += "Node type: " + node_type + "\n";
            if (!description.is_empty()) composed_prompt += "Requirements: " + description + "\n";
            composed_prompt += "Return only the complete file content.";
            apply_args["prompt"] = composed_prompt;

            result = EditorTools::apply_edit(apply_args);
		} else if (function_name == "delete_file_safe") {
			// delete_file_safe method no longer exists
			result["success"] = false;
			result["message"] = "delete_file_safe is no longer available";
        } else if (function_name == "edit_file_with_diff") {
            // Deprecated tool removed from backend; return a hard error instructing the model to switch.
            result["success"] = false;
            result["message"] = "Tool 'edit_file_with_diff' has been removed. Use 'apply_edit' instead.";
		} else if (function_name == "image_operation") {
			// This tool should be handled by the backend, not the frontend
			// If we receive it here, it means something went wrong in the backend filtering
			result["success"] = false;
			result["message"] = "Image generation should be handled by backend, not frontend";
			print_line("AI Chat: Received image_operation tool in frontend - this should be handled by backend");
        } else if (function_name == "editor_introspect") {
            // Execute multiplexed introspection/debug tool
            result = EditorTools::editor_introspect(args);
        } else if (function_name == "search_across_godot_docs") {
            // Forward docs search to EditorTools (proxied to backend)
            result = EditorTools::search_across_godot_docs(args);
        } else if (function_name == "save_image_to_path") {
            // Frontend-local tool: save an image to a path on the editor machine
            String image_id = String(args.get("image_id", ""));
            String dest_path = String(args.get("path", ""));
            String format_hint = String(args.get("format", "png"));
            bool saved = false;
            String error_msg;
            if (dest_path.is_empty()) {
                error_msg = "Missing path";
            } else {
                String b64;
                // Search recent messages for an attached image matching the id
                Vector<AIChatDock::ChatMessage> &history = _get_current_chat_history();
                for (int h = history.size() - 1; h >= 0 && b64.is_empty(); h--) {
                    for (int f = 0; f < history[h].attached_files.size(); f++) {
                        const AttachedFile &af = history[h].attached_files[f];
                        if (!af.is_image) continue;
                        if (image_id.is_empty() || af.name == image_id || af.path.ends_with("/" + image_id) || af.path == (String("generated://") + image_id)) {
                            b64 = af.base64_data;
                            break;
                        }
                    }
                }
                if (b64.is_empty() && args.has("base64_data")) {
                    b64 = String(args.get("base64_data", ""));
                }
                if (!b64.is_empty()) {
                    saved = _save_base64_image_to_path(b64, dest_path);
                    if (!saved) error_msg = "Failed to write file";
                } else {
                    error_msg = image_id.is_empty() ? "No image data provided" : (String("Image not found: ") + image_id);
                }
            }
            result["success"] = saved;
            if (saved) {
                result["message"] = String("Saved image to: ") + dest_path;
                result["path"] = dest_path;
                result["format"] = format_hint;
            } else {
                result["message"] = error_msg;
            }
        } else {
            result["success"] = false;
            result["message"] = "Unknown tool: " + function_name;
        }

        // Add a proper, separate tool bubble for the output.
        _add_tool_response_to_chat(tool_call_id, function_name, args, result);

        // Note: Do not inject unsolicited tool results here. The model must request tools.
		
		// Don't add system messages - they break OpenAI format!
		// bool success = result.get("success", false);
		// String message = result.get("message", "");
		// String status_icon = success ? "✅" : "❌";
		// _add_message_to_chat("system", status_icon + " " + function_name + ": " + message);
	}

    // If there are async tool tasks running, defer finalization until they complete.
    if (pending_tool_tasks > 0) {
        print_line("AI Chat: Waiting for " + String::num_int64(pending_tool_tasks) + " async tool task(s) to finish...");
        return;
    }

    // We've finished with this turn's tool calls. Clear the current label
    // so the next content_delta or assistant_message creates a new bubble
    // or finds the correct existing one.
    current_assistant_message_label = nullptr;
    
    // Send the conversation with tool results back to backend to continue processing
    // This is expected by the backend after frontend tool execution
    _send_chat_request();
}

// Struct to pass data to apply_edit thread
struct ApplyEditTaskData {
    AIChatDock *dock = nullptr;
    String tool_call_id;
    Dictionary args;
    Dictionary result;
};

void AIChatDock::_execute_apply_edit_async(const String &p_tool_call_id, const Dictionary &p_args) {
    ApplyEditTaskData *task = memnew(ApplyEditTaskData);
    task->dock = this;
    task->tool_call_id = p_tool_call_id;
    task->args = p_args;

    Thread *t = memnew(Thread);
    // Detached thread; it will clean itself by calling back into _on_apply_edit_thread_done
    t->start(_apply_edit_thread, task);
}

void AIChatDock::_apply_edit_thread(void *p_userdata) {
    ApplyEditTaskData *task = static_cast<ApplyEditTaskData *>(p_userdata);
    if (!task || !task->dock) {
        return;
    }
    // Mark AI frontend busy for overlay usage and UI state
    task->dock->ai_busy_mutex.lock();
    task->dock->ai_busy_count++;
    task->dock->ai_busy_mutex.unlock();
    // Perform the heavy work off the main thread
    Dictionary result = EditorTools::apply_edit(task->args);
    // Decrement busy when finished
    task->dock->ai_busy_mutex.lock();
    task->dock->ai_busy_count = MAX(0, task->dock->ai_busy_count - 1);
    task->dock->ai_busy_mutex.unlock();
    task->result = result;
    // Stash pointer for main thread consumption (deferred)
    if (!task->dock->apply_edit_mutex) {
        task->dock->apply_edit_mutex = memnew(Mutex);
    }
    task->dock->apply_edit_mutex->lock();
    task->dock->apply_edit_done.push_back(task);
    task->dock->apply_edit_mutex->unlock();
    // Hop back to main thread to update UI/state
    task->dock->call_deferred("_on_apply_edit_thread_done");
}
void AIChatDock::_on_apply_edit_thread_done() {
    // Drain any completed tasks queued by background threads
    Vector<void *> to_process;
    if (apply_edit_mutex) apply_edit_mutex->lock();
    to_process = apply_edit_done;
    apply_edit_done.clear();
    if (apply_edit_mutex) apply_edit_mutex->unlock();

    for (int i = 0; i < to_process.size(); i++) {
        ApplyEditTaskData *task = static_cast<ApplyEditTaskData *>(to_process[i]);
        if (!task) continue;

        pending_tool_tasks = MAX(0, pending_tool_tasks - 1);
        // Keep UI busy if background apply threads still running
        bool busy = false;
        ai_busy_mutex.lock();
        busy = ai_busy_count > 0;
        ai_busy_mutex.unlock();
        is_waiting_for_response = busy || pending_tool_tasks > 0;
        const String tool_name = "apply_edit";
        _update_tool_placeholder_status(task->tool_call_id, tool_name, "completed");
        _add_tool_response_to_chat(task->tool_call_id, tool_name, task->args, task->result);
        memdelete(task);
    }

    if (pending_tool_tasks == 0) {
        current_assistant_message_label = nullptr;
        _send_chat_request();
    }
}
void AIChatDock::_add_message_to_chat(const String &p_role, const String &p_content, const Array &p_tool_calls) {
	AIChatDock::ChatMessage msg;
	msg.role = p_role;
	msg.content = p_content;
	if (msg.content == "<null>") {
		msg.content = "";
	}
	msg.timestamp = _get_timestamp();
	msg.tool_calls = p_tool_calls;

	Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
	chat_history.push_back(msg);
	_create_message_bubble(msg, chat_history.size() - 1);
	// Show tool call placeholders immediately for better UX
	if (p_role == "assistant" && !p_tool_calls.is_empty()) {
		_create_tool_call_bubbles(p_tool_calls);
		// Make sure the message bubble is visible when we have tool calls
		if (chat_container != nullptr && chat_container->get_child_count() > 0) {
			Node *last_child = chat_container->get_child(chat_container->get_child_count() - 1);
			PanelContainer *message_panel = Object::cast_to<PanelContainer>(last_child);
			if (message_panel) {
				message_panel->set_visible(true);
			}
		}
	}

	// Update conversation timestamp and save
	if (current_conversation_index >= 0) {
		conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
		_queue_delayed_save();
	}

	// Auto-scroll to bottom
	call_deferred("_scroll_to_bottom");
}

void AIChatDock::_add_tool_response_to_chat(const String &p_tool_call_id, const String &p_name, const Dictionary &p_args, const Dictionary &p_result) {
	Ref<JSON> json;
	json.instantiate();

	AIChatDock::ChatMessage msg;
	msg.role = "tool";
	msg.tool_call_id = p_tool_call_id;
	msg.name = p_name;
    // Prepare a content payload we can enrich with metadata like image_name
    Dictionary result_for_content = p_result;
	msg.timestamp = _get_timestamp();
	
	// Store the original tool arguments for proper UI recreation
	msg.tool_results.clear();
	msg.tool_results.push_back(p_result);
	msg.tool_results.push_back(p_args); // Store args as second element

	// If this is an image generation/edit result, attach the image to the message so
	// subsequent requests can reference it via top-level 'images' (derived at serialization).
	if (p_name == "image_operation" && p_result.get("success", false) && p_result.has("image_data")) {
		AIChatDock::AttachedFile gen_file;
		gen_file.path = "generated://tool_result";
        gen_file.name = String("generated_") + String::num_int64(OS::get_singleton()->get_ticks_msec());
		gen_file.content = "";
		gen_file.is_image = true;
		gen_file.mime_type = "image/png";
		gen_file.base64_data = p_result.get("image_data", "");
		msg.attached_files.push_back(gen_file);

        // Expose the image identifier in the tool message content for the model to reference
        result_for_content["image_name"] = gen_file.name;
	}

    msg.content = json->stringify(result_for_content);

	Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
	chat_history.push_back(msg);

	// Find the placeholder for this tool and replace its content.
	if (chat_container == nullptr) {
		return;
	}
	// Debounce: if UI for this tool id has already been applied, skip creating again
	if (tool_calls_ui_applied.has(p_tool_call_id)) {
		print_line("AI Chat: Skipping duplicate tool UI for id=" + p_tool_call_id);
		return;
	}

	PanelContainer *placeholder = Object::cast_to<PanelContainer>(chat_container->find_child("tool_placeholder_" + p_tool_call_id, true, false));
	if (!placeholder) {
		// Avoid creating a second bubble; schedule a deferred retry to give
		// the assistant message a moment to render its placeholders.
		call_deferred("_apply_tool_result_deferred", p_tool_call_id, p_name, msg.content, msg.tool_results);
		return;
	}

	// Clear the "loading" text immediately.
	while (placeholder->get_child_count() > 0) {
		Node *child = placeholder->get_child(0);
		placeholder->remove_child(child);
		child->queue_free();
	}

	// Create the replacement UI.
	VBoxContainer *tool_container = memnew(VBoxContainer);
	placeholder->add_child(tool_container);

	Button *toggle_button = memnew(Button);
	
	// Show success/failure status in the button
	Dictionary data = p_result;
	bool success = data.get("success", false);
	String message = data.get("message", "");
	String status_text = success ? "SUCCESS" : "ERROR";
	toggle_button->set_text(status_text + " - " + p_name + ": " + message);
	
	toggle_button->set_flat(false);
	toggle_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	toggle_button->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	// Prevent long status lines from expanding the dock width
	toggle_button->set_clip_text(true);
	toggle_button->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	// Preserve full text in tooltip for accessibility
	toggle_button->set_tooltip_text(toggle_button->get_text());
	toggle_button->add_theme_icon_override("icon", get_theme_icon(success ? SNAME("StatusSuccess") : SNAME("StatusError"), SNAME("EditorIcons")));
	toggle_button->add_theme_color_override("font_color", success ? get_theme_color(SNAME("success_color"), SNAME("Editor")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
	tool_container->add_child(toggle_button);

				PanelContainer *content_panel = memnew(PanelContainer);
	content_panel->set_visible(false); // Collapsed by default.
	content_panel->set_clip_contents(true);
				tool_container->add_child(content_panel);
				toggle_button->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_output_toggled).bind(content_panel));

				Ref<StyleBoxFlat> content_style = memnew(StyleBoxFlat);
				content_style->set_bg_color(get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
	content_style->set_border_width_all(1);
	content_style->set_border_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
				content_style->set_content_margin_all(10);
				content_panel->add_theme_style_override("panel", content_style);

	VBoxContainer *content_vbox = memnew(VBoxContainer);
	content_panel->add_child(content_vbox);

	HBoxContainer *header_hbox = memnew(HBoxContainer);
	content_vbox->add_child(header_hbox);

	// Dictionary data = p_result; // Already declared above
	// bool success = data.get("success", false); // Already declared above

	Label *status_label = memnew(Label);
	status_label->set_text(success ? "Tool Succeeded" : "Tool Failed");
	status_label->add_theme_color_override("font_color", success ? get_theme_color(SNAME("success_color"), SNAME("Editor")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
	status_label->add_theme_icon_override("icon", get_theme_icon(success ? SNAME("StatusSuccess") : SNAME("StatusError"), SNAME("EditorIcons")));
	header_hbox->add_child(status_label);

	content_vbox->add_child(memnew(HSeparator));

	// Use the shared tool-specific UI creation function
	_create_tool_specific_ui(content_vbox, p_name, p_result, success, p_args);

	tool_calls_ui_applied.insert(p_tool_call_id);
	call_deferred("_scroll_to_bottom");
}

void AIChatDock::_scroll_to_bottom() {
	call_deferred("_perform_scroll");
}

void AIChatDock::_perform_scroll() {
	if (chat_scroll) {
		VScrollBar *vbar = chat_scroll->get_v_scroll_bar();
		if (vbar) {
			vbar->set_value(vbar->get_max());
		}
	}
}

bool AIChatDock::_is_at_bottom() const {
	if (!chat_scroll) {
		return true;
	}
	
	VScrollBar *vbar = chat_scroll->get_v_scroll_bar();
	if (!vbar) {
		return true;
	}
	
	// Consider user at bottom if within a small threshold of the end.
	// End is when value + page >= max.
	const float threshold = 20.0f;
	const float value = vbar->get_value();
	const float page = vbar->get_page();
	const float max_value = vbar->get_max();
	
	return (max_value - (value + page)) <= threshold;
}

void AIChatDock::_on_chat_scroll_changed(float p_value) {
	// Update auto-scroll state based on user's scroll position.
	auto_scroll_at_bottom = _is_at_bottom();
}

void AIChatDock::_scroll_to_bottom_smooth() {
	// For now, just use regular scroll_to_bottom
	// TODO: Implement smooth scrolling animation
	_scroll_to_bottom();
}

void AIChatDock::_on_chat_content_min_size_changed() {
	// Only auto-scroll when the user is already at the bottom.
	if (auto_scroll_at_bottom) {
		call_deferred("_scroll_to_bottom");
	}
}

String AIChatDock::_truncate_text_for_context(const String &p_text, int p_max_chars) {
    if (p_text.length() <= p_max_chars) {
        return p_text;
    }
    // Keep head and tail to preserve useful context
    int head = p_max_chars * 3 / 4;
    int tail = p_max_chars - head;
    String head_str = p_text.substr(0, head);
    String tail_str = p_text.substr(p_text.length() - tail, tail);
    return head_str + "\n\n…\n[Middle omitted; content truncated to fit context]\n\n" + tail_str;
}

String AIChatDock::_convert_to_godot_path(const String &p_path) {
    String path = p_path;
    
    // Remove .uid suffix if present
    if (path.ends_with(".uid")) {
        path = path.trim_suffix(".uid");
    }
    
    // If not an engine path and not absolute, assume project-relative and prefix with res://
    bool is_engine_path = path.begins_with("res://") || path.begins_with("user://");
    bool is_windows_abs = (path.length() >= 2 && path[1] == ':');
    bool is_absolute = path.begins_with("/") || path.begins_with("\\") || is_windows_abs;
    if (!is_engine_path && !is_absolute) {
        String normalized_rel = path.replace("\\", "/");
        if (normalized_rel.begins_with("./")) {
            normalized_rel = normalized_rel.substr(2);
        }
        path = String("res://") + normalized_rel;
    }
    
    // If path is absolute, try to convert to res:// when possible
    if (path.begins_with("/") || path.begins_with("\\") || is_windows_abs) {
        String project_root = ProjectSettings::get_singleton()->globalize_path("res://");
        
        // Handle both forward and backward slashes
        String normalized_path = path.replace("\\", "/");
        String normalized_root = project_root.replace("\\", "/");
        
        if (normalized_path.begins_with(normalized_root)) {
            String rel = normalized_path.substr(normalized_root.length());
            if (rel.begins_with("/")) rel = rel.substr(1);
            path = String("res://") + rel;
        }
    }
    
    return path;
}

void AIChatDock::_on_tool_file_link_pressed(const String &p_path) {
    String path = _convert_to_godot_path(p_path);
    
    print_line("AI Chat: Opening file from search: " + path);
    String ext = path.get_extension().to_lower();
    if (ext == "tscn" || ext == "scn") {
        // Open scene in editor
        EditorInterface::get_singleton()->open_scene_from_path(path);
        return;
    }
    if (ext == "gd" || ext == "cs") {
        // Open script in script editor
        Ref<Resource> res = ResourceLoader::load(path);
        Ref<Script> script = res;
        if (script.is_valid()) {
            ScriptEditor *se = ScriptEditor::get_singleton();
            if (se) {
                se->edit(script);
                return;
            }
        }
    }
    // Fallback: try generic load
    EditorNode::get_singleton()->load_scene(path);
}
void AIChatDock::_create_message_bubble(const AIChatDock::ChatMessage &p_message, int p_message_index) {
	if (chat_container == nullptr) {
		return;
	}
	PanelContainer *message_panel = memnew(PanelContainer);
	
	// Add spacing before each message for cleaner layout
	if (chat_container->get_child_count() > 0) {
		Control *spacer = memnew(Control);
		spacer->set_custom_minimum_size(Size2(0, 8)); // 8px gap between messages
		chat_container->add_child(spacer);
	}
	
    chat_container->add_child(message_panel);
    // Name the panel with its message index for easy lookup/replacement later.
    if (p_message_index >= 0) {
        message_panel->set_name("message_panel_" + String::num_int64(p_message_index));
    }

	// Default to invisible. We'll show it only if it has content.
	message_panel->set_visible(false);

	// --- Modern Message Bubble Styling ---
	Ref<StyleBoxFlat> panel_style = memnew(StyleBoxFlat);
	panel_style->set_content_margin_all(12); // Slightly more padding for modern look
	panel_style->set_corner_radius_all(8); // Rounded corners for modern appearance
	Color role_color;

	if (p_message.role == "user") {
		// User messages: subtle blue background like modern chat apps
		panel_style->set_bg_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.08));
		panel_style->set_border_width_all(1);
		panel_style->set_border_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.2));
		role_color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
	} else { // Assistant and System
		// Assistant messages: clean subtle background
		panel_style->set_bg_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
		panel_style->set_border_width_all(1);
		panel_style->set_border_color(get_theme_color(SNAME("dark_color_3"), SNAME("Editor")));
		role_color = (p_message.role == "system") ? get_theme_color(SNAME("warning_color"), SNAME("Editor")) : get_theme_color(SNAME("font_color"), SNAME("Editor"));
	}
	message_panel->add_theme_style_override("panel", panel_style);

	// --- Content ---
	VBoxContainer *message_vbox = memnew(VBoxContainer);
	message_panel->add_child(message_vbox);

	// Role label with edit button for user messages
	HBoxContainer *role_container = memnew(HBoxContainer);
	message_vbox->add_child(role_container);
	
	Label *role_label = memnew(Label);
	role_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
	role_label->set_text(p_message.role.capitalize());
	role_label->add_theme_color_override("font_color", role_color);
	role_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	role_container->add_child(role_label);
	
    // Add edit button for user messages only
    if (p_message.role == "user" && p_message_index >= 0) {
        Button *edit_button = memnew(Button);
		edit_button->set_text("Edit");
		edit_button->set_custom_minimum_size(Size2(50, 20));
		edit_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Edit"), SNAME("EditorIcons")));
		
		edit_button->connect("pressed", callable_mp(this, &AIChatDock::_on_edit_message_pressed).bind(p_message_index));
		role_container->add_child(edit_button);
	}

	// Always create a content label for assistant to allow streaming.
	RichTextLabel *content_label = memnew(RichTextLabel);
	content_label->set_fit_content(true);
	content_label->set_selection_enabled(true);
	content_label->set_use_bbcode(true);
	content_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	message_vbox->add_child(content_label);

	if (p_message.role == "assistant") {
		current_assistant_message_label = content_label;
	}

	if (!p_message.content.strip_edges().is_empty()) {
		String bbcode_content = _markdown_to_bbcode(p_message.content);
		if (!bbcode_content.is_empty()) {
			content_label->set_text(bbcode_content);
		}
		message_panel->set_visible(true);
	}

	if (!p_message.tool_calls.is_empty()) {
		message_panel->set_visible(true);
		// Recreate tool call placeholders when loading saved conversations
		_create_tool_call_bubbles(p_message.tool_calls);
	}
	
	// Handle tool result messages specially
	if (p_message.role == "tool" && !p_message.tool_call_id.is_empty()) {
		message_panel->set_visible(true);
		// Find and update the corresponding tool placeholder
		_update_tool_placeholder_with_result(p_message);
	}
	
	// Display attached files if any
	if (!p_message.attached_files.is_empty()) {
		message_panel->set_visible(true);
		
		// Track displayed generated images to avoid duplication
		HashSet<String> displayed_generated_images;
		
		// Display all attached files with unified UI
		for (const AttachedFile &file : p_message.attached_files) {
			if (file.is_image && !file.base64_data.is_empty()) {
				// Create metadata dictionary for unified display
				Dictionary metadata;
				metadata["name"] = file.name;
				metadata["path"] = file.path;
				metadata["mime_type"] = file.mime_type;
				metadata["original_size_x"] = file.original_size.x;
				metadata["original_size_y"] = file.original_size.y;
				metadata["was_downsampled"] = file.was_downsampled;
				
				// For generated images, track them to prevent duplication
				if (file.path.begins_with("generated://")) {
					displayed_generated_images.insert(file.base64_data);
				}
				
				_display_image_unified(message_vbox, file.base64_data, metadata);
			} else if (!file.is_image) {
				// Display non-image files with existing logic
				VBoxContainer *files_container = memnew(VBoxContainer);
				message_vbox->add_child(files_container);
				
				Label *files_header = memnew(Label);
				files_header->set_text("Attached Files:");
				files_header->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
				files_header->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
				files_container->add_child(files_header);
				
				HFlowContainer *files_flow = memnew(HFlowContainer);
				files_container->add_child(files_flow);
				
				HBoxContainer *file_row = memnew(HBoxContainer);
				files_flow->add_child(file_row);
				
				Label *file_icon = memnew(Label);
				file_icon->add_theme_icon_override("icon", get_theme_icon(SNAME("File"), SNAME("EditorIcons")));
				file_row->add_child(file_icon);
				
				Button *file_link = memnew(Button);
				file_link->set_text(file.name);
				file_link->set_flat(true);
				file_link->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
				file_link->set_tooltip_text("Click to open: " + file.path);
				file_link->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_file_link_pressed).bind(file.path));
				file_row->add_child(file_link);
			}
		}
		
		// Store displayed images to prevent duplication in tool_results section  
		current_displayed_images = displayed_generated_images;
	}
	
	// Display tool results (like generated images) if any, but avoid duplication
	if (!p_message.tool_results.is_empty()) {
		message_panel->set_visible(true);
		
		for (int i = 0; i < p_message.tool_results.size(); i++) {
			Dictionary tool_result = p_message.tool_results[i];
			
			// Check if this is an image generation result
			if (tool_result.get("success", false) && tool_result.has("image_data")) {
				String image_data = tool_result.get("image_data", "");
				String prompt = tool_result.get("prompt", "Generated Image");
				
				// Only display if not already displayed in attached_files section
				if (!image_data.is_empty() && !current_displayed_images.has(image_data)) {
					print_line("AI Chat: Displaying saved image from tool result: " + prompt + " (" + String::num(image_data.length()) + " chars base64)");
					
					// Create metadata for unified display
					Dictionary metadata;
					metadata["prompt"] = prompt;
					metadata["model"] = tool_result.get("model", "DALL-E");
					metadata["path"] = "generated://tool_result";
					
					_display_image_unified(message_vbox, image_data, metadata);
				}
			}
		}
	}
	
	// Clear the displayed images set for next message
	current_displayed_images.clear();

    // Single spacing strategy handled before each message; avoid adding an extra trailing spacer per message.
}
void AIChatDock::_build_message_content(PanelContainer *p_message_panel, const AIChatDock::ChatMessage &p_message, int p_message_index) {
	if (p_message_panel == nullptr) {
		return;
	}

	// Default to invisible. We'll show it only if it has content.
	p_message_panel->set_visible(false);

	// --- Modern Message Bubble Styling ---
	Ref<StyleBoxFlat> panel_style = memnew(StyleBoxFlat);
	panel_style->set_content_margin_all(12); // Slightly more padding for modern look
	panel_style->set_corner_radius_all(8); // Rounded corners for modern appearance
	Color role_color;

	if (p_message.role == "user") {
		// User messages: subtle blue background like modern chat apps
		panel_style->set_bg_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.08));
		panel_style->set_border_width_all(1);
		panel_style->set_border_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.2));
		role_color = get_theme_color(SNAME("accent_color"), SNAME("Editor"));
	} else { // Assistant and System
		// Assistant messages: clean subtle background
		panel_style->set_bg_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
		panel_style->set_border_width_all(1);
		panel_style->set_border_color(get_theme_color(SNAME("dark_color_3"), SNAME("Editor")));
		role_color = (p_message.role == "system") ? get_theme_color(SNAME("warning_color"), SNAME("Editor")) : get_theme_color(SNAME("font_color"), SNAME("Editor"));
	}
	p_message_panel->add_theme_style_override("panel", panel_style);

	// --- Content ---
	VBoxContainer *message_vbox = memnew(VBoxContainer);
	p_message_panel->add_child(message_vbox);

	// Role label with edit button for user messages
	HBoxContainer *role_container = memnew(HBoxContainer);
	message_vbox->add_child(role_container);
	
	Label *role_label = memnew(Label);
	role_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
	role_label->set_text(p_message.role.capitalize());
	role_label->add_theme_color_override("font_color", role_color);
	role_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	role_container->add_child(role_label);
	
    // Add edit button for user messages only
    if (p_message.role == "user" && p_message_index >= 0) {
        Button *edit_button = memnew(Button);
		edit_button->set_text("Edit");
		edit_button->set_custom_minimum_size(Size2(50, 20));
		edit_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Edit"), SNAME("EditorIcons")));
		
		edit_button->connect("pressed", callable_mp(this, &AIChatDock::_on_edit_message_pressed).bind(p_message_index));
		role_container->add_child(edit_button);
	}

	// Always create a content label for assistant to allow streaming.
	RichTextLabel *content_label = memnew(RichTextLabel);
	content_label->set_fit_content(true);
	content_label->set_selection_enabled(true);
	content_label->set_use_bbcode(true);
	content_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	message_vbox->add_child(content_label);

	if (p_message.role == "assistant") {
		current_assistant_message_label = content_label;
	}

	if (!p_message.content.strip_edges().is_empty()) {
		String bbcode_content = _markdown_to_bbcode(p_message.content);
		if (!bbcode_content.is_empty()) {
			content_label->set_text(bbcode_content);
		}
		p_message_panel->set_visible(true);
	}

	if (!p_message.tool_calls.is_empty()) {
		p_message_panel->set_visible(true);
		// Recreate tool call placeholders when loading saved conversations
		_create_tool_call_bubbles(p_message.tool_calls);
	}
	
	// Handle tool result messages specially
	if (p_message.role == "tool" && !p_message.tool_call_id.is_empty()) {
		p_message_panel->set_visible(true);
		// Find and update the corresponding tool placeholder
		_update_tool_placeholder_with_result(p_message);
	}
	
	// Display attached files if any
	if (!p_message.attached_files.is_empty()) {
		p_message_panel->set_visible(true);
		
		// Track displayed generated images to avoid duplication
		HashSet<String> displayed_generated_images;
		
		// Display all attached files with unified UI
		for (const AttachedFile &file : p_message.attached_files) {
			if (file.is_image && !file.base64_data.is_empty()) {
				// Create metadata dictionary for unified display
				Dictionary metadata;
				metadata["name"] = file.name;
				metadata["path"] = file.path;
				metadata["mime_type"] = file.mime_type;
				metadata["original_size_x"] = file.original_size.x;
				metadata["original_size_y"] = file.original_size.y;
				metadata["was_downsampled"] = file.was_downsampled;
				
				// For generated images, track them to prevent duplication
				if (file.path.begins_with("generated://")) {
					displayed_generated_images.insert(file.base64_data);
				}
				
				_display_image_unified(message_vbox, file.base64_data, metadata);
			} else if (!file.is_image) {
				// Display non-image files with existing logic
				VBoxContainer *files_container = memnew(VBoxContainer);
				message_vbox->add_child(files_container);
				
				Label *files_header = memnew(Label);
				files_header->set_text("Attached Files:");
				files_header->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
				files_header->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
				files_container->add_child(files_header);
				
				HFlowContainer *files_flow = memnew(HFlowContainer);
				files_container->add_child(files_flow);
				
				HBoxContainer *file_row = memnew(HBoxContainer);
				files_flow->add_child(file_row);
				
				Label *file_icon = memnew(Label);
				file_icon->add_theme_icon_override("icon", get_theme_icon(SNAME("File"), SNAME("EditorIcons")));
				file_row->add_child(file_icon);
				
				Button *file_link = memnew(Button);
				file_link->set_text(file.name);
				file_link->set_flat(true);
				file_link->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
				file_link->set_tooltip_text("Click to open: " + file.path);
				file_link->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_file_link_pressed).bind(file.path));
				file_row->add_child(file_link);
			}
		}
		
		// Store displayed images to prevent duplication in tool_results section  
		current_displayed_images = displayed_generated_images;
	}
	
	// Display tool results (like generated images) if any, but avoid duplication
	if (!p_message.tool_results.is_empty()) {
		p_message_panel->set_visible(true);
		
		for (int i = 0; i < p_message.tool_results.size(); i++) {
			Dictionary tool_result = p_message.tool_results[i];
			
			// Check if this is an image generation result
			if (tool_result.get("success", false) && tool_result.has("image_data")) {
				String image_data = tool_result.get("image_data", "");
				String prompt = tool_result.get("prompt", "Generated Image");
				
				// Only display if not already displayed in attached_files section
				if (!image_data.is_empty() && !current_displayed_images.has(image_data)) {
					print_line("AI Chat: Displaying saved image from tool result: " + prompt + " (" + String::num(image_data.length()) + " chars base64)");
					
					// Create metadata for unified display
					Dictionary metadata;
					metadata["prompt"] = prompt;
					metadata["model"] = tool_result.get("model", "DALL-E");
					metadata["path"] = "generated://tool_result";
					
					_display_image_unified(message_vbox, image_data, metadata);
				}
			}
		}
	}
	
	// Clear the displayed images set for next message
	current_displayed_images.clear();
}

void AIChatDock::_create_tool_call_bubbles(const Array &p_tool_calls) {
	if (!current_assistant_message_label || p_tool_calls.is_empty()) {
		return;
	}

	Control *bubble_panel = Object::cast_to<Control>(current_assistant_message_label->get_parent()->get_parent());
	VBoxContainer *message_vbox = Object::cast_to<VBoxContainer>(bubble_panel->get_child(0));

	if (!message_vbox) {
		return;
	}

	// Create a single container for all tool calls to group them together
	VBoxContainer *tools_container = memnew(VBoxContainer);
	tools_container->set_name("tools_container");
	message_vbox->add_child(tools_container);

	// Add a label for the tool calls section
	if (p_tool_calls.size() > 1) {
		Label *tools_header = memnew(Label);
		tools_header->set_text("Tools (" + String::num_int64(p_tool_calls.size()) + "):");
		tools_header->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		tools_header->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.8));
		tools_container->add_child(tools_header);
	}

	// Create individual tool placeholders within the grouped container
	for (int i = 0; i < p_tool_calls.size(); i++) {
		Dictionary tool_call = p_tool_calls[i];
		String tool_call_id = tool_call.get("id", "");
		Dictionary function_dict = tool_call.get("function", Dictionary());
		String func_name = function_dict.get("name", "unknown_tool");

		// Create a container that will hold either the "loading" label or the final tool output.
		PanelContainer *placeholder = memnew(PanelContainer);
		placeholder->set_name("tool_placeholder_" + tool_call_id);
		tools_container->add_child(placeholder);

		Ref<StyleBoxFlat> placeholder_style = memnew(StyleBoxFlat);
		placeholder_style->set_bg_color(get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
		placeholder_style->set_content_margin_all(10);
		placeholder_style->set_border_width_all(1);
		placeholder_style->set_border_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
		placeholder_style->set_corner_radius_all(5);
		placeholder->add_theme_style_override("panel", placeholder_style);

		HBoxContainer *tool_hbox = memnew(HBoxContainer);
		placeholder->add_child(tool_hbox);

        Label *tool_label = memnew(Label);
        tool_label->set_text("🔧 " + func_name + "...");
		tool_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.6));
		tool_hbox->add_child(tool_label);
	}
}

void AIChatDock::_update_tool_placeholder_with_result(const ChatMessage &p_tool_message) {
	if (chat_container == nullptr) {
		return;
	}
	
	// Find the placeholder for this tool call ID
	PanelContainer *placeholder = Object::cast_to<PanelContainer>(chat_container->find_child("tool_placeholder_" + p_tool_message.tool_call_id, true, false));
	if (!placeholder) {
		// If no placeholder exists, this might be a saved conversation 
		// where the tool call UI wasn't recreated yet. Skip for now.
		print_line("AI Chat: Warning - Could not find tool placeholder for ID: " + p_tool_message.tool_call_id + " (normal for loaded conversations)");
		return;
	}

	// Clear the "loading" text or any existing content immediately
	while (placeholder->get_child_count() > 0) {
		Node *child = placeholder->get_child(0);
		placeholder->remove_child(child);
		child->queue_free();
	}

	// Parse the tool result from JSON content with better error handling
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(p_tool_message.content);
	Dictionary result;
	if (err == OK) {
		result = json->get_data();
	} else {
		// If parsing fails, check if tool_results array has the data
		if (!p_tool_message.tool_results.is_empty()) {
			result = p_tool_message.tool_results[0]; // First element is the result
		} else {
			// Fallback if both JSON parsing and tool_results fail
			result["success"] = false;
			result["message"] = "Failed to parse tool result: " + p_tool_message.content;
		}
	}

	// Create the tool result UI using the same logic as _add_tool_response_to_chat
	VBoxContainer *tool_container = memnew(VBoxContainer);
	placeholder->add_child(tool_container);

    Button *toggle_button = memnew(Button);
	
	// Show success/failure status in the button
	bool success = result.get("success", false);
	String message = result.get("message", "");
	String status_text = success ? "SUCCESS" : "ERROR";
	toggle_button->set_text(status_text + " - " + p_tool_message.name + ": " + message);
	
    toggle_button->set_flat(false);
	toggle_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	toggle_button->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
    toggle_button->set_clip_text(true);
    toggle_button->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	toggle_button->add_theme_icon_override("icon", get_theme_icon(success ? SNAME("StatusSuccess") : SNAME("StatusError"), SNAME("EditorIcons")));
	toggle_button->add_theme_color_override("font_color", success ? get_theme_color(SNAME("success_color"), SNAME("Editor")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
	tool_container->add_child(toggle_button);

    PanelContainer *content_panel = memnew(PanelContainer);
    content_panel->set_visible(false); // Collapsed by default (may be opened below conditionally)
    content_panel->set_clip_contents(true); // Prevent visual overflow
    tool_container->add_child(content_panel);
    toggle_button->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_output_toggled).bind(content_panel));

    Ref<StyleBoxFlat> content_style = memnew(StyleBoxFlat);
    content_style->set_bg_color(get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
    content_style->set_border_width_all(1);
    content_style->set_border_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
    content_style->set_content_margin_all(6);
    content_panel->add_theme_style_override("panel", content_style);

    // Wrap tool content in a scroll container to avoid growing beyond view
    ScrollContainer *content_scroll = memnew(ScrollContainer);
    content_scroll->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    content_scroll->set_v_size_flags(Control::SIZE_FILL);
    content_scroll->set_custom_minimum_size(Size2(0, 320)); // cap visible height
    content_scroll->set_horizontal_scroll_mode(ScrollContainer::SCROLL_MODE_DISABLED);
    content_panel->add_child(content_scroll);

    VBoxContainer *content_vbox = memnew(VBoxContainer);
    content_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    content_vbox->set_v_size_flags(Control::SIZE_SHRINK_BEGIN);
    content_scroll->add_child(content_vbox);

	HBoxContainer *header_hbox = memnew(HBoxContainer);
	content_vbox->add_child(header_hbox);

	Label *status_label = memnew(Label);
	status_label->set_text(success ? "Tool Succeeded" : "Tool Failed");
	status_label->add_theme_color_override("font_color", success ? get_theme_color(SNAME("success_color"), SNAME("Editor")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
	status_label->add_theme_icon_override("icon", get_theme_icon(success ? SNAME("StatusSuccess") : SNAME("StatusError"), SNAME("EditorIcons")));
	header_hbox->add_child(status_label);

	content_vbox->add_child(memnew(HSeparator));

	// Extract arguments if they were stored - better error handling
	Dictionary args;
	if (p_tool_message.tool_results.size() > 1) {
		args = p_tool_message.tool_results[1]; // Args are stored as second element
	}

    // Create specific UI based on the tool that was called
    _create_tool_specific_ui(content_vbox, p_tool_message.name, result, success, args);

    // Open image tool results by default for better UX
    if (p_tool_message.name == "image_operation" || (result.get("success", false) && result.has("image_data"))) {
        content_panel->set_visible(true);
    }
}
void AIChatDock::_create_tool_specific_ui(VBoxContainer *p_content_vbox, const String &p_tool_name, const Dictionary &p_result, bool p_success, const Dictionary &p_args) {
	Ref<JSON> json;
	json.instantiate();
	
	// Create specific UI based on the tool that was called
	if (p_tool_name == "list_project_files" && p_success) {
		Tree *file_tree = memnew(Tree);
		file_tree->set_hide_root(true);
		file_tree->set_custom_minimum_size(Size2(0, 300));
		TreeItem *root = file_tree->create_item();

		Dictionary tree_items;

		Array files = p_result.get("files", Array());
		for (int i = 0; i < files.size(); i++) {
			Variant f = files[i];
			String file_path;
			int64_t line_count = -1;
			if (f.get_type() == Variant::DICTIONARY) {
				Dictionary fd = f;
				file_path = fd.get("path", String());
				line_count = (int64_t)fd.get("line_count", (int64_t)-1);
				if (file_path.is_empty() && fd.has("name")) {
					// Fallback to name only
					file_path = String(fd.get("name", String()));
				}
			} else {
				file_path = String(f);
			}
			if (file_path.is_empty()) {
				continue;
			}
			Vector<String> parts = file_path.split("/");
			TreeItem *current_item = root;
			String current_path = "";

			for (int j = 0; j < parts.size(); j++) {
				current_path += parts[j];
				if (tree_items.has(current_path)) {
					current_item = (TreeItem *)((Object *)tree_items[current_path]);
				} else {
					TreeItem *new_item = file_tree->create_item(current_item);
					new_item->set_text(0, parts[j]);
					bool is_dir = j < parts.size() - 1;
					new_item->set_icon(0, is_dir ? get_theme_icon(SNAME("Folder"), SNAME("EditorIcons")) : get_theme_icon(SNAME("File"), SNAME("EditorIcons")));
					// Add line count tooltip on leaf when available
					if (!is_dir && line_count >= 0) {
						new_item->set_tooltip_text(0, String::num_int64(line_count) + " lines");
					}
					tree_items[current_path] = new_item;
					current_item = new_item;
				}
				current_path += "/";
			}
		}
		p_content_vbox->add_child(file_tree);

    } else if ((p_tool_name == "read_file" || p_tool_name == "read_file_content" || p_tool_name == "read_file_advanced") && p_success) {
		VBoxContainer *file_content_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(file_content_vbox);

		// Get the file path from the original arguments
        String file_path = p_args.get("path", p_result.get("file_path", "Unknown file"));
		String display_path = _convert_to_godot_path(file_path);
		
		Button *file_link = memnew(Button);
		file_link->set_text(display_path);
		file_link->set_flat(true);
		file_link->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
		file_link->add_theme_icon_override("icon", get_theme_icon(SNAME("File"), SNAME("EditorIcons")));
		file_link->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_file_link_pressed).bind(file_path));
		file_content_vbox->add_child(file_link);

		TextEdit *file_content = memnew(TextEdit);
		file_content->set_text(p_result.get("content", ""));
		file_content->set_editable(false);
		file_content->set_custom_minimum_size(Size2(0, 300));
		file_content_vbox->add_child(file_content);

	} else if (p_tool_name == "get_scene_info" && p_success) {
		VBoxContainer *scene_info_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(scene_info_vbox);

		Dictionary root_node = p_result.get("root_node", Dictionary());
		String scene_name = p_result.get("scene_name", "Unknown");
		
		// Scene info summary
		Label *scene_label = memnew(Label);
		scene_label->set_text("Scene: " + scene_name);
		scene_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		scene_info_vbox->add_child(scene_label);
		
		if (!root_node.is_empty()) {
			HBoxContainer *root_hbox = memnew(HBoxContainer);
			scene_info_vbox->add_child(root_hbox);
			
			Label *root_label = memnew(Label);
			String root_name = root_node.get("name", "Unknown");
			String root_type = root_node.get("type", "Unknown");
			root_label->set_text("Root Node: " + root_name + " (" + root_type + ")");
			root_label->add_theme_icon_override("icon", get_theme_icon(SNAME("Node"), SNAME("EditorIcons")));
			root_hbox->add_child(root_label);
			
			// Child count
			int child_count = root_node.get("child_count", 0);
			Label *child_label = memnew(Label);
			child_label->set_text("Children: " + String::num_int64(child_count));
			root_hbox->add_child(child_label);
		}

	} else if (p_tool_name == "get_all_nodes" && p_success) {
		VBoxContainer *nodes_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(nodes_vbox);

		Array nodes = p_result.get("nodes", Array());
		
		Label *count_label = memnew(Label);
		count_label->set_text("Found " + String::num_int64(nodes.size()) + " nodes:");
		count_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		nodes_vbox->add_child(count_label);
		
		// Create a tree for better organization
		Tree *nodes_tree = memnew(Tree);
		nodes_tree->set_hide_root(true);
		nodes_tree->set_custom_minimum_size(Size2(0, 200));
		nodes_tree->set_columns(2); // FIX: Set to 2 columns before accessing column 1
		TreeItem *root = nodes_tree->create_item();
		
		for (int i = 0; i < nodes.size(); i++) {
			Dictionary node = nodes[i];
			TreeItem *item = nodes_tree->create_item(root);
			
			String name = node.get("name", "Unknown");
			String type = node.get("type", "Unknown");
			String path = node.get("path", "");
			
			item->set_text(0, name + " (" + type + ")");
			item->set_tooltip_text(0, "Path: " + path);
			item->set_icon(0, get_theme_icon(SNAME("Node"), SNAME("EditorIcons")));
			
			// Show child count if available
			int child_count = node.get("child_count", -1);
			if (child_count >= 0) {
				item->set_text(1, String::num_int64(child_count) + " children");
			}
		}
		
		nodes_tree->set_column_title(0, "Node");
		nodes_tree->set_column_title(1, "Children");
		nodes_tree->set_column_titles_visible(true);
		nodes_vbox->add_child(nodes_tree);

	} else if (p_tool_name == "create_node" && p_success) {
		VBoxContainer *create_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(create_vbox);

		String node_path = p_result.get("node_path", "Unknown");
		String node_type = p_args.get("type", "Unknown");
		String node_name = p_args.get("name", "Unknown");
		String parent_path = p_args.get("parent", "Scene Root");
		
		HBoxContainer *node_hbox = memnew(HBoxContainer);
		create_vbox->add_child(node_hbox);
		
		Label *node_label = memnew(Label);
		node_label->set_text("Created: " + node_name + " (" + node_type + ")");
		node_label->add_theme_icon_override("icon", get_theme_icon(SNAME("Node"), SNAME("EditorIcons")));
		node_hbox->add_child(node_label);
		
		Label *parent_label = memnew(Label);
		parent_label->set_text("Parent: " + parent_path);
		parent_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
		create_vbox->add_child(parent_label);

	} else if (p_tool_name == "search_nodes_by_type" && p_success) {
		VBoxContainer *search_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(search_vbox);

		Array nodes = p_result.get("nodes", Array());
		String node_type = p_args.get("node_type", "Unknown");
		
		Label *count_label = memnew(Label);
		count_label->set_text("Found " + String::num_int64(nodes.size()) + " nodes of type: " + node_type);
		count_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		search_vbox->add_child(count_label);
		
		if (nodes.size() > 0) {
			Tree *nodes_tree = memnew(Tree);
			nodes_tree->set_hide_root(true);
			nodes_tree->set_custom_minimum_size(Size2(0, 150));
			TreeItem *root = nodes_tree->create_item();
			
			for (int i = 0; i < nodes.size(); i++) {
				Dictionary node = nodes[i];
				TreeItem *item = nodes_tree->create_item(root);
				
				String name = node.get("name", "Unknown");
				String path = node.get("path", "");
				
				item->set_text(0, name);
				item->set_tooltip_text(0, "Path: " + path);
				item->set_icon(0, get_theme_icon(SNAME("Node"), SNAME("EditorIcons")));
			}
			search_vbox->add_child(nodes_tree);
		}

	} else if (p_tool_name == "search_across_godot_docs" && p_success) {
		// Compact rows that expand on demand; keep DOM light by toggling two nodes
		Array results = p_result.get("results", Array());
		Label *count_label = memnew(Label);
		count_label->set_text("Docs results: " + String::num_int64(results.size()));
		count_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		p_content_vbox->add_child(count_label);

		for (int i = 0; i < results.size(); i++) {
			if (results[i].get_type() != Variant::DICTIONARY) continue;
			Dictionary r = results[i];
			String title = r.get("title", "(untitled)");
			String snippet = r.get("snippet", "");
			String full_text = r.get("full_content", snippet);
			String file_path = r.get("file_path", "");
			double sim = r.get("similarity", 0.0);

			PanelContainer *row = memnew(PanelContainer);
			Ref<StyleBoxFlat> sb = memnew(StyleBoxFlat);
			sb->set_bg_color(get_theme_color(SNAME("base_color"), SNAME("Editor")) * Color(1.02, 1.02, 1.06, 1));
			sb->set_border_width_all(1);
			sb->set_border_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
			sb->set_corner_radius_all(6);
			sb->set_content_margin_all(12);
			// Subtle accent on the left for visual grouping
			sb->set_border_width(SIDE_LEFT, 3);
			sb->set_border_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
			row->add_theme_style_override("panel", sb);
			p_content_vbox->add_child(row);

			VBoxContainer *holder = memnew(VBoxContainer);
			holder->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			row->add_child(holder);

			// Header row: title + score
			HBoxContainer *hdr = memnew(HBoxContainer);
			holder->add_child(hdr);
			Label *t = memnew(Label);
			t->set_h_size_flags(Control::SIZE_EXPAND_FILL);
			t->set_text(title);
			t->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
			hdr->add_child(t);
			Label *score = memnew(Label);
			score->set_text("  •  sim: " + String::num_real(sim).substr(0, 5));
			score->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1,1,1,0.6));
			hdr->add_child(score);

			// Collapsed snippet node
			RichTextLabel *snippet_rt = memnew(RichTextLabel);
			snippet_rt->set_fit_content(true);
			snippet_rt->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
			snippet_rt->set_custom_minimum_size(Size2(0, 100));
			snippet_rt->set_text(snippet);
			holder->add_child(snippet_rt);

			// Expanded full text node (initially hidden)
			RichTextLabel *full_rt = memnew(RichTextLabel);
			full_rt->set_fit_content(false); // allow internal scroll
			full_rt->set_scroll_active(true);
			full_rt->set_autowrap_mode(TextServer::AUTOWRAP_WORD);
			full_rt->set_v_size_flags(Control::SIZE_FILL);
			full_rt->set_custom_minimum_size(Size2(0, 320));
			full_rt->set_selection_enabled(true);
			full_rt->set_text(full_text);
			full_rt->set_visible(false);
			holder->add_child(full_rt);

			HBoxContainer *btns = memnew(HBoxContainer);
			holder->add_child(btns);
			Button *toggle_btn = memnew(Button);
			toggle_btn->set_text("Show more");
			toggle_btn->set_flat(true);
			btns->add_child(toggle_btn);
			toggle_btn->connect("pressed", callable_mp(this, &AIChatDock::_toggle_docs_card).bind(snippet_rt, full_rt, toggle_btn, holder));

			if (!file_path.is_empty()) {
				Label *fp = memnew(Label);
				fp->set_text(file_path);
				fp->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1,1,1,0.5));
				holder->add_child(fp);
			}
		}

	} else if (p_tool_name == "get_editor_selection" && p_success) {
		VBoxContainer *selection_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(selection_vbox);

		Array selected_nodes = p_result.get("selected_nodes", Array());
		
		Label *count_label = memnew(Label);
		count_label->set_text("Selected Nodes: " + String::num_int64(selected_nodes.size()));
		count_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		selection_vbox->add_child(count_label);
		
		if (selected_nodes.size() > 0) {
			for (int i = 0; i < selected_nodes.size(); i++) {
				Dictionary node = selected_nodes[i];
				HBoxContainer *node_hbox = memnew(HBoxContainer);
				selection_vbox->add_child(node_hbox);
				
				Label *node_label = memnew(Label);
				String name = node.get("name", "Unknown");
				String type = node.get("type", "Unknown");
				node_label->set_text(name + " (" + type + ")");
				node_label->add_theme_icon_override("icon", get_theme_icon(SNAME("Node"), SNAME("EditorIcons")));
				node_hbox->add_child(node_label);
			}
		} else {
			Label *empty_label = memnew(Label);
			empty_label->set_text("No nodes selected");
			empty_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.6));
			selection_vbox->add_child(empty_label);
		}

	} else if (p_tool_name == "get_node_properties" && p_success) {
		VBoxContainer *props_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(props_vbox);

		String node_path = p_args.get("node_path", "Unknown");
		Dictionary properties = p_result.get("properties", Dictionary());
		
		Label *node_label = memnew(Label);
		node_label->set_text("Properties for: " + node_path);
		node_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		props_vbox->add_child(node_label);
		
		if (properties.size() > 0) {
			Tree *props_tree = memnew(Tree);
			props_tree->set_hide_root(true);
			props_tree->set_custom_minimum_size(Size2(0, 200));
			props_tree->set_columns(2); // FIX: Set to 2 columns before accessing column 1
			TreeItem *root = props_tree->create_item();
			props_tree->set_column_title(0, "Property");
			props_tree->set_column_title(1, "Value");
			props_tree->set_column_titles_visible(true);
			
			Array keys = properties.keys();
			for (int i = 0; i < keys.size(); i++) {
				TreeItem *item = props_tree->create_item(root);
				String key = keys[i];
				String value = String(properties[key]);
				item->set_text(0, key);
				item->set_text(1, value);
			}
			props_vbox->add_child(props_tree);
		}

	} else if (p_tool_name == "delete_node" && p_success) {
		VBoxContainer *delete_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(delete_vbox);

		String node_path = p_args.get("node_path", "Unknown");
		
		HBoxContainer *delete_hbox = memnew(HBoxContainer);
		delete_vbox->add_child(delete_hbox);
		
		Label *delete_label = memnew(Label);
		delete_label->set_text("Deleted node: " + node_path);
		delete_label->add_theme_icon_override("icon", get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
		delete_label->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), SNAME("Editor")));
		delete_hbox->add_child(delete_label);

	} else if (p_tool_name == "set_node_property" && p_success) {
		VBoxContainer *set_prop_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(set_prop_vbox);

		String node_path = p_args.get("node_path", "Unknown");
		String property = p_args.get("property", "Unknown");
		String value = String(p_args.get("value", ""));
		
		Label *node_label = memnew(Label);
		node_label->set_text("Updated: " + node_path);
		node_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		set_prop_vbox->add_child(node_label);
		
		HBoxContainer *prop_hbox = memnew(HBoxContainer);
		set_prop_vbox->add_child(prop_hbox);
		
		Label *prop_label = memnew(Label);
		prop_label->set_text(property + " = " + value);
		prop_label->add_theme_icon_override("icon", get_theme_icon(SNAME("Edit"), SNAME("EditorIcons")));
		prop_hbox->add_child(prop_label);

	} else if (p_tool_name == "batch_set_node_properties" && p_success) {
		VBoxContainer *batch_prop_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(batch_prop_vbox);

		int applied = p_result.get("applied", 0);
		Array failures = p_result.get("failures", Array());
		Array operations = p_args.get("operations", Array());
		
		Label *summary_label = memnew(Label);
		summary_label->set_text("Batch Update: " + String::num_int64(applied) + " properties updated");
		summary_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		summary_label->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
		batch_prop_vbox->add_child(summary_label);
		
		// Show successful operations in a tree
		if (applied > 0 && operations.size() > 0) {
			Tree *props_tree = memnew(Tree);
			props_tree->set_hide_root(true);
			props_tree->set_custom_minimum_size(Size2(0, MIN(200, applied * 25 + 50)));
			props_tree->set_columns(3);
			props_tree->set_column_title(0, "Node");
			props_tree->set_column_title(1, "Property");
			props_tree->set_column_title(2, "Value");
			props_tree->set_column_titles_visible(true);
			TreeItem *root = props_tree->create_item();
			
			// Show successful operations (skip failures for now)
			int success_count = 0;
			for (int i = 0; i < operations.size() && success_count < applied; i++) {
				Dictionary op = operations[i];
				String node_path = op.get("path", "");
				String property = op.get("property", "");
				String value = String(op.get("value", ""));
				
				// Check if this operation failed by looking in failures array
				bool op_failed = false;
				for (int f = 0; f < failures.size(); f++) {
					Dictionary failure = failures[f];
					// Simple heuristic: if failure message contains node path and property, it's this op
					String fail_msg = failure.get("message", "");
					if (fail_msg.find(node_path) != -1 && fail_msg.find(property) != -1) {
						op_failed = true;
						break;
					}
				}
				
				if (!op_failed) {
					TreeItem *item = props_tree->create_item(root);
					item->set_text(0, node_path);
					item->set_text(1, property);
					item->set_text(2, value);
					item->set_icon(0, get_theme_icon(SNAME("Edit"), SNAME("EditorIcons")));
					success_count++;
				}
			}
			batch_prop_vbox->add_child(props_tree);
		}
		
		if (failures.size() > 0) {
			Label *failures_label = memnew(Label);
			failures_label->set_text(String::num_int64(failures.size()) + " operations failed");
			failures_label->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), SNAME("Editor")));
			batch_prop_vbox->add_child(failures_label);
			
			// Show first few failures
			for (int i = 0; i < MIN(3, failures.size()); i++) {
				Dictionary failure = failures[i];
				String error_msg = failure.get("message", "Unknown error");
				Label *error_label = memnew(Label);
				error_label->set_text("• " + error_msg);
				error_label->add_theme_font_size_override("font_size", 10);
				error_label->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
				batch_prop_vbox->add_child(error_label);
			}
		}

    } else if (p_tool_name == "get_available_classes" && p_success) {
		VBoxContainer *classes_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(classes_vbox);

		Array classes = p_result.get("classes", Array());
		
		Label *count_label = memnew(Label);
		count_label->set_text("Available Classes: " + String::num_int64(classes.size()));
		count_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		classes_vbox->add_child(count_label);
		
		if (classes.size() > 0) {
			Tree *classes_tree = memnew(Tree);
			classes_tree->set_hide_root(true);
			classes_tree->set_custom_minimum_size(Size2(0, 200));
			TreeItem *root = classes_tree->create_item();
			
			for (int i = 0; i < classes.size(); i++) {
				TreeItem *item = classes_tree->create_item(root);
				String class_name = classes[i];
				item->set_text(0, class_name);
				item->set_icon(0, get_theme_icon(SNAME("Object"), SNAME("EditorIcons")));
			}
			classes_vbox->add_child(classes_tree);
		}

	} else if (p_tool_name == "move_node" && p_success) {
		VBoxContainer *move_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(move_vbox);

		String node_path = p_args.get("node_path", "Unknown");
		String new_parent = p_args.get("new_parent", "Unknown");
		
		Label *move_label = memnew(Label);
		move_label->set_text("Moved: " + node_path);
		move_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		move_vbox->add_child(move_label);
		
		Label *parent_label = memnew(Label);
		parent_label->set_text("New Parent: " + new_parent);
		parent_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
		move_vbox->add_child(parent_label);

	} else if (p_tool_name == "search_project_files" && p_success) {
		VBoxContainer *search_files_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(search_files_vbox);

		Array files = p_result.get("files", Array());
		String search_term = p_args.get("search_term", "");
		
		Label *count_label = memnew(Label);
		count_label->set_text("Found " + String::num_int64(files.size()) + " files matching: " + search_term);
		count_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		search_files_vbox->add_child(count_label);
		
		if (files.size() > 0) {
			Tree *files_tree = memnew(Tree);
			files_tree->set_hide_root(true);
			files_tree->set_custom_minimum_size(Size2(0, 200));
			TreeItem *root = files_tree->create_item();
			
			for (int i = 0; i < files.size(); i++) {
				String file_path = files[i];
				TreeItem *item = files_tree->create_item(root);
				item->set_text(0, file_path);
				item->set_icon(0, get_theme_icon(SNAME("File"), SNAME("EditorIcons")));
			}
			search_files_vbox->add_child(files_tree);
		}

    } else if (p_tool_name == "apply_edit" && p_success) {
        // Inline Accept/Reject diff preview directly in the chat tool result (self-contained)
        String file_path = p_args.has("path") ? p_args.get("path", "Unknown") : p_args.get("file_path", "Unknown");
        String original_content = p_result.get("original_content", "");
        String edited_content = p_result.get("edited_content", "");
        Dictionary structured_edits = p_result.get("structured_edits", Dictionary());
        Array comp_errors = p_result.get("compilation_errors", Array());
        bool has_errors = p_result.get("has_errors", false);

        // Make edited content visible to subsequent read_file calls before user accepts
        if (!file_path.is_empty()) {
            EditorTools::set_preview_overlay(file_path, edited_content);
            
            // For script files, also show diff in script editor
            String ext = file_path.get_extension().to_lower();
            bool is_script_like = (ext == "gd" || ext == "cs" || ext == "shader" || ext == "glsl");
            if (is_script_like) {
                _show_diff_in_script_editor(file_path, original_content, edited_content);
            }
        }

        // Also render a compact status in the chat bubble
        VBoxContainer *edit_vbox = memnew(VBoxContainer);
        p_content_vbox->add_child(edit_vbox);
        Label *status = memnew(Label);
        String display_path = _convert_to_godot_path(file_path);
        String status_text = "Edited: " + display_path;
        // Show line range if it was a partial edit
        int start_line = p_args.get("start_line", 0);
        int end_line = p_args.get("end_line", 0);
        if (start_line > 0 && end_line > 0) {
            status_text += " (lines " + String::num_int64(start_line) + "-" + String::num_int64(end_line) + ")";
        }
        if (has_errors) {
            status_text += " — Errors detected (" + String::num_int64(comp_errors.size()) + ")";
        }
        status->set_text(status_text);
        status->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1,1,1,0.8));
        edit_vbox->add_child(status);

        // Provide explicit controls to apply or discard the preview (all files) and show an inline diff
        if (!file_path.is_empty()) {
            HBoxContainer *btns = memnew(HBoxContainer);
            btns->set_name("apply_discard_buttons");
            edit_vbox->add_child(btns);

            Button *apply_btn = memnew(Button);
            apply_btn->set_name("apply_button");
            // Tailor button text by type
            bool is_script_like_btn = (file_path.get_extension().to_lower() == "gd" || file_path.get_extension().to_lower() == "cs" || file_path.get_extension().to_lower() == "shader" || file_path.get_extension().to_lower() == "glsl");
            apply_btn->set_text(is_script_like_btn ? "Apply to Editor" : "Write file & open in Inspector");
            apply_btn->add_theme_icon_override("icon", get_theme_icon(SNAME("Edit"), SNAME("EditorIcons")));
            btns->add_child(apply_btn);
            // Pass stable NodePaths for this bubble's buttons and status label so we can update reliably
            NodePath btns_path = NodePath(btns->get_path());
            Label *status_label = status; // created just above
            NodePath status_path = NodePath(status_label ? status_label->get_path() : NodePath());
            apply_btn->connect("pressed", callable_mp(this, &AIChatDock::_on_apply_preview_to_editor).bind(file_path, edited_content, btns_path, status_path));

            Button *discard_btn = memnew(Button);
            discard_btn->set_name("discard_button");
            discard_btn->set_text("Discard Preview");
            discard_btn->add_theme_icon_override("icon", get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
            btns->add_child(discard_btn);
            discard_btn->connect("pressed", callable_mp(this, &AIChatDock::_on_discard_preview).bind(file_path, btns_path, status_path));

            // Register this pending edit for tracking
            _register_pending_edit(file_path, btns_path, status_path);

            // Inline diff block (simple textual diff view for all files)
            RichTextLabel *diff_view = memnew(RichTextLabel);
            diff_view->set_selection_enabled(true);
            diff_view->set_fit_content(true);
            diff_view->push_mono();
            // Very simple unified-like diff
            diff_view->add_text("--- original\n");
            diff_view->add_text("+++ edited\n");
            // For large files, avoid flooding UI
            const int kMaxShow = 40000;
            String preview_original = original_content.substr(0, kMaxShow);
            String preview_edited = edited_content.substr(0, kMaxShow);
            diff_view->add_text("\n[Original]\n" + preview_original + "\n\n[Edited]\n" + preview_edited);
            diff_view->pop();
            edit_vbox->add_child(diff_view);
        }

        // After Accept/Discard, collapse to view-only mode by clearing the overlay and disabling buttons.
        // This transition is handled inside the respective handlers.

        // If there are errors, show a brief list
        if (has_errors && comp_errors.size() > 0) {
            RichTextLabel *errs = memnew(RichTextLabel);
            errs->set_fit_content(true);
            errs->set_selection_enabled(true);
            errs->push_color(get_theme_color(SNAME("warning_color"), SNAME("Editor")));
            errs->add_text("Problems in edited file:\n");
            errs->pop();
            int shown = 0;
            for (int i = 0; i < comp_errors.size() && shown < 5; i++) {
                Dictionary e = comp_errors[i];
                String line = String(e.get("type", "error")) + String(" @ ") + String::num_int64(int64_t(e.get("line", 0))) + ": " + String(e.get("message", ""));
                errs->add_text("• " + line + "\n");
                shown++;
            }
            if (comp_errors.size() > shown) {
                errs->add_text("… and more");
            }
            edit_vbox->add_child(errs);
        }
	} else if (p_tool_name == "get_scene_tree_hierarchy" && p_success) {
		VBoxContainer *hierarchy_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(hierarchy_vbox);

		Dictionary hierarchy = p_result.get("hierarchy", Dictionary());
		
		Label *title_label = memnew(Label);
		title_label->set_text("Scene Tree Hierarchy:");
		title_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		hierarchy_vbox->add_child(title_label);
		
		// Create a simplified hierarchy display
		Tree *hierarchy_tree = memnew(Tree);
		hierarchy_tree->set_hide_root(true);
		hierarchy_tree->set_custom_minimum_size(Size2(0, 300));
		TreeItem *root = hierarchy_tree->create_item();
		
		// Build tree recursively (simplified version)
		if (!hierarchy.is_empty()) {
			_build_hierarchy_tree_item(hierarchy_tree, root, hierarchy);
		}
		
		hierarchy_vbox->add_child(hierarchy_tree);

	} else if (p_tool_name == "take_screenshot" && p_success) {
		VBoxContainer *screenshot_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(screenshot_vbox);

		String image_path = p_result.get("image_path", "");
		
		HBoxContainer *screenshot_hbox = memnew(HBoxContainer);
		screenshot_vbox->add_child(screenshot_hbox);
		
		Label *screenshot_label = memnew(Label);
		screenshot_label->set_text("Screenshot saved: " + image_path);
		screenshot_label->add_theme_icon_override("icon", get_theme_icon(SNAME("Image"), SNAME("EditorIcons")));
		screenshot_label->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
		screenshot_hbox->add_child(screenshot_label);

	} else if (p_tool_name == "create_script_file" && p_success) {
		VBoxContainer *script_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(script_vbox);

		String script_path = p_result.get("script_path", "Unknown");
		String script_language = p_args.get("language", "GDScript");
		
		Label *script_label = memnew(Label);
		script_label->set_text("Created " + script_language + " script:");
		script_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		script_vbox->add_child(script_label);
		
		Button *file_link = memnew(Button);
		file_link->set_text(script_path);
		file_link->set_flat(true);
		file_link->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
		file_link->add_theme_icon_override("icon", get_theme_icon(SNAME("Script"), SNAME("EditorIcons")));
		file_link->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_file_link_pressed).bind(script_path));
		script_vbox->add_child(file_link);

	} else if (p_tool_name == "attach_script" && p_success) {
		VBoxContainer *attach_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(attach_vbox);

		String node_path = p_args.get("node_path", "Unknown");
		String script_path = p_args.get("script_path", "Unknown");
		
		Label *attach_label = memnew(Label);
		attach_label->set_text("Attached script to: " + node_path);
		attach_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		attach_vbox->add_child(attach_label);
		
		Label *script_label = memnew(Label);
		script_label->set_text("Script: " + script_path);
		script_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
		attach_vbox->add_child(script_label);
	} else if (p_tool_name == "run_scene" && p_success) {
		VBoxContainer *run_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(run_vbox);

		String scene_path = p_args.get("scene_path", "Current scene");
		
		HBoxContainer *run_hbox = memnew(HBoxContainer);
		run_vbox->add_child(run_hbox);
		
		Label *run_label = memnew(Label);
		run_label->set_text("Running scene: " + scene_path);
		run_label->add_theme_icon_override("icon", get_theme_icon(SNAME("Play"), SNAME("EditorIcons")));
		run_label->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
		run_hbox->add_child(run_label);

	} else if (p_tool_name == "check_compilation_errors" && p_success) {
		VBoxContainer *compile_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(compile_vbox);

		Array errors = p_result.get("errors", Array());
		String mode = p_result.get("mode", "scripts");
		
		Label *status_label = memnew(Label);
		if (errors.size() == 0) {
			if (mode == "output") {
				status_label->set_text("No errors or warnings found in output panel");
			} else {
				status_label->set_text("No compilation errors found");
			}
			status_label->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
		} else {
			if (mode == "output") {
				status_label->set_text("✗ Found " + String::num_int64(errors.size()) + " errors/warnings in output");
			} else {
				status_label->set_text("✗ Found " + String::num_int64(errors.size()) + " compilation errors");
			}
			status_label->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
		}
		status_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		compile_vbox->add_child(status_label);
		
		if (errors.size() > 0) {
			for (int i = 0; i < errors.size(); i++) {
				Dictionary error = errors[i];
				VBoxContainer *error_vbox = memnew(VBoxContainer);
				compile_vbox->add_child(error_vbox);
				
				Label *error_label = memnew(Label);
				String file = error.get("file", "Unknown");
				int line = error.get("line", 0);
				String message = error.get("message", "Unknown error");
				String type = error.get("type", "error");
				String source = error.get("source", "");
				
				String error_text = file + ":" + String::num_int64(line) + " - " + message;
				if (mode == "output" && !source.is_empty()) {
					error_text = "[" + source + "] " + error_text;
				}
				
				error_label->set_text(error_text);
				if (type == "warning") {
					error_label->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), SNAME("Editor")));
				} else {
					error_label->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
				}
				error_vbox->add_child(error_label);
			}
		}
	} else if (p_tool_name == "image_operation" && p_success) {
		// Special handling for image generation results
		String base64_data = p_result.get("image_data", "");
		
		if (!base64_data.is_empty()) {
			// Display the generated image with unified method
			Dictionary metadata;
			metadata["prompt"] = p_result.get("prompt", "Generated Image");
			metadata["model"] = p_result.get("model", "DALL-E");
			metadata["path"] = "generated://tool_operation";
			
			_display_image_unified(p_content_vbox, base64_data, metadata);
		} else {
			// Fallback to text display if no image data
			RichTextLabel *content_label = memnew(RichTextLabel);
			content_label->add_theme_font_override("normal_font", get_theme_font(SNAME("source"), SNAME("EditorFonts")));
			content_label->set_text(json->stringify(p_result, "  "));
			content_label->set_fit_content(true);
			content_label->set_selection_enabled(true);
			p_content_vbox->add_child(content_label);
		}
	} else if (p_tool_name == "search_across_project" && p_success) {
		// Display search results with nice formatting
		VBoxContainer *search_vbox = memnew(VBoxContainer);
		p_content_vbox->add_child(search_vbox);

		Dictionary results = p_result.get("results", Dictionary());
		Array similar_files = results.get("similar_files", Array());
		Array central_files = results.get("central_files", Array());
		Dictionary graph_summary = results.get("graph_summary", Dictionary());
		String query;
		if (p_args.has("query")) {
			query = String(p_args.get("query", ""));
		} else if (p_result.has("query")) {
			query = String(p_result.get("query", ""));
		} else {
			query = "";
		}
		
		// Header with query info
		Label *query_label = memnew(Label);
		query_label->set_text("Search across project for: \"" + query + "\"");
		query_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		query_label->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
		search_vbox->add_child(query_label);
		
		search_vbox->add_child(memnew(HSeparator));
		
        // Similar files section
		if (similar_files.size() > 0) {
			Label *similar_header = memnew(Label);
			similar_header->set_text("📁 Similar Files (" + String::num_int64(similar_files.size()) + ")");
			similar_header->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
			search_vbox->add_child(similar_header);
			
			for (int i = 0; i < similar_files.size(); i++) {
				Dictionary file_result = similar_files[i];
				String file_path = file_result.get("file_path", "");
				float similarity = file_result.get("similarity", 0.0);
				String modality = file_result.get("modality", "text");
                int64_t chunk_index = (int64_t)file_result.get("chunk_index", -1);
                int64_t chunk_start = (int64_t)file_result.get("chunk_start", -1);
                int64_t chunk_end = (int64_t)file_result.get("chunk_end", -1);
                // Suppress unused variable warnings - these may be used for debugging or future features
                (void)similarity;
                (void)chunk_index;
				
				HBoxContainer *file_hbox = memnew(HBoxContainer);
				search_vbox->add_child(file_hbox);
				
				// File link button
				Button *file_link = memnew(Button);
				file_link->set_text(_convert_to_godot_path(file_path));
				file_link->set_flat(true);
				file_link->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
				file_link->set_h_size_flags(Control::SIZE_EXPAND_FILL);
				file_link->add_theme_icon_override("icon", get_theme_icon(SNAME("File"), SNAME("EditorIcons")));
				file_link->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_file_link_pressed).bind(file_path));
				file_hbox->add_child(file_link);
				
                // Hide similarity score for a cleaner UI

                // Try to display scene node/component name for .tscn/.tres sections
                // Extract a header line around chunk_start for readability
                String node_hint;
                if (file_path.ends_with(".tscn") || file_path.ends_with(".tres")) {
                    String fixed_path = _convert_to_godot_path(file_path);
                    String abs_path = ProjectSettings::get_singleton()->globalize_path(fixed_path);
                    Ref<FileAccess> f = FileAccess::open(abs_path, FileAccess::READ);
                    if (f.is_valid()) {
                        // Read up to chunk_end, but only scan nearby lines for section header
                        Vector<String> lines;
                        int64_t max_lines_to_read = chunk_end > 0 ? chunk_end : (chunk_start + 20);
                        for (int64_t ln = 1; ln <= max_lines_to_read && !f->eof_reached(); ln++) {
                            lines.push_back(f->get_line());
                        }
                        // Look backwards from chunk_start for a line that starts a section
                        int64_t idx = MAX((int64_t)0, chunk_start - 1);
                        for (int64_t i = idx; i >= 0 && i >= idx - 20; i--) {
                            if (i < lines.size()) {
                                String l = lines[i].strip_edges();
                                if (l.begins_with("[node")) {
                                    // Extract clean node info: name and type
                                    String clean_hint = "";
                                    int name_idx = l.find("name=\"");
                                    int type_idx = l.find("type=\"");
                                    
                                    if (name_idx != -1) {
                                        int start = name_idx + 6;
                                        int end = l.find("\"", start + 1);
                                        String name = end > start ? l.substr(start + 1, end - start - 1) : String("Node");
                                        clean_hint = "🎯 " + name;
                                    }
                                    
                                    if (type_idx != -1) {
                                        int start = type_idx + 6;
                                        int end = l.find("\"", start + 1);
                                        String type = end > start ? l.substr(start + 1, end - start - 1) : String("Node");
                                        if (!clean_hint.is_empty() && type != "Node") {
                                            clean_hint += " (" + type + ")";
                                        } else if (clean_hint.is_empty()) {
                                            clean_hint = "🎯 " + type;
                                        }
                                    }
                                    
                                    node_hint = clean_hint.is_empty() ? "🎯 Scene Node" : clean_hint;
                                    break;
                                } else if (l.begins_with("[sub_resource")) {
                                    // Clean display: extract just the resource type
                                    int type_idx = l.find("type=\"");
                                    if (type_idx != -1) {
                                        int start = type_idx + 6; // after type="
                                        int end = l.find("\"", start + 1);
                                        String t = end > start ? l.substr(start + 1, end - start - 1) : String("Resource");
                                        node_hint = "🔧 " + t + " Resource";
                                    } else {
                                        node_hint = "🔧 Sub Resource";
                                    }
                                    break;
                                } else if (l.begins_with("[ext_resource")) {
                                    // Clean display for external resources
                                    int type_idx = l.find("type=\"");
                                    if (type_idx != -1) {
                                        int start = type_idx + 6;
                                        int end = l.find("\"", start + 1);
                                        String t = end > start ? l.substr(start + 1, end - start - 1) : String("Resource");
                                        node_hint = "📦 " + t + " (External)";
                                    } else {
                                        node_hint = "📦 External Resource";
                                    }
                                    break;
                                } else if (l.begins_with("[resource")) {
                                    node_hint = "📄 Main Resource";
                                    break;
                                }
                            }
                        }
                    }
                }

                HBoxContainer *meta_box = memnew(HBoxContainer);
                file_hbox->add_child(meta_box);
                if (!node_hint.is_empty()) {
                    Label *node_label = memnew(Label);
                    node_label->set_text(node_hint);
                    node_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.6));
                    meta_box->add_child(node_label);
                }
                // Line range (if present): show as [1-2]
                if (chunk_start >= 0 && chunk_end > chunk_start) {
                    Label *range_label = memnew(Label);
                    range_label->set_text(vformat(" [%d-%d]", (int)chunk_start, (int)chunk_end));
                    range_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.6));
                    meta_box->add_child(range_label);
                }

                // Simple structure view for scenes/resources
                if (p_result.has("graph") && p_result["graph"].get_type() == Variant::DICTIONARY) {
                    Dictionary graph_map = p_result["graph"];
                    if (graph_map.has(file_path) && graph_map[file_path].get_type() == Variant::DICTIONARY) {
                        Dictionary fg = graph_map[file_path];
                        Array g_nodes = fg.get("nodes", Array());
                        Array g_edges = fg.get("edges", Array());

                        // Find scene nodes and connections
                        Array scene_nodes;
                        HashMap<String, String> node_scripts; // node_id -> script_path
                        HashMap<String, Array> connections;  // node_id -> array of connected target names
                        
                        for (int ni = 0; ni < g_nodes.size(); ni++) {
                            Dictionary nd = g_nodes[ni];
                            String nkind = nd.get("kind", "");
                            if (nkind == "SceneNode") {
                                scene_nodes.push_back(nd);
                            }
                        }

                        // Build connection and script maps
                        for (int ei = 0; ei < g_edges.size(); ei++) {
                            Dictionary ed = g_edges[ei];
                            String ek = ed.get("kind", "");
                            String src_id = ed.get("src_id", "");
                            String dst_id = ed.get("dst_id", "");
                            
                            if (ek == "CONNECTS_SIGNAL" && !src_id.is_empty() && !dst_id.is_empty()) {
                                if (!connections.has(src_id)) {
                                    connections[src_id] = Array();
                                }
                                // Find target node name
                                for (int ni = 0; ni < g_nodes.size(); ni++) {
                                    Dictionary nd = g_nodes[ni];
                                    if (nd.get("id", "") == dst_id) {
                                        String target_name = nd.get("name", "unknown");
                                        connections[src_id].push_back(target_name);
                                        break;
                                    }
                                }
                            } else if (ek == "ATTACHES_SCRIPT" && !src_id.is_empty() && !dst_id.is_empty()) {
                                // Find script file path
                                for (int ni = 0; ni < g_nodes.size(); ni++) {
                                    Dictionary nd = g_nodes[ni];
                                    if (nd.get("id", "") == dst_id && nd.get("kind", "") == "File") {
                                        String script_path = nd.get("file_path", "");
                                        if (!script_path.is_empty()) {
                                            node_scripts[src_id] = script_path.get_file(); // Just filename
                                        }
                                        break;
                                    }
                                }
                            }
                        }

                        if (scene_nodes.size() > 0) {
                            // Toggle button - more friendly name
                            Button *structure_toggle = memnew(Button);
                            structure_toggle->set_text("Structure");
                            structure_toggle->set_tooltip_text("Show scene nodes and their connections");
                            structure_toggle->add_theme_icon_override("icon", get_theme_icon(SNAME("SceneTree"), SNAME("EditorIcons")));
                            file_hbox->add_child(structure_toggle);

                            // Collapsible box
                            VBoxContainer *structure_box = memnew(VBoxContainer);
                            structure_box->set_visible(false);
                            search_vbox->add_child(structure_box);

                            // Show scene nodes in a clean tree-like format
                            int max_show = 6;
                            for (int ni = 0; ni < MIN(max_show, scene_nodes.size()); ni++) {
                                Dictionary nd = scene_nodes[ni];
                                String nname = nd.get("name", "");
                                String ntype = nd.get("node_type", "");
                                String npath = nd.get("node_path", "");
                                String node_id = nd.get("id", "");

                                VBoxContainer *node_vbox = memnew(VBoxContainer);
                                structure_box->add_child(node_vbox);

                                HBoxContainer *node_hbox = memnew(HBoxContainer);
                                node_vbox->add_child(node_hbox);

                                // Indent based on path depth
                                int depth = npath.get_slice_count("/") - 1;
                                String indent = "";
                                for (int d = 0; d < depth; d++) {
                                    indent += "  ";
                                }

                                Label *node_label = memnew(Label);
                                String display_text = indent + "📄 " + nname;
                                if (!ntype.is_empty() && ntype != "Node") {
                                    display_text += " (" + ntype + ")";
                                }
                                node_label->set_text(display_text);
                                node_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(0.9, 0.9, 1.0, 0.9));
                                node_hbox->add_child(node_label);

                                // Show script attachment
                                if (node_scripts.has(node_id)) {
                                    Label *script_label = memnew(Label);
                                    script_label->set_text(" 📜 " + node_scripts[node_id]);
                                    script_label->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), SNAME("Editor")) * Color(1, 1, 1, 0.8));
                                    node_hbox->add_child(script_label);
                                }

                                // Show signal connections
                                if (connections.has(node_id)) {
                                    Array targets = connections[node_id];
                                    if (targets.size() > 0) {
                                        HBoxContainer *conn_hbox = memnew(HBoxContainer);
                                        node_vbox->add_child(conn_hbox);
                                        
                                        Label *conn_label = memnew(Label);
                                        String conn_text = indent + "  ↳ signals to: ";
                                        for (int ti = 0; ti < targets.size(); ti++) {
                                            if (ti > 0) conn_text += ", ";
                                            conn_text += String(targets[ti]);
                                        }
                                        conn_label->set_text(conn_text);
                                        conn_label->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
                                        conn_hbox->add_child(conn_label);
                                    }
                                }
                            }

                            if (scene_nodes.size() > max_show) {
                                Label *more_label = memnew(Label);
                                more_label->set_text("... and " + String::num_int64(scene_nodes.size() - max_show) + " more nodes");
                                more_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.5));
                                structure_box->add_child(more_label);
                            }

                            structure_toggle->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_output_toggled).bind(structure_box));
                        }
                    }
                }
			}
		}
		
		// Central files section
		if (central_files.size() > 0) {
			search_vbox->add_child(memnew(HSeparator));
			
			Label *central_header = memnew(Label);
			central_header->set_text("⭐ Central Files (" + String::num_int64(central_files.size()) + ")");
			central_header->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
			search_vbox->add_child(central_header);
			
			for (int i = 0; i < central_files.size(); i++) {
				Dictionary central_file = central_files[i];
				String file_path = central_file.get("file_path", "");
				float centrality = central_file.get("centrality", 0.0);
				
				HBoxContainer *central_hbox = memnew(HBoxContainer);
				search_vbox->add_child(central_hbox);
				
				// File link button
				Button *file_link = memnew(Button);
				file_link->set_text(_convert_to_godot_path(file_path));
				file_link->set_flat(true);
				file_link->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
				file_link->set_h_size_flags(Control::SIZE_EXPAND_FILL);
				file_link->add_theme_icon_override("icon", get_theme_icon(SNAME("File"), SNAME("EditorIcons")));
				file_link->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_file_link_pressed).bind(file_path));
				central_hbox->add_child(file_link);
				
				// Centrality score
				Label *centrality_label = memnew(Label);
				centrality_label->set_text("Centrality: " + String::num(centrality, 3));
				centrality_label->add_theme_color_override("font_color", get_theme_color(SNAME("warning_color"), SNAME("Editor")));
				centrality_label->set_custom_minimum_size(Size2(120, 0));
				central_hbox->add_child(centrality_label);
			}
		}
		
		// Graph summary section
		if (!graph_summary.is_empty()) {
			search_vbox->add_child(memnew(HSeparator));
			
			Label *graph_header = memnew(Label);
			graph_header->set_text("🔗 Project Graph Summary");
			graph_header->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
			search_vbox->add_child(graph_header);
			
			int total_files = graph_summary.get("total_files", 0);
			int total_connections = graph_summary.get("total_connections", 0);
			
			Label *summary_label = memnew(Label);
			summary_label->set_text("Files: " + String::num_int64(total_files) + " • Connections: " + String::num_int64(total_connections));
			summary_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.8));
			search_vbox->add_child(summary_label);
		}
		
	} else {
		// Default case - show formatted JSON output with better styling
		VBoxContainer *json_container = memnew(VBoxContainer);
		p_content_vbox->add_child(json_container);
		
		Label *json_header = memnew(Label);
		json_header->set_text("Tool Result Data:");
		json_header->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
		json_header->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
		json_container->add_child(json_header);
		
		RichTextLabel *content_label = memnew(RichTextLabel);
		content_label->add_theme_font_override("normal_font", get_theme_font(SNAME("source"), SNAME("EditorFonts")));
		content_label->set_text(json->stringify(p_result, "  "));
		content_label->set_fit_content(true);
		content_label->set_selection_enabled(true);
		content_label->set_custom_minimum_size(Size2(0, 150));
		json_container->add_child(content_label);
	}
	
	// If the tool provided a path_to_save for images, save it now (non-intrusive for other tools)
	if (p_tool_name == "image_operation" && p_success) {
		String path_to_save = p_args.get("path_to_save", "");
		if (path_to_save.is_empty()) path_to_save = p_args.get("path", "");
		String image_b64 = p_result.get("image_data", "");
		if (!path_to_save.is_empty() && !image_b64.is_empty()) {
			bool ok = _save_base64_image_to_path(image_b64, path_to_save);
			if (ok) {
				Label *saved = memnew(Label);
				saved->set_text("Saved image: " + path_to_save.get_file());
				saved->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
				p_content_vbox->add_child(saved);
			} else {
				Label *err = memnew(Label);
				err->set_text("Failed to save image to: " + path_to_save);
				err->add_theme_color_override("font_color", get_theme_color(SNAME("error_color"), SNAME("Editor")));
				p_content_vbox->add_child(err);
			}
		}
	}
}
void AIChatDock::_on_apply_preview_to_editor(const String &p_path, const String &p_content, const NodePath &p_btns_path, const NodePath &p_status_label_path) {
    if (p_path.is_empty()) {
        // Disable buttons in the corresponding tool bubble and keep content visible in view mode
        if (!p_btns_path.is_empty() && has_node(p_btns_path)) {
            if (HBoxContainer *btns = Object::cast_to<HBoxContainer>(get_node(p_btns_path))) {
                btns->set_visible(false);
            }
        }
        return;
    }
    // Decide based on file extension: scripts vs non-script resources (.tres, .tscn, etc.)
    String ext = p_path.get_extension().to_lower();
    bool is_script_like = (ext == "gd" || ext == "cs" || ext == "shader" || ext == "glsl");

    if (!is_script_like) {
        // Non-script resource: write immediately and open in inspector for review
        _apply_file_edit_immediate(p_path, p_content);
        Ref<Resource> res = ResourceLoader::load(p_path);
        if (res.is_valid()) {
            EditorNode::get_singleton()->edit_resource(res);
        }
        // Hide buttons and mark status as Accepted
        if (!p_btns_path.is_empty() && has_node(p_btns_path)) {
            if (HBoxContainer *btns = Object::cast_to<HBoxContainer>(get_node(p_btns_path))) {
                btns->set_visible(false);
            }
        }
        if (!p_status_label_path.is_empty() && has_node(p_status_label_path)) {
            if (Label *lbl = Object::cast_to<Label>(get_node(p_status_label_path))) {
                String t = lbl->get_text();
                if (!t.ends_with(" — Accepted")) {
                    lbl->set_text(t + " — Accepted");
                }
            }
        }
        return;
    }

    // Script-like handling (existing behavior)
    ScriptEditor *script_editor = ScriptEditor::get_singleton();
    if (!script_editor) {
        return;
    }
    Ref<Resource> resource = ResourceLoader::load(p_path);
    Ref<Script> script = resource;
    if (!script.is_valid()) {
        // Create a temporary script resource to host the text
        Ref<GDScript> tmp_script;
        tmp_script.instantiate();
        tmp_script->set_path_cache(p_path);
        tmp_script->set_source_code(p_content);
        script = tmp_script;
        script_editor->edit(script);
    } else {
        script_editor->edit(script);
        script->set_source_code(p_content);
    }

    // Save the script immediately to disk
    if (script.is_valid() && resource.is_valid()) {
        // Remove from cache first to ensure fresh reload
        GDScriptCache::remove_script(p_path);

        EditorNode::get_singleton()->save_resource(resource);

        // Force complete reload
        script->reload(true);

        // Trigger reload to ensure all caches are updated
        script_editor->trigger_live_script_reload(p_path);
        script_editor->reload_scripts(false);
    }

    ScriptTextEditor *ste = Object::cast_to<ScriptTextEditor>(script_editor->get_current_editor());
    if (ste) {
        // Do not clear inline diff view here; ScriptTextEditor now manages persistence until accept/reject
        
        CodeTextEditor *code_editor = ste->get_code_editor();
        if (code_editor) {
            CodeEdit *text_editor = code_editor->get_text_editor();
            if (text_editor) {
                // Preserve caret when possible
                int caret_line = text_editor->get_caret_line();
                int caret_col = text_editor->get_caret_column();
                text_editor->set_text(p_content);
                if (caret_line < text_editor->get_line_count()) {
                    text_editor->set_caret_line(caret_line);
                    int max_col = text_editor->get_line(caret_line).length();
                    text_editor->set_caret_column(MIN(caret_col, max_col));
                }
                // Mark as saved version so it doesn't show as modified
                text_editor->tag_saved_version();
            }
        }
        // Validate to refresh errors and clear stale diagnostics
        ste->validate();
    }
    // Once applied to the editor buffer, overlay is no longer needed
    EditorTools::clear_preview_overlay(p_path);
    // Switch to Script main screen
    EditorInterface::get_singleton()->set_main_screen_editor("Script");
    // Disable buttons in the corresponding tool bubble and keep content visible in view mode
    if (!p_btns_path.is_empty() && has_node(p_btns_path)) {
        if (HBoxContainer *btns = Object::cast_to<HBoxContainer>(get_node(p_btns_path))) {
            btns->set_visible(false);
        }
    }
    if (!p_status_label_path.is_empty() && has_node(p_status_label_path)) {
        if (Label *lbl = Object::cast_to<Label>(get_node(p_status_label_path))) {
            String t = lbl->get_text();
            if (!t.ends_with(" — Accepted")) {
                lbl->set_text(t + " — Accepted");
            }
        }
    }
    // Clear from pending edits tracking
    _clear_pending_edit(p_path);
}

void AIChatDock::_on_discard_preview(const String &p_path, const NodePath &p_btns_path, const NodePath &p_status_label_path) {
    if (p_path.is_empty()) return;
    EditorTools::clear_preview_overlay(p_path);
    // Collapse to view mode: hide buttons
    if (!p_btns_path.is_empty() && has_node(p_btns_path)) {
        if (HBoxContainer *btns = Object::cast_to<HBoxContainer>(get_node(p_btns_path))) {
            btns->set_visible(false);
        }
    }
    if (!p_status_label_path.is_empty() && has_node(p_status_label_path)) {
        if (Label *lbl = Object::cast_to<Label>(get_node(p_status_label_path))) {
            String t = lbl->get_text();
            if (!t.ends_with(" — Discarded")) {
                lbl->set_text(t + " — Discarded");
            }
        }
    }
    // Clear from pending edits tracking
    _clear_pending_edit(p_path);
}

void AIChatDock::_toggle_expand_label(RichTextLabel *p_label, Button *p_button, const String &p_full_text, const String &p_snippet_text) {
    if (!p_label || !p_button) {
        return;
    }
    String current = p_label->get_text();
    bool showing_snippet = (current == p_snippet_text);
    if (showing_snippet) {
        p_label->set_text(p_full_text);
        p_button->set_text("Show less");
        p_label->set_custom_minimum_size(Size2(0, 240));
    } else {
        p_label->set_text(p_snippet_text);
        p_button->set_text("Show more");
        p_label->set_custom_minimum_size(Size2(0, 120));
    }
}

void AIChatDock::_toggle_docs_card(Control *p_snippet_node, Control *p_full_node, Button *p_button, VBoxContainer *p_holder) {
    if (!p_snippet_node || !p_full_node || !p_button) {
        return;
    }
    bool expanding = p_snippet_node->is_visible();
    p_snippet_node->set_visible(!expanding);
    p_full_node->set_visible(expanding);
    p_button->set_text(expanding ? "Show less" : "Show more");
    // Nudge layout to recompute height
    if (p_holder) {
        p_holder->queue_redraw();
    }
}

void AIChatDock::_on_filesystem_debounced_scan(uint64_t p_scheduled_at) {
    static uint64_t s_last_scan_request_ms = 0; // mirrors writer-side variable
    (void)s_last_scan_request_ms; // Suppress unused variable warning - may be used in future
    // Only perform scan if this timer matches the last scheduled time to avoid multiple scans.
    uint64_t now = OS::get_singleton()->get_ticks_msec();
    // A small guard window to ensure newest request wins.
    if (p_scheduled_at + 200 <= now) {
        // Proceed with scan on the main thread using deferred call to reduce re-entrancy risk.
        call_deferred("_perform_filesystem_scan_changes");
    }
}

void AIChatDock::_perform_filesystem_scan_changes() {
    if (EditorFileSystem::get_singleton()) {
        EditorFileSystem::get_singleton()->scan_changes();
    }
}

void AIChatDock::_rebuild_conversation_ui(const Vector<ChatMessage> &p_messages) {
	// Build a mapping of tool call IDs to their results for efficient lookup
	HashMap<String, ChatMessage> tool_results;
	for (int i = 0; i < p_messages.size(); i++) {
		const ChatMessage &msg = p_messages[i];
		if (msg.role == "tool" && !msg.tool_call_id.is_empty()) {
			tool_results[msg.tool_call_id] = msg;
		}
	}
	
	// First pass: Create all non-tool messages (this includes assistant messages with tool_calls)
	for (int i = 0; i < p_messages.size(); i++) {
		const ChatMessage &msg = p_messages[i];
		if (msg.role != "tool") {
			_create_message_bubble(msg, i);
		}
	}
	
	// Second pass: Apply tool results to their corresponding placeholders
	for (int i = 0; i < p_messages.size(); i++) {
		const ChatMessage &msg = p_messages[i];
		if (msg.role == "tool" && !msg.tool_call_id.is_empty()) {
			// Use a small delay to ensure the UI has been updated from the first pass
			call_deferred("_apply_tool_result_deferred", msg.tool_call_id, msg.name, msg.content, msg.tool_results);
		}
	}
}
void AIChatDock::_apply_tool_result_deferred(const String &p_tool_call_id, const String &p_tool_name, const String &p_content, const Array &p_tool_results) {
	// Find the placeholder for this tool call ID
	if (chat_container == nullptr) {
		return;
	}
	
	if (tool_calls_ui_applied.has(p_tool_call_id)) {
		return; // already applied
	}

	PanelContainer *placeholder = Object::cast_to<PanelContainer>(chat_container->find_child("tool_placeholder_" + p_tool_call_id, true, false));
	if (!placeholder) {
		// Assistant message may not be fully constructed yet; schedule a bounded retry via timer
		int tries = (int)tool_result_retry_counts.get(p_tool_call_id, 0);
		if (tries >= 5) {
			print_line("AI Chat: Max retries reached for tool id " + p_tool_call_id + ", skipping UI update");
			tool_result_retry_counts.erase(p_tool_call_id);
			return;
		}
		tool_result_retry_counts[p_tool_call_id] = tries + 1;
		Ref<SceneTreeTimer> timer = get_tree()->create_timer(0.05, true);
		timer->connect("timeout", callable_mp(this, &AIChatDock::_on_tool_result_retry_timeout).bind(p_tool_call_id, p_tool_name, p_content, p_tool_results));
		return;
	}

	// Clear the "loading" text or any existing content immediately
	while (placeholder->get_child_count() > 0) {
		Node *child = placeholder->get_child(0);
		placeholder->remove_child(child);
		child->queue_free();
	}

	// Parse the tool result from JSON content
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(p_content);
	Dictionary result;
	if (err == OK) {
		result = json->get_data();
	} else {
		// Fallback if JSON parsing fails
		result["success"] = false;
		result["message"] = "Failed to parse tool result: " + p_content;
	}

	// Extract arguments if they were stored
	Dictionary args;
	if (p_tool_results.size() > 1) {
		args = p_tool_results[1]; // Args are stored as second element
	}

	// Create the tool result UI
	VBoxContainer *tool_container = memnew(VBoxContainer);
	placeholder->add_child(tool_container);

	Button *toggle_button = memnew(Button);
	
	// Show success/failure status in the button
	bool success = result.get("success", false);
	String message = result.get("message", "");
	String status_text = success ? "SUCCESS" : "ERROR";
	toggle_button->set_text(status_text + " - " + p_tool_name + ": " + message);
	
	toggle_button->set_flat(false);
	toggle_button->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	toggle_button->set_text_alignment(HORIZONTAL_ALIGNMENT_LEFT);
	// Prevent long status lines from expanding the dock width
	toggle_button->set_clip_text(true);
	toggle_button->set_text_overrun_behavior(TextServer::OVERRUN_TRIM_ELLIPSIS);
	// Preserve full text in tooltip for accessibility
	toggle_button->set_tooltip_text(toggle_button->get_text());
	toggle_button->add_theme_icon_override("icon", get_theme_icon(success ? SNAME("StatusSuccess") : SNAME("StatusError"), SNAME("EditorIcons")));
	toggle_button->add_theme_color_override("font_color", success ? get_theme_color(SNAME("success_color"), SNAME("Editor")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
	tool_container->add_child(toggle_button);

	PanelContainer *content_panel = memnew(PanelContainer);
	content_panel->set_visible(false); // Collapsed by default
	content_panel->set_clip_contents(true);
	tool_container->add_child(content_panel);
	toggle_button->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_output_toggled).bind(content_panel));

	Ref<StyleBoxFlat> content_style = memnew(StyleBoxFlat);
	content_style->set_bg_color(get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
	content_style->set_border_width_all(1);
	content_style->set_border_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
	content_style->set_content_margin_all(10);
	content_panel->add_theme_style_override("panel", content_style);

	VBoxContainer *content_vbox = memnew(VBoxContainer);
	content_panel->add_child(content_vbox);

	HBoxContainer *header_hbox = memnew(HBoxContainer);
	content_vbox->add_child(header_hbox);

	Label *status_label = memnew(Label);
	status_label->set_text(success ? "Tool Succeeded" : "Tool Failed");
	status_label->add_theme_color_override("font_color", success ? get_theme_color(SNAME("success_color"), SNAME("Editor")) : get_theme_color(SNAME("error_color"), SNAME("Editor")));
	status_label->add_theme_icon_override("icon", get_theme_icon(success ? SNAME("StatusSuccess") : SNAME("StatusError"), SNAME("EditorIcons")));
	header_hbox->add_child(status_label);

	content_vbox->add_child(memnew(HSeparator));

	// Create specific UI based on the tool that was called
	_create_tool_specific_ui(content_vbox, p_tool_name, result, success, args);

	tool_calls_ui_applied.insert(p_tool_call_id);
}

void AIChatDock::_on_tool_output_toggled(Control *p_content) {
	p_content->set_visible(!p_content->is_visible());
}
void AIChatDock::_send_chat_request() {
	Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
	
	// For large conversation histories, process in chunks to prevent UI blocking
	if (chat_history.size() > 50) {
		print_line("AI Chat: Large conversation detected (" + String::num_int64(chat_history.size()) + " messages), processing in chunks");
		call_deferred("_send_chat_request_chunked", 0);
		return;
	}
	
	// Build the messages array for the API
	Array messages;
	messages.resize(chat_history.size());
	
	// Process messages efficiently for smaller conversations
	for (int i = 0; i < chat_history.size(); i++) {
		const ChatMessage &msg = chat_history[i];
		Dictionary api_msg;
		api_msg["role"] = msg.role;
		
		// For user messages with attached files, handle images and text differently
		if (msg.role == "user" && !msg.attached_files.is_empty()) {
			// Check if we have any images
			bool has_images = false;
			for (const AttachedFile &file : msg.attached_files) {
				if (file.is_image) {
					has_images = true;
					break;
				}
			}
			
			if (has_images) {
				// Use the new multi-modal format for images
				Array content_array;
				
				// Add text content
				if (!msg.content.is_empty()) {
					Dictionary text_part;
					text_part["type"] = "text";
					text_part["text"] = msg.content;
					content_array.push_back(text_part);
				}
				
							// Add images and text files
			for (const AttachedFile &file : msg.attached_files) {
				if (file.is_image) {
					Dictionary image_part;
					image_part["type"] = "image_url";
					Dictionary image_url;
					image_url["url"] = "data:" + file.mime_type + ";base64," + file.base64_data;
					image_part["image_url"] = image_url;
					content_array.push_back(image_part);
					
            // Add image ID marker; and promote to top-level 'images' for backend tool context
            Dictionary text_part;
            text_part["type"] = "text";
            text_part["text"] = "\n*[Image ID: " + file.name + "]*";
            content_array.push_back(text_part);
				} else {
					// Add text files as text content
					Dictionary text_part;
					text_part["type"] = "text";
					text_part["text"] = "\n\n**File: " + file.name + " (" + file.path + ")**\n```\n" + file.content + "\n```\n";
					content_array.push_back(text_part);
				}
			}
				
				api_msg["content"] = content_array;
			} else {
				// Legacy format for text-only attachments
				String combined_content = msg.content;
				combined_content += "\n\n**Attached Files:**\n";
				
				for (const AttachedFile &file : msg.attached_files) {
					combined_content += "\n### " + file.name + " (" + file.path + ")\n";
					combined_content += "```\n" + file.content + "\n```\n";
				}
				
				api_msg["content"] = combined_content;
			}
		} else {
			// Handle assistant messages with attached files (like generated images)
			if (msg.role == "assistant" && !msg.attached_files.is_empty()) {
				// Check if we have any images  
				bool has_images = false;
				for (const AttachedFile &file : msg.attached_files) {
					if (file.is_image) {
						has_images = true;
						break;
					}
				}
				
				if (has_images) {
								// For assistant messages with images, we need to structure them for OpenAI
			// Since OpenAI doesn't support assistant messages with image content,
			// we'll include the images as a special marker in the content
			String content_with_images = msg.content;
			content_with_images += "\n\n**Generated Images:**";
			
			for (const AttachedFile &file : msg.attached_files) {
				if (file.is_image) {
					content_with_images += "\n- Image ID: `" + file.name + "`";
				}
			}
					
					api_msg["content"] = content_with_images;
					
                    // Include generated image data for backend-only image editing context.
                    // This is NOT sent to OpenAI by the backend; it's only used by /chat tool execution.
                    Array images_data;
                    for (const AttachedFile &file : msg.attached_files) {
                        if (file.is_image) {
                            Dictionary image_info;
                            image_info["name"] = file.name;
                            image_info["mime_type"] = file.mime_type;
                            image_info["base64_data"] = file.base64_data;
                            image_info["original_size"] = Vector2(file.original_size.x, file.original_size.y);
                            images_data.push_back(image_info);
                        }
                    }
                    if (!images_data.is_empty()) {
                        api_msg["images"] = images_data;
                    }
				} else {
					api_msg["content"] = msg.content;
				}
			} else {
				api_msg["content"] = msg.content;
			}
		}
		
        if (msg.role == "assistant" && !msg.tool_calls.is_empty()) {
			api_msg["tool_calls"] = msg.tool_calls;
		}
        if (msg.role == "tool") {
            api_msg["tool_call_id"] = msg.tool_call_id;
            api_msg["name"] = msg.name;
            // Only strip image_data for image_operation; keep full content for other tools
            if (msg.name == "image_operation") {
                Dictionary raw;
                if (!msg.tool_results.is_empty()) {
                    raw = msg.tool_results[0];
                } else {
                    Ref<JSON> json_tool;
                    json_tool.instantiate();
                    if (json_tool->parse(msg.content) == OK) {
                        raw = json_tool->get_data();
                    }
                }
                // Promote assistant images to top-level 'images' so backend edits can find them
                Array images_data;
                for (const AttachedFile &file : msg.attached_files) {
                    if (file.is_image) {
                        Dictionary image_info;
                        image_info["name"] = file.name;
                        image_info["mime_type"] = file.mime_type;
                        image_info["base64_data"] = file.base64_data;
                        image_info["original_size"] = Vector2(file.original_size.x, file.original_size.y);
                        images_data.push_back(image_info);
                    }
                }
                if (!images_data.is_empty()) {
                    api_msg["images"] = images_data;
                }
                raw.erase("image_data");
                Ref<JSON> json_min;
                json_min.instantiate();
                api_msg["content"] = json_min->stringify(raw);
            } else {
                // Non-image tools: pass original content
                api_msg["content"] = msg.content;
            }
        }
		messages.push_back(api_msg);
	}

	// Store messages and route to finalize method
	_chunked_messages = messages;
	call_deferred("_finalize_chat_request");
}

void AIChatDock::_send_chat_request_chunked(int p_start_index) {
	Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
	
	// Process messages in chunks of 10 to avoid blocking UI
	const int CHUNK_SIZE = 10;
	int end_index = MIN(p_start_index + CHUNK_SIZE, chat_history.size());
	
	// If this is the first chunk, initialize the messages array
	if (p_start_index == 0) {
		// Store the work in progress
		_chunked_messages.clear();
		_chunked_messages.resize(chat_history.size());
	}
	
	// Process this chunk
	for (int i = p_start_index; i < end_index; i++) {
		const ChatMessage &msg = chat_history[i];
		Dictionary api_msg = _build_api_message(msg);
		_chunked_messages[i] = api_msg;
	}
	
	print_line("AI Chat: Processed message chunk " + String::num_int64(p_start_index) + "-" + String::num_int64(end_index-1) + " of " + String::num_int64(chat_history.size()));
	
	// If more chunks to process, defer the next chunk
	if (end_index < chat_history.size()) {
		call_deferred("_send_chat_request_chunked", end_index);
		return;
	}
	
	// All chunks processed, now send the request
	call_deferred("_finalize_chat_request");
}
Dictionary AIChatDock::_build_api_message(const ChatMessage &p_msg) {
	Dictionary api_msg;
	api_msg["role"] = p_msg.role;
	
	// For user messages with attached files, handle images and text differently
	if (p_msg.role == "user" && !p_msg.attached_files.is_empty()) {
		// Check if we have any images
		bool has_images = false;
		for (const AttachedFile &file : p_msg.attached_files) {
			if (file.is_image) {
				has_images = true;
				break;
			}
		}
		
		if (has_images) {
			// Use the new multi-modal format for images
			Array content_array;
			
			// Add text content
			if (!p_msg.content.is_empty()) {
				Dictionary text_part;
				text_part["type"] = "text";
				text_part["text"] = p_msg.content;
				content_array.push_back(text_part);
			}
			
					// Add images and text files
		for (const AttachedFile &file : p_msg.attached_files) {
			if (file.is_image) {
				Dictionary image_part;
				image_part["type"] = "image_url";
				Dictionary image_url;
				image_url["url"] = "data:" + file.mime_type + ";base64," + file.base64_data;
				image_part["image_url"] = image_url;
				content_array.push_back(image_part);
				
				// Add image ID as text so AI knows how to reference it
				Dictionary text_part;
				text_part["type"] = "text";
				text_part["text"] = "\n*[Image ID: " + file.name + "]*";
				content_array.push_back(text_part);
			} else {
				// Add text files as text content
				Dictionary text_part;
				text_part["type"] = "text";
				text_part["text"] = "\n\n**File: " + file.name + " (" + file.path + ")**\n```\n" + file.content + "\n```\n";
				content_array.push_back(text_part);
			}
		}
			
			api_msg["content"] = content_array;
		} else {
			// Legacy format for text-only attachments
			String combined_content = p_msg.content;
			combined_content += "\n\n**Attached Files:**\n";
			
			for (const AttachedFile &file : p_msg.attached_files) {
				combined_content += "\n### " + file.name + " (" + file.path + ")\n";
				combined_content += "```\n" + file.content + "\n```\n";
			}
			
			api_msg["content"] = combined_content;
		}
	} else {
		// Handle assistant messages with attached files (like generated images)
		if (p_msg.role == "assistant" && !p_msg.attached_files.is_empty()) {
			// Check if we have any images  
			bool has_images = false;
			for (const AttachedFile &file : p_msg.attached_files) {
				if (file.is_image) {
					has_images = true;
					break;
				}
			}
			
			if (has_images) {
						// For assistant messages with images, we need to structure them for OpenAI
		// Since OpenAI doesn't support assistant messages with image content,
		// we'll include the images as a special marker in the content
		String content_with_images = p_msg.content;
		content_with_images += "\n\n**Generated Images:**";
		
		for (const AttachedFile &file : p_msg.attached_files) {
			if (file.is_image) {
				content_with_images += "\n- Image ID: `" + file.name + "`";
			}
		}
				
				api_msg["content"] = content_with_images;
				
                // For assistant messages with images, include 'images' array so backend image editing has context
                Array images_data;
                for (const AttachedFile &file : p_msg.attached_files) {
                    if (file.is_image) {
                        Dictionary image_info;
                        image_info["name"] = file.name;
                        image_info["mime_type"] = file.mime_type;
                        image_info["base64_data"] = file.base64_data;
                        image_info["original_size"] = Vector2(file.original_size.x, file.original_size.y);
                        images_data.push_back(image_info);
                    }
                }
                if (!images_data.is_empty()) {
                    api_msg["images"] = images_data;
                }
			} else {
				api_msg["content"] = p_msg.content;
			}
		} else {
			api_msg["content"] = p_msg.content;
		}
	}
	
	// Save attached files for ALL message types (user, assistant, etc.)
    if (!p_msg.attached_files.is_empty()) {
		Array attached_files_array;
		for (const AttachedFile &file : p_msg.attached_files) {
			Dictionary file_dict;
			file_dict["path"] = file.path;
			file_dict["name"] = file.name;
			file_dict["content"] = file.content;
			file_dict["is_image"] = file.is_image;
			file_dict["mime_type"] = file.mime_type;
            // Only include raw base64 for user attachments (inputs), never for assistant/tool outputs
            if (p_msg.role == "user") {
                file_dict["base64_data"] = file.base64_data;
            }
			file_dict["original_size_x"] = file.original_size.x;
			file_dict["original_size_y"] = file.original_size.y;
			file_dict["display_size_x"] = file.display_size.x;
			file_dict["display_size_y"] = file.display_size.y;
			file_dict["was_downsampled"] = file.was_downsampled;
			attached_files_array.push_back(file_dict);
		}
		api_msg["attached_files"] = attached_files_array;
	}
	
	// Save tool calls and tool data
	if (p_msg.role == "assistant" && !p_msg.tool_calls.is_empty()) {
		api_msg["tool_calls"] = p_msg.tool_calls;
	}
	if (p_msg.role == "tool") {
		api_msg["tool_call_id"] = p_msg.tool_call_id;
		api_msg["name"] = p_msg.name;
        if (p_msg.name == "image_operation") {
            Dictionary raw;
            if (!p_msg.tool_results.is_empty()) {
                raw = p_msg.tool_results[0];
            }
			raw.erase("image_data");
			Ref<JSON> json_min;
			json_min.instantiate();
			api_msg["content"] = json_min->stringify(raw);
		} else {
			api_msg["content"] = p_msg.content;
		}
	}
	
	// Save tool results
	if (!p_msg.tool_results.is_empty()) {
		api_msg["tool_results"] = p_msg.tool_results;
	}
	
	// Save timestamp
	if (!p_msg.timestamp.is_empty()) {
		api_msg["timestamp"] = p_msg.timestamp;
	}
	
	return api_msg;
}

void AIChatDock::_finalize_chat_request() {
	// Build final request data
	Dictionary request_data;
	request_data["messages"] = _chunked_messages;
	request_data["model"] = model;

	// Debug logs for OpenAI messages have been quieted to reduce console noise.

	Ref<JSON> json;
	json.instantiate();
	String request_body = json->stringify(request_data);

	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	// Add authentication headers
	if (!auth_token.is_empty()) {
		headers.push_back("Authorization: Bearer " + auth_token);
	}
    headers.push_back("X-User-ID: " + current_user_id);
	headers.push_back("X-Machine-ID: " + get_machine_id());
    // Always pass current project root to ensure backend targets the right project
    headers.push_back("X-Project-Root: " + ProjectSettings::get_singleton()->globalize_path("res://"));

	// Setup HTTPClient for streaming
	http_client->set_read_chunk_size(4096); // 4kb chunk size

	// Extract host and port from endpoint
	String host = api_endpoint;
	int port = 80;
	bool use_ssl = false;

	if (host.begins_with("https://")) {
		host = host.trim_prefix("https://");
		use_ssl = true;
		port = 443;
	} else if (host.begins_with("http://")) {
		host = host.trim_prefix("http://");
	}

	String path = "/";
	if (host.find("/") != -1) {
		path = host.substr(host.find("/"), -1);
		host = host.substr(0, host.find("/"));
	}

	if (host.find(":") != -1) {
		port = host.substr(host.find(":") + 1, -1).to_int();
		host = host.substr(0, host.find(":"));
	}

    PackedByteArray request_body_data = request_body.to_utf8_buffer();
    print_line("AI Chat: Attempting to connect to " + host + ":" + String::num_int64(port) + path);
    Ref<TLSOptions> tls_options;
    if (use_ssl) {
        // Use default platform CA and standard hostname verification.
        tls_options = TLSOptions::client();
    }
    Error err = http_client->connect_to_host(host, port, tls_options);
	if (err != OK) {
		_add_message_to_chat("system", "Failed to connect to backend: " + host + ":" + String::num_int64(port) + " (Error: " + String::num_int64(err) + ")");
		is_waiting_for_response = false;
		_update_ui_state();
		return;
	}

	http_status = STATUS_CONNECTING;
	pending_request_path = path;
	pending_request_headers = headers;
	pending_request_body = request_body_data;

	// Clear response buffer for new request
	response_buffer.clear();

	set_process(true);
	
	// Clear the chunked messages to free memory
	_chunked_messages.clear();
}

void AIChatDock::_update_ui_state() {
	// Handle send button state
	send_button->set_disabled(input_field->get_text().strip_edges().is_empty() || is_waiting_for_response);
	input_field->set_editable(!is_waiting_for_response);

	// Handle stop button state
	if (is_waiting_for_response) {
		// Show stop button and hide/disable send button during request
		send_button->set_visible(false);
		stop_button->set_visible(true);
		bool should_disable_stop = current_request_id.is_empty();
		stop_button->set_disabled(should_disable_stop); // Only enable if we have a request ID
		
		print_line("AI Chat: UI State - waiting for response, stop button visible=" + String(stop_button->is_visible() ? "true" : "false") + ", disabled=" + String(should_disable_stop ? "true" : "false") + ", request_id='" + current_request_id + "'");
		
		// Also disable new conversation button during processing
		if (new_conversation_button) {
			new_conversation_button->set_disabled(true);
		}
	} else {
		// Show send button and hide stop button when not waiting
		send_button->set_visible(true);
		send_button->set_text("Send");
		stop_button->set_visible(false);
		
		print_line("AI Chat: UI State - not waiting, send button visible, stop button hidden");
		
		if (new_conversation_button) {
			new_conversation_button->set_disabled(false);
		}
	}
}


void AIChatDock::_request_completed() {
	stream_completed_successfully = true;
	print_line("AI Chat: Server signaled end of stream");

	// Finalize/save regardless of async state
	if (current_conversation_index >= 0) {
		conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
		_queue_delayed_save();
	}

	// If async frontend tools (e.g., apply_edit) are still running, keep UI in waiting state
	bool has_async_work = pending_tool_tasks > 0;
	is_waiting_for_response = has_async_work ? true : false;
	// Preserve the last valid request_id if async work is running so the Stop button stays enabled.
	if (!has_async_work) {
		stop_requested = false;
		current_request_id = "";
	}
	_update_ui_state();

	// Ensure the HTTP connection is closed so the poll loop exits promptly.
	if (http_client.is_valid()) {
		http_client->close();
	}
	current_assistant_message_label = nullptr;
	set_process(false);
}

String AIChatDock::_get_timestamp() {
	Dictionary time_dict = Time::get_singleton()->get_datetime_dict_from_system();
	return String::num_int64(time_dict["hour"]).pad_zeros(2) + ":" +
		   String::num_int64(time_dict["minute"]).pad_zeros(2);
}
String AIChatDock::_process_inline_markdown(String p_line) {
	String line = p_line;

	// Bold (**text** or __text__)
	while (true) {
		int start = line.find("**");
		if (start == -1) {
			start = line.find("__");
			if (start == -1) {
				break;
			}
		}

		String marker = line.substr(start, 2);
		int end = line.find(marker, start + 2);
		if (end == -1) {
			break;
		}

		String before = line.substr(0, start);
		String bold_text = line.substr(start + 2, end - start - 2);
		String after = line.substr(end + 2);

		line = before + "[b]" + bold_text + "[/b]" + after;
	}

	// Italic (*text* or _text_) - but not inside ** or __
	int pos = 0;
	while (pos < line.length()) {
		int star_pos = line.find("*", pos);
		int underscore_pos = line.find("_", pos);

		int start = -1;
		String marker;

		if (star_pos != -1 && (underscore_pos == -1 || star_pos < underscore_pos)) {
			// Check it's not part of **
			if ((star_pos > 0 && line[star_pos - 1] == '*') || (star_pos < line.length() - 1 && line[star_pos + 1] == '*')) {
				pos = star_pos + 1;
				continue;
			}
			start = star_pos;
			marker = "*";
		} else if (underscore_pos != -1) {
			// Check it's not part of __
			if ((underscore_pos > 0 && line[underscore_pos - 1] == '_') || (underscore_pos < line.length() - 1 && line[underscore_pos + 1] == '_')) {
				pos = underscore_pos + 1;
				continue;
			}
			start = underscore_pos;
			marker = "_";
		}

		if (start == -1) {
			break;
		}

		int end = line.find(marker, start + 1);
		if (end == -1) {
			pos = start + 1;
			continue;
		}

		String before = line.substr(0, start);
		String italic_text = line.substr(start + 1, end - start - 1);
		String after = line.substr(end + 1);

		line = before + "[i]" + italic_text + "[/i]" + after;
		pos = before.length() + 3 + italic_text.length() + 4; // Skip past [i]...[/i]
	}

	// Inline code (`text`)
	while (true) {
		int start = line.find("`");
		if (start == -1) {
			break;
		}

		int end = line.find("`", start + 1);
		if (end == -1) {
			break;
		}

		String before = line.substr(0, start);
		String code_text = line.substr(start + 1, end - start - 1);
		String after = line.substr(end + 1);

		line = before + "[code]" + code_text + "[/code]" + after;
	}

	return line;
}
String AIChatDock::_markdown_to_bbcode(const String &p_markdown) {
	// Safety check for empty strings
	if (p_markdown.is_empty()) {
		return String();
	}

	Vector<String> lines = p_markdown.split("\n");
	String result = "";
	bool in_code_block = false;

	for (int i = 0; i < lines.size(); i++) {
		String line = lines[i];

		if (line.strip_edges().begins_with("```")) {
			in_code_block = !in_code_block;
			if (in_code_block) {
				result += "[code]";
			} else {
				result += "[/code]";
			}
		} else if (in_code_block) {
			result += line.xml_escape(); // Escape to prevent BBCode parsing inside code blocks.
		} else {
			if (line.strip_edges().is_empty()) {
				// Preserve blank lines between paragraphs.
				result += "";
			} else {
				String processed_line = line;
				String trimmed_line = processed_line.lstrip(" \t");

				// Headers
				if (trimmed_line.begins_with("#")) {
					int header_level = 0;
					while (header_level < trimmed_line.length() && trimmed_line[header_level] == '#') {
						header_level++;
					}
					String header_content = trimmed_line.substr(header_level).strip_edges();
					if (!header_content.is_empty()) {
						int font_size = 22 - (header_level * 2);
						processed_line = "[font_size=" + String::num_int64(font_size) + "][b]" + _process_inline_markdown(header_content) + "[/b][/font_size]";
					}
					// Lists
				} else if (trimmed_line.begins_with("- ") || trimmed_line.begins_with("* ")) {
					String item_content = trimmed_line.substr(trimmed_line.find(" ") + 1);
					processed_line = "[indent]* " + _process_inline_markdown(item_content) + "[/indent]";
				} else {
					// Regular paragraph
					processed_line = _process_inline_markdown(processed_line);
				}
				result += processed_line;
			}
		}

		if (i < lines.size() - 1) {
			result += "\n";
		}
	}

	return result;
}

void AIChatDock::clear_chat_history() {
	conversations.clear();
	current_conversation_index = -1;
	_queue_delayed_save();
	_update_conversation_dropdown();

	// Clear UI
	if (chat_container != nullptr) {
		for (int i = chat_container->get_child_count() - 1; i >= 0; i--) {
			Node *child = chat_container->get_child(i);
			if (child != nullptr) {
				child->queue_free();
			}
		}
	}
}

void AIChatDock::clear_current_conversation() {
	if (current_conversation_index >= 0) {
		Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
		chat_history.clear();
		
		// Update conversation timestamp
		conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
		_queue_delayed_save();

		// Clear UI
		if (chat_container != nullptr) {
			for (int i = chat_container->get_child_count() - 1; i >= 0; i--) {
				Node *child = chat_container->get_child(i);
				if (child != nullptr) {
					child->queue_free();
				}
			}
		}
	}
}

void AIChatDock::set_api_key(const String &p_api_key) {
	api_key = p_api_key;
	EditorSettings::get_singleton()->set_setting("ai_chat/api_key", p_api_key);
}

void AIChatDock::set_api_endpoint(const String &p_endpoint) {
	api_endpoint = p_endpoint;
	EditorTools::set_api_endpoint(p_endpoint);
}

void AIChatDock::set_model(const String &p_model) {
	model = p_model;
}

void AIChatDock::_save_layout_to_config(Ref<ConfigFile> p_layout, const String &p_section) const {
    // Force saving local endpoint during development
    // Persist the resolved endpoint for UX; will be recalculated next session as well
    p_layout->set_value(p_section, "api_endpoint", api_endpoint);
    p_layout->set_value(p_section, "model", model);
}

void AIChatDock::_load_layout_from_config(Ref<ConfigFile> p_layout, const String &p_section) {



    // Resolve endpoint using same IS_DEV logic as main initialization
    String base_url;
    String is_dev = OS::get_singleton()->get_environment("IS_DEV");
    if (is_dev.is_empty()) {
        // Backward-compat with DEV_MODE
        is_dev = OS::get_singleton()->get_environment("DEV_MODE");
    }
    if (!is_dev.is_empty() && is_dev.to_lower() == "true") {
        base_url = "http://127.0.0.1:8000";
    } else {
        base_url = "https://gamechat.simplifine.com";
    }
    
    // Allow override via editor settings or environment variable
    if (EditorSettings::get_singleton() && EditorSettings::get_singleton()->has_setting("ai_chat/base_url")) {
        String override_url = EditorSettings::get_singleton()->get_setting("ai_chat/base_url");
        if (!override_url.is_empty()) {
            base_url = override_url;
        }
    } else if (!OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL").is_empty()) {
        base_url = OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL");
    }
    
    if (base_url.ends_with("/")) {
        base_url = base_url.substr(0, base_url.length() - 1);
    }
    api_endpoint = base_url + "/chat";
	if (p_layout->has_section_key(p_section, "model")) {
		model = p_layout->get_value(p_section, "model");
	}
}
// Conversation management methods
void AIChatDock::_load_conversations() {
    // Recover from temp file if present and final missing
    String final_path = conversations_file_path;
    String base_dir = final_path.get_base_dir();
    String final_name = final_path.get_file();
    String temp_name = final_name + ".tmp";
    String temp_path = base_dir.path_join(temp_name);

    // Fallback: also check project settings dir (some distributions differ)
    String alt_final_path = final_path;
    if (EditorPaths::get_singleton() && EditorPaths::get_singleton()->are_paths_valid()) {
        alt_final_path = EditorPaths::get_singleton()->get_project_settings_dir().path_join("ai_chat_conversations.simplifine");
    }

    if (!FileAccess::exists(final_path) && FileAccess::exists(temp_path)) {
        Ref<DirAccess> da = DirAccess::open(base_dir);
        if (da.is_valid()) {
            da->rename(temp_name, final_name);
        }
    }

    if (!FileAccess::exists(final_path)) {
        if (FileAccess::exists(alt_final_path)) {
            final_path = alt_final_path; // load from alternative location
            base_dir = final_path.get_base_dir();
            final_name = final_path.get_file();
            temp_name = final_name + ".tmp";
            temp_path = base_dir.path_join(temp_name);
        } else {
            print_line("AI Chat: Conversations file not found at: " + conversations_file_path);
            return;
        }
    }

    Error err;
    String file_content = FileAccess::get_file_as_string(final_path, &err);
    if (err != OK) {
        print_line("AI Chat: Failed to load conversations file");
        return;
    }

    auto parse_json_string = [](const String &p_json, Dictionary &r_out) -> bool {
        Ref<JSON> json;
        json.instantiate();
        Error perr = json->parse(p_json);
        if (perr != OK) {
            return false;
        }
        r_out = json->get_data();
        return true;
    };

    Dictionary data;
    bool parsed = parse_json_string(file_content, data);
    if (!parsed && FileAccess::exists(temp_path)) {
        // Try temp as fallback
        String tmp_content = FileAccess::get_file_as_string(temp_path, &err);
        if (err == OK) {
            parsed = parse_json_string(tmp_content, data);
        }
    }
    if (!parsed) {
        print_line("AI Chat: Failed to parse conversations file (and temp fallback)");
        return;
    }

    if (!data.has("conversations")) {
        print_line("AI Chat: Conversations key missing in file: " + final_path);
        return;
    }

    Array conversations_array = data["conversations"];
    conversations.clear();

    for (int i = 0; i < conversations_array.size(); i++) {
        Dictionary conv_dict = conversations_array[i];
        AIChatDock::Conversation conv;
        conv.id = conv_dict.get("id", "");
        conv.title = conv_dict.get("title", "");
        conv.created_timestamp = conv_dict.get("created_timestamp", "");
        conv.last_modified_timestamp = conv_dict.get("last_modified_timestamp", "");

        Array messages_array = conv_dict.get("messages", Array());
        for (int j = 0; j < messages_array.size(); j++) {
            Dictionary msg_dict = messages_array[j];
            AIChatDock::ChatMessage msg;
            msg.role = msg_dict.get("role", "");
            msg.content = msg_dict.get("content", "");
            msg.timestamp = msg_dict.get("timestamp", "");
            msg.tool_calls = msg_dict.get("tool_calls", Array());
            msg.tool_call_id = msg_dict.get("tool_call_id", "");
            msg.name = msg_dict.get("name", "");

            // Load attached files
            Array attached_files_array = msg_dict.get("attached_files", Array());
            for (int k = 0; k < attached_files_array.size(); k++) {
                Dictionary file_dict = attached_files_array[k];
                AIChatDock::AttachedFile file;
                file.path = file_dict.get("path", "");
                file.name = file_dict.get("name", "");
                file.content = file_dict.get("content", "");
                file.is_image = file_dict.get("is_image", false);
                file.mime_type = file_dict.get("mime_type", "");
                file.base64_data = file_dict.get("base64_data", "");
                // Support v2 array sizes and v1 x/y sizes
                if (file_dict.has("original_size")) {
                    Array os = file_dict.get("original_size", Array());
                    if (os.size() >= 2) {
                        file.original_size.x = int(os[0]);
                        file.original_size.y = int(os[1]);
                    }
                } else {
                    file.original_size.x = file_dict.get("original_size_x", 0);
                    file.original_size.y = file_dict.get("original_size_y", 0);
                }
                if (file_dict.has("display_size")) {
                    Array ds = file_dict.get("display_size", Array());
                    if (ds.size() >= 2) {
                        file.display_size.x = int(ds[0]);
                        file.display_size.y = int(ds[1]);
                    }
                } else {
                    file.display_size.x = file_dict.get("display_size_x", 0);
                    file.display_size.y = file_dict.get("display_size_y", 0);
                }
                file.was_downsampled = file_dict.get("was_downsampled", false);
                file.is_node = file_dict.get("is_node", false);
                file.node_path = file_dict.get("node_path", NodePath());
                file.node_type = file_dict.get("node_type", "");
                msg.attached_files.push_back(file);
            }

            // Load tool results
            msg.tool_results = msg_dict.get("tool_results", Array());

            conv.messages.push_back(msg);
        }
        conversations.push_back(conv);
    }
    print_line("AI Chat: Loaded conversations: " + itos(conversations.size()) + " from: " + final_path);
}

void AIChatDock::_save_conversations() {
    Dictionary data;
    data["version"] = 2;
    Array conversations_array;

    for (int i = 0; i < conversations.size(); i++) {
        const AIChatDock::Conversation &conv = conversations[i];
        Dictionary conv_dict;
        conv_dict["id"] = conv.id;
        conv_dict["title"] = conv.title;
        conv_dict["created_timestamp"] = conv.created_timestamp;
        conv_dict["last_modified_timestamp"] = conv.last_modified_timestamp;

        Array messages_array;
        for (int j = 0; j < conv.messages.size(); j++) {
            const ChatMessage &msg = conv.messages[j];
            Dictionary msg_dict;
            msg_dict["role"] = msg.role;
            msg_dict["content"] = msg.content;
            msg_dict["timestamp"] = msg.timestamp;
            msg_dict["tool_calls"] = msg.tool_calls;
            msg_dict["tool_call_id"] = msg.tool_call_id;
            msg_dict["name"] = msg.name;
            msg_dict["tool_results"] = msg.tool_results;

            // Save attached files
            Array attached_files_array;
            for (int k = 0; k < msg.attached_files.size(); k++) {
                const AttachedFile &file = msg.attached_files[k];
                Dictionary file_dict;
                file_dict["path"] = file.path;
                file_dict["name"] = file.name;
                file_dict["content"] = file.content;
                file_dict["is_image"] = file.is_image;
                file_dict["mime_type"] = file.mime_type;
                file_dict["base64_data"] = file.base64_data;
                // v2 array sizes + v1 compatibility keys
                Array os;
                os.push_back(file.original_size.x);
                os.push_back(file.original_size.y);
                file_dict["original_size"] = os;
                Array ds;
                ds.push_back(file.display_size.x);
                ds.push_back(file.display_size.y);
                file_dict["display_size"] = ds;
                file_dict["original_size_x"] = file.original_size.x;
                file_dict["original_size_y"] = file.original_size.y;
                file_dict["display_size_x"] = file.display_size.x;
                file_dict["display_size_y"] = file.display_size.y;
                file_dict["was_downsampled"] = file.was_downsampled;
                file_dict["is_node"] = file.is_node;
                file_dict["node_path"] = file.node_path;
                file_dict["node_type"] = file.node_type;
                attached_files_array.push_back(file_dict);
            }
            msg_dict["attached_files"] = attached_files_array;

            messages_array.push_back(msg_dict);
        }
        conv_dict["messages"] = messages_array;
        conversations_array.push_back(conv_dict);
    }

    data["conversations"] = conversations_array;

    Ref<JSON> json;
    json.instantiate();
    String json_string = json->stringify(data, "  ");

    // Safe write via temp + rename
    Error err;
    String final_path = conversations_file_path;
    String base_dir = final_path.get_base_dir();
    String final_name = final_path.get_file();
    String temp_name = final_name + ".tmp";
    String temp_path = base_dir.path_join(temp_name);

    // Ensure destination directory exists
    {
        Ref<DirAccess> da_mk = DirAccess::create_for_path(base_dir);
        if (da_mk.is_valid() && !da_mk->dir_exists(base_dir)) {
            da_mk->make_dir_recursive(base_dir);
        }
    }

    {
        Ref<FileAccess> tmp = FileAccess::open(temp_path, FileAccess::WRITE, &err);
        if (err != OK) {
            print_line("AI Chat: Failed to open temp file for save");
            return;
        }
        tmp->store_string(json_string);
        tmp->close();
    }

    Ref<DirAccess> da = DirAccess::open(base_dir);
    if (da.is_valid()) {
        if (da->file_exists(final_name)) {
            da->remove(final_name);
        }
        da->rename(temp_name, final_name);
    } else {
        Ref<FileAccess> file = FileAccess::open(final_path, FileAccess::WRITE, &err);
        if (err != OK) {
            print_line("AI Chat: Failed to save conversations file");
            return;
        }
        file->store_string(json_string);
        file->close();
    }
}
void AIChatDock::_create_new_conversation() {
	AIChatDock::Conversation new_conv;
	new_conv.id = _generate_conversation_id();
	new_conv.title = "New Conversation";
	new_conv.created_timestamp = _get_timestamp();
	new_conv.last_modified_timestamp = _get_timestamp();
	
	conversations.push_back(new_conv);
	current_conversation_index = conversations.size() - 1;
	// Don't save immediately - defer for better UI responsiveness
	
	// Clear UI for new conversation
	if (chat_container != nullptr) {
		for (int i = chat_container->get_child_count() - 1; i >= 0; i--) {
			Node *child = chat_container->get_child(i);
			if (child != nullptr) {
				child->queue_free();
			}
		}
	}
}

void AIChatDock::_create_new_conversation_instant() {
	// Instant version - no save operation to avoid blocking UI
	AIChatDock::Conversation new_conv;
	new_conv.id = _generate_conversation_id();
	new_conv.title = "New Conversation";
	new_conv.created_timestamp = _get_timestamp();
	new_conv.last_modified_timestamp = _get_timestamp();
	
	conversations.push_back(new_conv);
	current_conversation_index = conversations.size() - 1;
	// Note: Save is deferred in caller
	
	// Clear UI for new conversation
	if (chat_container != nullptr) {
		for (int i = chat_container->get_child_count() - 1; i >= 0; i--) {
			Node *child = chat_container->get_child(i);
			if (child != nullptr) {
				child->queue_free();
			}
		}
	}
}

void AIChatDock::_switch_to_conversation(int p_index) {
	if (p_index < 0 || p_index >= conversations.size()) {
		return;
	}
	
	current_conversation_index = p_index;
	
	// Clear current UI
	if (chat_container != nullptr) {
		for (int i = chat_container->get_child_count() - 1; i >= 0; i--) {
			Node *child = chat_container->get_child(i);
			if (child != nullptr) {
				child->queue_free();
			}
		}
	}
	
	// Rebuild UI from conversation messages with proper tool call handling
	const Vector<AIChatDock::ChatMessage> &messages = conversations[p_index].messages;
	_rebuild_conversation_ui(messages);
	
	call_deferred("_scroll_to_bottom");
}

void AIChatDock::_update_conversation_dropdown() {
	if (!conversation_history_dropdown) {
		return;
	}
	
	conversation_history_dropdown->clear();
	
	for (int i = 0; i < conversations.size(); i++) {
		const AIChatDock::Conversation &conv = conversations[i];
		String display_title = conv.title;
		if (display_title.length() > 30) {
			display_title = display_title.substr(0, 27) + "...";
		}
		display_title += " (" + conv.last_modified_timestamp + ")";
		conversation_history_dropdown->add_item(display_title);
	}
	
	if (current_conversation_index >= 0) {
		conversation_history_dropdown->select(current_conversation_index);
	}
}

String AIChatDock::_generate_conversation_id() {
	return "conv_" + String::num_uint64(Time::get_singleton()->get_unix_time_from_system()) + "_" + String::num(Math::rand() % 10000);
}

String AIChatDock::_generate_conversation_title(const Vector<AIChatDock::ChatMessage> &p_messages) {
	// Generate title from first user message
	for (int i = 0; i < p_messages.size(); i++) {
		if (p_messages[i].role == "user" && !p_messages[i].content.is_empty()) {
			String content = p_messages[i].content.strip_edges();
			if (content.length() > 50) {
				content = content.substr(0, 47) + "...";
			}
			return content;
		}
	}
	return "New Conversation";
}

Vector<AIChatDock::ChatMessage> &AIChatDock::_get_current_chat_history() {
	static Vector<AIChatDock::ChatMessage> empty_history;
	if (current_conversation_index >= 0 && current_conversation_index < conversations.size()) {
		return conversations.write[current_conversation_index].messages;
	}
	return empty_history;
}

void AIChatDock::_on_conversation_selected(int p_index) {
	if (p_index != current_conversation_index) {
		_switch_to_conversation(p_index);
	}
}

void AIChatDock::_on_new_conversation_pressed() {
	// PERFORMANCE OPTIMIZATION: Instant conversation creation with deferred save
	_create_new_conversation_instant();
	_update_conversation_dropdown();
	
	// Use delayed save to avoid UI blocking
	_queue_delayed_save();
	_execute_delayed_save(); // start background save immediately
}
void AIChatDock::_build_hierarchy_tree_item(Tree *p_tree, TreeItem *p_parent, const Dictionary &p_node_data) {
	if (p_node_data.is_empty()) {
		return;
	}
	
	TreeItem *item = p_tree->create_item(p_parent);
	String name = p_node_data.get("name", "Unknown");
	String type = p_node_data.get("type", "");
	
	if (!type.is_empty()) {
		item->set_text(0, name + " (" + type + ")");
	} else {
		item->set_text(0, name);
	}
	
	item->set_icon(0, get_theme_icon(SNAME("Node"), SNAME("EditorIcons")));
	
	// Recursively add children if they exist
	Array children = p_node_data.get("children", Array());
	for (int i = 0; i < children.size(); i++) {
		Dictionary child = children[i];
		_build_hierarchy_tree_item(p_tree, item, child);
	}
}

void AIChatDock::_ensure_related_graph_ui() {
	if (!related_graph_panel) {
		related_graph_panel = memnew(PopupPanel);
		add_child(related_graph_panel);
		VBoxContainer *vb = memnew(VBoxContainer);
		related_graph_panel->add_child(vb);
		Label *title = memnew(Label);
		title->set_text("Related graph");
		vb->add_child(title);
		related_graph_tree = memnew(Tree);
		related_graph_tree->set_v_size_flags(SIZE_EXPAND_FILL);
		related_graph_tree->set_h_size_flags(SIZE_EXPAND_FILL);
		related_graph_tree->set_custom_minimum_size(Size2(520, 320));
		vb->add_child(related_graph_tree);
	}
}

void AIChatDock::_populate_related_graph(const Dictionary &p_graph) {
	if (!related_graph_tree) return;
	related_graph_tree->clear();
	TreeItem *root = related_graph_tree->create_item();
	Array file_keys = p_graph.keys();
	for (int i = 0; i < file_keys.size(); i++) {
		String file_path = (String)file_keys[i];
		Dictionary ctx = p_graph[file_path];
		TreeItem *file_item = related_graph_tree->create_item(root);
		file_item->set_text(0, file_path);
		Array nodes = ctx.has("nodes") ? (Array)ctx["nodes"] : Array();
		Array edges = ctx.has("edges") ? (Array)ctx["edges"] : Array();
		TreeItem *nodes_item = related_graph_tree->create_item(file_item);
		nodes_item->set_text(0, vformat("Nodes (%d)", nodes.size()));
		for (int n = 0; n < nodes.size(); n++) {
			Dictionary nd = nodes[n];
			String ntext = String(nd.get("name", ""));
			String nkind = String(nd.get("kind", ""));
			String npath = String(nd.get("node_path", ""));
			int64_t sline = (int64_t)nd.get("start_line", 0);
			TreeItem *ni = related_graph_tree->create_item(nodes_item);
			String display = ntext;
			if (!nkind.is_empty()) display += " [" + nkind + "]";
			if (!npath.is_empty()) display += " — " + npath;
			if (sline > 0) display += vformat(" :%d", (int)sline);
			ni->set_text(0, display);
		}
		TreeItem *edges_item = related_graph_tree->create_item(file_item);
		edges_item->set_text(0, vformat("Edges (%d)", edges.size()));
		for (int e = 0; e < edges.size(); e++) {
			Dictionary ed = edges[e];
			String ekind = String(ed.get("kind", ""));
			int64_t sline = (int64_t)ed.get("start_line", 0);
			TreeItem *ei = related_graph_tree->create_item(edges_item);
			String display = ekind;
			if (sline > 0) display += vformat(" :%d", (int)sline);
			ei->set_text(0, display);
		}
	}
}

void AIChatDock::_show_related_graph(const Dictionary &p_graph) {
	_ensure_related_graph_ui();
	_populate_related_graph(p_graph);
	if (related_graph_panel) {
		related_graph_panel->popup_centered();
	}
}
AIChatDock::AIChatDock() {
	set_name("AI Chat");
	
	// Enable drag and drop for the dock
	set_drag_forwarding(Callable(), callable_mp(this, &AIChatDock::can_drop_data_fw), callable_mp(this, &AIChatDock::drop_data_fw));

	// HTTP client for streaming API calls
	http_client = HTTPClient::create();
	
	// Timer for deferred saving
	save_timer = memnew(Timer);
	save_timer->set_one_shot(true);
	add_child(save_timer);
	save_timer->connect("timeout", callable_mp(this, &AIChatDock::_execute_delayed_save));
	
	// Initialize mutex for background saving
	save_mutex = memnew(Mutex);
	
	// HTTP request for stop requests (non-streaming)
	stop_http_request = memnew(HTTPRequest);
	add_child(stop_http_request);
	stop_http_request->connect("request_completed", callable_mp(this, &AIChatDock::_on_stop_request_completed));

	// HTTP request for models endpoint
	models_http_request = memnew(HTTPRequest);
	add_child(models_http_request);
	models_http_request->connect("request_completed", callable_mp(this, &AIChatDock::_on_models_request_completed));

	diff_viewer = memnew(DiffViewer);
	add_child(diff_viewer);
	diff_viewer->connect("diff_accepted", callable_mp(this, &AIChatDock::_on_diff_accepted));

	// Start the AI tool server
	tool_server.instantiate();
	Error err = tool_server->listen(8001);
	if (err == OK) {
		print_line("AI Chat Dock: Tool server started on port 8001");
	} else {
		print_line("AI Chat Dock: Failed to start tool server on port 8001");
	}
	
	// Initialize embedding system
	call_deferred("_initialize_embedding_system");
	
	// Connect to script editor save signals for dual acceptance
	// We'll do this in a deferred call to ensure ScriptEditor is ready
	call_deferred("_connect_script_editor_signals");
}


void AIChatDock::_on_diff_accepted(const String &p_path, const String &p_content) {
    // For script-like files, we need to properly update and save the script
    String ext = p_path.get_extension().to_lower();
    if (ext == "gd" || ext == "cs" || ext == "shader" || ext == "glsl" ) {
        print_line("Diff accepted for script-like file: " + p_path);
        
        // CRITICAL: Remove the script from cache first to force a fresh reload
        // This is necessary because Godot caches compiled scripts and errors
        GDScriptCache::remove_script(p_path);
        
        // Load the script resource and update its source code
        Ref<Resource> resource = ResourceLoader::load(p_path);
        Ref<Script> script = resource;
        if (script.is_valid()) {
            script->set_source_code(p_content);
            
            // Save the script to disk immediately
            // This ensures the game runtime sees the updated content
            EditorNode::get_singleton()->save_resource(resource);
            
            // Force a complete reload of the script
            script->reload(true);
            
            // Trigger script reload to update any cached state
            ScriptEditor *script_editor = ScriptEditor::get_singleton();
            if (script_editor) {
                script_editor->trigger_live_script_reload(p_path);
                // Also reload the script in the editor
                script_editor->reload_scripts(false);
            }
        }
        
        // Clear in-memory overlay after saving
        EditorTools::clear_preview_overlay(p_path);
        return;
    }

    // For non-script resources (e.g., .tscn, .tres, .cfg), write immediately to avoid
    // Godot trying to open them as scripts (which would yield .tscn.gd tabs and save prompts).
    _apply_file_edit_immediate(p_path, p_content);
}

void AIChatDock::_apply_file_edit_immediate(const String &p_path, const String &p_content) {
    if (p_path.is_empty()) {
        return;
    }
    String abs_dir = ProjectSettings::get_singleton()->globalize_path(p_path.get_base_dir());
    DirAccess::make_dir_recursive_absolute(abs_dir);
    Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
    if (f.is_null()) {
        print_line("AI Chat: Failed to write edited content to: " + p_path);
        return;
    }
    f->store_string(p_content);
    f->close();

    // If we edited a recognized resource, reload it so the editor reflects the change
    Ref<Resource> res = ResourceLoader::load(p_path);
    if (res.is_valid()) {
        // For resources like .tres/.tscn, reload from disk and save to ensure proper import metadata
        ResourceSaver::save(res, p_path);
    } else {
        // If it wasn't loadable yet (new file), try to load it now to register in the FS
        res = ResourceLoader::load(p_path);
        if (res.is_valid()) {
            ResourceSaver::save(res, p_path);
        }
    }
    print_line("AI Chat: Wrote edited content to: " + p_path);

    // Notify the editor about the change so the file system picks it up
    if (EditorFileSystem::get_singleton()) {
        EditorFileSystem::get_singleton()->update_file(p_path);
        EditorFileSystem::get_singleton()->scan_changes();
    }

    // Clear overlay once written to disk to avoid masking real file content
    EditorTools::clear_preview_overlay(p_path);
}
void AIChatDock::send_error_message(const String &p_error_text) {
	String formatted_message = "Please help fix this error:\n\n" + p_error_text;
	
	// Set the message in the input field
	input_field->set_text(formatted_message);
	
	// Show the AI Chat dock if it's not visible
	// This will make the dock visible in the editor
	set_visible(true);
	
	// Focus on the AI Chat dock to bring user's attention
	input_field->grab_focus();
	
	// Automatically send the message
	call_deferred("_on_send_button_pressed");
}

// Image processing methods
bool AIChatDock::_is_image_file(const String &p_path) {
	String ext = p_path.get_extension().to_lower();
	return ext == "png" || ext == "jpg" || ext == "jpeg" || ext == "gif" || 
		   ext == "bmp" || ext == "webp" || ext == "svg";
}

String AIChatDock::_get_mime_type_from_extension(const String &p_path) {
	String ext = p_path.get_extension().to_lower();
	if (ext == "png") return "image/png";
	if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
	if (ext == "gif") return "image/gif";
	if (ext == "bmp") return "image/bmp";
	if (ext == "webp") return "image/webp";
	if (ext == "svg") return "image/svg+xml";
	return "text/plain";
}
bool AIChatDock::_process_image_attachment(AttachedFile &p_file) {
	Ref<Image> image = Image::load_from_file(p_file.path);
	if (image.is_null() || image->is_empty()) {
		return false;
	}

	Vector2i original_size = Vector2i(image->get_width(), image->get_height());
	p_file.original_size = original_size;
	
	// Check if image needs to be downsampled (max 1024px on any side)
	const int MAX_DIMENSION = 1024;
	Vector2i target_size = _calculate_downsampled_size(original_size, MAX_DIMENSION);
	
	if (target_size != original_size) {
		p_file.was_downsampled = true;
		image->resize(target_size.x, target_size.y, Image::INTERPOLATE_LANCZOS);
		
		// Show warning dialog
		call_deferred("_show_image_warning_dialog", p_file.name, original_size, target_size);
	}
	
	p_file.display_size = target_size;
	
	// Convert to base64 for API transmission
	Vector<uint8_t> png_buffer;
	if (p_file.mime_type == "image/jpeg" || p_file.mime_type == "image/jpg") {
		png_buffer = image->save_jpg_to_buffer(0.85f); // Good quality JPEG
	} else {
		png_buffer = image->save_png_to_buffer();
	}
	
	if (png_buffer.size() == 0) {
		return false;
	}
	
	// Encode to base64
	p_file.base64_data = CoreBind::Marshalls::get_singleton()->raw_to_base64(png_buffer);
	
	return true;
}

Vector2i AIChatDock::_calculate_downsampled_size(const Vector2i &p_original, int p_max_dimension) {
	if (p_original.x <= p_max_dimension && p_original.y <= p_max_dimension) {
		return p_original;
	}
	
	float aspect_ratio = (float)p_original.x / (float)p_original.y;
	Vector2i new_size;
	
	if (p_original.x > p_original.y) {
		// Landscape
		new_size.x = p_max_dimension;
		new_size.y = (int)(p_max_dimension / aspect_ratio);
	} else {
		// Portrait or square
		new_size.y = p_max_dimension;
		new_size.x = (int)(p_max_dimension * aspect_ratio);
	}
	
	return new_size;
}

void AIChatDock::_show_image_warning_dialog(const String &p_filename, const Vector2i &p_original, const Vector2i &p_new_size) {
	if (!image_warning_dialog) {
		return;
	}
	
	String message = String("Image '{0}' was downsampled from {1}×{2} to {3}×{4} to reduce file size for transmission.")
		.format(varray(p_filename, p_original.x, p_original.y, p_new_size.x, p_new_size.y));
	
	image_warning_dialog->set_text(message);
	image_warning_dialog->popup_centered(Size2i(500, 150));
}

void AIChatDock::_handle_generated_image(const String &p_base64_data, const String &p_id) {
	print_line("AI Chat: _handle_generated_image called with ID: " + p_id + ", data length: " + String::num_int64(p_base64_data.length()));
	
	if (p_base64_data.is_empty()) {
		print_line("AI Chat: _handle_generated_image - base64 data is empty, aborting");
		return;
	}
	
	// Defer image display to next frame to avoid UI race conditions during streaming
	print_line("AI Chat: _handle_generated_image - calling deferred _display_generated_image_deferred");
	call_deferred("_display_generated_image_deferred", p_base64_data, p_id);
}
void AIChatDock::_display_generated_image_deferred(const String &p_base64_data, const String &p_id) {
	// Decode base64 to image
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	if (image_data.size() == 0) {
		print_line("AI Chat: Failed to decode generated image data");
		return;
	}
	
	// Create image from data
	Ref<Image> generated_image = memnew(Image);
	Error err = generated_image->load_png_from_buffer(image_data);
	if (err != OK) {
		// Try JPEG if PNG fails
		err = generated_image->load_jpg_from_buffer(image_data);
		if (err != OK) {
			print_line("AI Chat: Failed to load generated image");
			return;
		}
	}
	
	if (generated_image->is_empty()) {
		print_line("AI Chat: Generated image is empty");
		return;
	}
	
	// Safely find the last assistant message bubble without creating new ones
	PanelContainer *bubble_panel = nullptr;
	if (chat_container) {
		print_line("AI Chat: Searching for assistant message bubble, total children: " + String::num_int64(chat_container->get_child_count()));
		// Look for the last panel container (which should be our assistant message)
		for (int i = chat_container->get_child_count() - 1; i >= 0; i--) {
			Node *child = chat_container->get_child(i);
			print_line("AI Chat: Child " + String::num_int64(i) + " type: " + child->get_class());
			PanelContainer *panel = Object::cast_to<PanelContainer>(child);
			if (panel) {
				print_line("AI Chat: Found PanelContainer at index " + String::num_int64(i));
				bubble_panel = panel;
				break;
			}
		}
	}
	
	if (!bubble_panel) {
		print_line("AI Chat: Could not find assistant message bubble for generated image");
		return;
	} else {
		print_line("AI Chat: Successfully found bubble panel for image display");
	}

	
	// Find the VBoxContainer inside the message bubble
	VBoxContainer *message_vbox = nullptr;
	print_line("AI Chat: Searching for VBoxContainer in bubble panel, children count: " + String::num_int64(bubble_panel->get_child_count()));
	for (int i = 0; i < bubble_panel->get_child_count(); i++) {
		Node *child = bubble_panel->get_child(i);
		print_line("AI Chat: Bubble child " + String::num_int64(i) + " type: " + child->get_class());
		message_vbox = Object::cast_to<VBoxContainer>(child);
		if (message_vbox) {
			print_line("AI Chat: Found VBoxContainer at index " + String::num_int64(i));
			break;
		}
	}
	
	if (!message_vbox) {
		print_line("AI Chat: Could not find VBoxContainer in message bubble - aborting image display");
		return;
	}
	
    // Find and clear the tool placeholder containing "Running tool..." text
	print_line("AI Chat: Searching for tool placeholder in message vbox, children count: " + String::num_int64(message_vbox->get_child_count()));
	bool found_placeholder = false;
	for (int i = 0; i < message_vbox->get_child_count(); i++) {
		Node *child = message_vbox->get_child(i);
		print_line("AI Chat: VBox child " + String::num_int64(i) + " type: " + child->get_class() + " name: " + child->get_name());
		
		// Look for tool placeholder panels
		PanelContainer *panel = Object::cast_to<PanelContainer>(child);
		if (panel && String(panel->get_name()).begins_with("tool_placeholder_")) {
			print_line("AI Chat: Found tool placeholder panel: " + panel->get_name());
			// Clear the tool placeholder content and replace with success message
			while (panel->get_child_count() > 0) {
				Node *panel_child = panel->get_child(0);
				panel->remove_child(panel_child);
				panel_child->queue_free();
			}
			
			// Add success message
			Label *success_label = memnew(Label);
			success_label->set_text("Generated image");
			success_label->add_theme_color_override("font_color", get_theme_color(SNAME("success_color"), SNAME("Editor")));
			success_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
			panel->add_child(success_label);
			
			found_placeholder = true;
			break;
		}
		
		// Also check for RichTextLabel (fallback for other cases)
		RichTextLabel *label = Object::cast_to<RichTextLabel>(child);
		if (label && label->get_text().contains("Calling tool")) {
			print_line("AI Chat: Found RichTextLabel with tool text, updating");
			label->clear();
			label->append_text("Generated image\n\n");
			found_placeholder = true;
			break;
		}
	}
	
	if (!found_placeholder) {
		print_line("AI Chat: No tool placeholder found to clear - this might be okay for some flows");
	}
	
	// Create image display container
	print_line("AI Chat: Creating image display container");
	PanelContainer *image_panel = memnew(PanelContainer);
	message_vbox->add_child(image_panel);
	print_line("AI Chat: Added image panel to message vbox");
	
	// Style the image panel
	Ref<StyleBoxFlat> image_style = memnew(StyleBoxFlat);
	image_style->set_bg_color(get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
	image_style->set_border_width_all(2);
	image_style->set_border_color(get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	image_style->set_corner_radius_all(8);
	image_style->set_content_margin_all(8);
	image_panel->add_theme_style_override("panel", image_style);
	
	VBoxContainer *image_container = memnew(VBoxContainer);
	image_panel->add_child(image_container);
	
	// Add "Generated Image" label
    Label *header_label = memnew(Label);
    header_label->set_text("Generated Image");
	header_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
	header_label->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	image_container->add_child(header_label);
	
	// Resize image for display (max 512px to keep it reasonable)
	Vector2i original_size = Vector2i(generated_image->get_width(), generated_image->get_height());
	Vector2i display_size = _calculate_downsampled_size(original_size, 512);
	
	if (display_size != original_size) {
		generated_image->resize(display_size.x, display_size.y, Image::INTERPOLATE_LANCZOS);
	}
	
	// Create texture and display
	Ref<ImageTexture> generated_texture = ImageTexture::create_from_image(generated_image);
	
	TextureRect *image_display = memnew(TextureRect);
	image_display->set_texture(generated_texture);
	image_display->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_display->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_display->set_custom_minimum_size(Size2(display_size.x, display_size.y));
	image_container->add_child(image_display);
	
    // Add image info
    HBoxContainer *info_container = memnew(HBoxContainer);
	image_container->add_child(info_container);
	
	Label *size_label = memnew(Label);
	size_label->set_text(String::num_int64(original_size.x) + "x" + String::num_int64(original_size.y));
	size_label->add_theme_font_size_override("font_size", 10);
	size_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
	info_container->add_child(size_label);
	
	// Add spacer to push save button to the right
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	info_container->add_child(spacer);
	
	// Save button for generated images
	Button *save_button = memnew(Button);
	save_button->set_text("Save");
	save_button->set_flat(true);
	save_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Save"), SNAME("EditorIcons")));
	save_button->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->add_theme_color_override("icon_normal_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->set_tooltip_text("Save this image to your project");
	save_button->connect("pressed", callable_mp(this, &AIChatDock::_on_save_image_pressed).bind(p_base64_data, "png"));
	info_container->add_child(save_button);
	
	// Store the generated image in the current message
	Vector<AIChatDock::ChatMessage> &chat_history = _get_current_chat_history();
	print_line("AI Chat: Attempting to save image to chat history, total messages: " + String::num_int64(chat_history.size()));
	
	if (!chat_history.is_empty()) {
		ChatMessage &last_msg = chat_history.write[chat_history.size() - 1];
		print_line("AI Chat: Last message role: " + last_msg.role + ", content: '" + last_msg.content.substr(0, 50) + "...'");
		print_line("AI Chat: Last message current attached files count: " + String::num_int64(last_msg.attached_files.size()));
		
		if (last_msg.role == "assistant") {
					// Add the generated image as an attachment for persistence
		AIChatDock::AttachedFile generated_file;
		generated_file.path = "generated://" + p_id;
		// Create proper unique ID for generated images
		generated_file.name = "gen_img_" + String::num_int64(OS::get_singleton()->get_ticks_msec());
		generated_file.is_image = true;
		generated_file.mime_type = "image/png";
			generated_file.base64_data = p_base64_data;
			generated_file.original_size = original_size;
			generated_file.display_size = display_size;
			generated_file.was_downsampled = (display_size != original_size);
					last_msg.attached_files.push_back(generated_file);
		print_line("AI Chat: Successfully added generated image ID: " + generated_file.name + " to assistant message");
		} else {
			print_line("AI Chat: Cannot save image - last message is not from assistant (role: " + last_msg.role + ")");
		}
	} else {
		print_line("AI Chat: Cannot save image - chat history is empty");
	}
	
	// Update conversation and scroll
	if (current_conversation_index >= 0) {
		conversations.write[current_conversation_index].last_modified_timestamp = _get_timestamp();
		_queue_delayed_save();
	}
	
	print_line("AI Chat: Image display complete, forcing UI refresh");
	
	// Force UI update to show the newly added image immediately
	if (bubble_panel) {
		bubble_panel->queue_redraw();
		print_line("AI Chat: Queued bubble_panel redraw");
	}
	if (chat_container) {
		chat_container->queue_redraw();
		print_line("AI Chat: Queued chat_container redraw");
	}
	// Also update the main dock
	queue_redraw();
	print_line("AI Chat: Queued main dock redraw");
	
	call_deferred("_scroll_to_bottom");
	print_line("AI Chat: _display_generated_image_deferred completed successfully");
}
void AIChatDock::_display_generated_image_in_tool_result(VBoxContainer *p_container, const String &p_base64_data, const Dictionary &p_data) {
	if (!p_container || p_base64_data.is_empty()) {
		return;
	}
	
	// Decode base64 to image
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	if (image_data.size() == 0) {
		print_line("AI Chat: Failed to decode generated image data from tool result");
		return;
	}
	
	// Create image from data
	Ref<Image> generated_image = memnew(Image);
	Error err = generated_image->load_png_from_buffer(image_data);
	if (err != OK) {
		// Try JPEG if PNG fails
		err = generated_image->load_jpg_from_buffer(image_data);
		if (err != OK) {
			print_line("AI Chat: Failed to load generated image from tool result");
			return;
		}
	}
	
	if (generated_image->is_empty()) {
		print_line("AI Chat: Generated image from tool result is empty");
		return;
	}
	
	// Create image display container
	VBoxContainer *image_container = memnew(VBoxContainer);
	p_container->add_child(image_container);
	
	// Add image info
	HBoxContainer *info_container = memnew(HBoxContainer);
	image_container->add_child(info_container);
	
	HBoxContainer *prompt_container = memnew(HBoxContainer);
	info_container->add_child(prompt_container);
	
	Label *prompt_icon = memnew(Label);
	prompt_icon->add_theme_icon_override("icon", get_theme_icon(SNAME("Image"), SNAME("EditorIcons")));
	prompt_container->add_child(prompt_icon);
	
    Label *prompt_label = memnew(Label);
	String prompt = p_data.get("prompt", "Generated Image");
	prompt_label->set_text(prompt);
	prompt_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
	prompt_label->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
    prompt_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    prompt_label->set_clip_text(false);
    prompt_label->set_custom_minimum_size(Size2(0, 0));
	prompt_container->add_child(prompt_label);
	
	// Resize image for display (max 200px in tool results to keep them compact)
	Vector2i original_size = Vector2i(generated_image->get_width(), generated_image->get_height());
	Vector2i display_size = _calculate_downsampled_size(original_size, 200);
	
	if (display_size != original_size) {
		generated_image->resize(display_size.x, display_size.y, Image::INTERPOLATE_LANCZOS);
	}
	
	// Create texture and display
	Ref<ImageTexture> generated_texture = ImageTexture::create_from_image(generated_image);
	
	TextureRect *image_display = memnew(TextureRect);
	image_display->set_texture(generated_texture);
	image_display->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_display->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_display->set_custom_minimum_size(Size2(display_size.x, display_size.y));
	image_container->add_child(image_display);
	
	// Add technical details and save button
	HBoxContainer *tech_container = memnew(HBoxContainer);
	image_container->add_child(tech_container);
	
	Label *size_label = memnew(Label);
	size_label->set_text(String::num_int64(original_size.x) + "x" + String::num_int64(original_size.y));
	size_label->add_theme_font_size_override("font_size", 10);
	size_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
	tech_container->add_child(size_label);
	
	Label *model_label = memnew(Label);
	String model = p_data.get("model", "DALL-E");
	model_label->set_text(" • " + model);
	model_label->add_theme_font_size_override("font_size", 10);
	model_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
	tech_container->add_child(model_label);
	
	// Add spacer to push save button to the right
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tech_container->add_child(spacer);
	
	// Save button
	Button *save_button = memnew(Button);
	save_button->set_text("Save to...");
	save_button->set_flat(true);
	save_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Save"), SNAME("EditorIcons")));
	save_button->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->add_theme_color_override("icon_normal_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->set_tooltip_text("Save this image to your project");
	save_button->connect("pressed", callable_mp(this, &AIChatDock::_on_save_image_pressed).bind(p_base64_data, "png"));
	tech_container->add_child(save_button);
}
void AIChatDock::_display_image_unified(VBoxContainer *p_container, const String &p_base64_data, const Dictionary &p_metadata) {
	if (!p_container || p_base64_data.is_empty()) {
		return;
	}
	
	// Decode base64 to image
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	if (image_data.size() == 0) {
		print_line("AI Chat: Failed to decode image data");
		return;
	}
	
	// Create image from data
	Ref<Image> display_image = memnew(Image);
	Error err = display_image->load_png_from_buffer(image_data);
	if (err != OK) {
		// Try JPEG if PNG fails
		err = display_image->load_jpg_from_buffer(image_data);
		if (err != OK) {
			print_line("AI Chat: Failed to load image");
			return;
		}
	}
	
	if (display_image->is_empty()) {
		print_line("AI Chat: Image is empty");
		return;
	}
	
	// Create unified image display container
	VBoxContainer *image_container = memnew(VBoxContainer);
	p_container->add_child(image_container);
	
	// Extract metadata with defaults
	String title = p_metadata.get("prompt", p_metadata.get("name", "Image"));
	String model = p_metadata.get("model", "");
	String file_path = p_metadata.get("path", "");
	bool is_generated = file_path.begins_with("generated://");
	int max_display_size = is_generated ? 200 : 150; // Generated images slightly larger
	
    // Add image title/info with proper wrapping so long text doesn't expand panel width
    HBoxContainer *info_container = memnew(HBoxContainer);
    info_container->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    image_container->add_child(info_container);

    // Left: icon
    Label *icon = memnew(Label);
    icon->add_theme_icon_override("icon", get_theme_icon(SNAME("Image"), SNAME("EditorIcons")));
    info_container->add_child(icon);

    // Right: a wrapping label in a VBox so it can take full width and wrap
    VBoxContainer *title_vbox = memnew(VBoxContainer);
    title_vbox->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    info_container->add_child(title_vbox);

    Label *title_label = memnew(Label);
    title_label->set_text(title);
    title_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    title_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
    title_label->add_theme_font_override("font", get_theme_font(SNAME("bold"), SNAME("EditorFonts")));
    title_label->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
    title_vbox->add_child(title_label);
	
	// Resize image for display
	Vector2i original_size = Vector2i(display_image->get_width(), display_image->get_height());
	Vector2i display_size = _calculate_downsampled_size(original_size, max_display_size);
	
	if (display_size != original_size) {
		display_image->resize(display_size.x, display_size.y, Image::INTERPOLATE_LANCZOS);
	}
	
	// Create texture and display
	Ref<ImageTexture> image_texture = ImageTexture::create_from_image(display_image);
	
	TextureRect *image_display = memnew(TextureRect);
	image_display->set_texture(image_texture);
	image_display->set_expand_mode(TextureRect::EXPAND_FIT_WIDTH_PROPORTIONAL);
	image_display->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
	image_display->set_custom_minimum_size(Size2(display_size.x, display_size.y));
	image_container->add_child(image_display);
	
	// Add technical details and controls
	HBoxContainer *details_container = memnew(HBoxContainer);
	image_container->add_child(details_container);
	
	// Size info
	Label *size_label = memnew(Label);
	size_label->set_text(String::num_int64(original_size.x) + "x" + String::num_int64(original_size.y));
	size_label->add_theme_font_size_override("font_size", 10);
	size_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
	details_container->add_child(size_label);
	
	// Model info (for generated images)
	if (!model.is_empty()) {
		Label *model_label = memnew(Label);
		model_label->set_text(" • " + model);
		model_label->add_theme_font_size_override("font_size", 10);
		model_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(1, 1, 1, 0.7));
		details_container->add_child(model_label);
	}
	
	// File path button (for attached images)
	if (!file_path.is_empty() && !is_generated) {
		Button *file_link = memnew(Button);
		file_link->set_text(" • " + file_path.get_file());
		file_link->set_flat(true);
		file_link->add_theme_font_size_override("font_size", 10);
		file_link->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
		file_link->set_tooltip_text("Click to open: " + file_path);
		file_link->connect("pressed", callable_mp(this, &AIChatDock::_on_tool_file_link_pressed).bind(file_path));
		details_container->add_child(file_link);
	}
	
	// Add spacer to push save button to the right
	Control *spacer = memnew(Control);
	spacer->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	details_container->add_child(spacer);
	
	// Save button
	Button *save_button = memnew(Button);
	save_button->set_text("Save to...");
	save_button->set_flat(true);
	save_button->add_theme_icon_override("icon", get_theme_icon(SNAME("Save"), SNAME("EditorIcons")));
	save_button->add_theme_color_override("font_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->add_theme_color_override("icon_normal_color", get_theme_color(SNAME("accent_color"), SNAME("Editor")));
	save_button->set_tooltip_text("Save this image to your project");
	save_button->connect("pressed", callable_mp(this, &AIChatDock::_on_save_image_pressed).bind(p_base64_data, "png"));
	details_container->add_child(save_button);
}

bool AIChatDock::_save_base64_image_to_path(const String &p_base64_data, const String &p_file_path) {
	if (p_base64_data.is_empty() || p_file_path.is_empty()) {
		return false;
	}
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(p_base64_data);
	if (image_data.size() == 0) {
		return false;
	}
	Ref<FileAccess> file = FileAccess::open(p_file_path, FileAccess::WRITE);
	if (file.is_null()) {
		return false;
	}
	file->store_buffer(image_data);
	file->close();

	// Immediately notify the editor's file system so the resource is imported and discoverable.
	if (EditorFileSystem::get_singleton()) {
		EditorFileSystem::get_singleton()->update_file(p_file_path);
		// Debounce costly/rescanning calls to avoid re-entrancy crashes in SceneTreeDock.
		// Schedule a single scan shortly after the last write.
		static uint64_t s_last_scan_request_ms = 0;
		s_last_scan_request_ms = OS::get_singleton()->get_ticks_msec();
		uint64_t scheduled_at = s_last_scan_request_ms;
		Ref<SceneTreeTimer> timer = get_tree()->create_timer(0.4, true);
		timer->connect("timeout", callable_mp(this, &AIChatDock::_on_filesystem_debounced_scan).bind(scheduled_at));
		
		// Force immediate reimport to generate .import file
		Vector<String> to_reimport;
		to_reimport.push_back(p_file_path);
		EditorFileSystem::get_singleton()->reimport_files(to_reimport);
		
		// Clear any cached load failure so the resource can be loaded fresh
		if (ResourceCache::has(p_file_path)) {
			Ref<Resource> cached = ResourceCache::get_ref(p_file_path);
			if (cached.is_valid()) {
				cached->reload_from_file();
			}
		}
	}
	return true;
}

void AIChatDock::_on_save_image_pressed(const String &p_base64_data, const String &p_format) {
	if (p_base64_data.is_empty()) {
		return;
	}
	
	// Store the image data for saving
	pending_save_image_data = p_base64_data;
	pending_save_image_format = p_format;
	
	// Set up the save dialog
	save_image_dialog->set_current_file("generated_image." + p_format);
	save_image_dialog->popup_centered(Size2(800, 600));
}
void AIChatDock::_on_save_image_location_selected(const String &p_file_path) {
	if (p_file_path.is_empty() || pending_save_image_data.is_empty()) {
		return;
	}
	
	String save_path = p_file_path;
	
	// Decode base64 to image data
	Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(pending_save_image_data);
	if (image_data.size() == 0) {
		print_line("AI Chat: Failed to decode image data for saving");
		return;
	}
	
	// Save the image file
	Ref<FileAccess> file = FileAccess::open(save_path, FileAccess::WRITE);
	if (file.is_null()) {
		print_line("AI Chat: Failed to open file for writing: " + save_path);
		return;
	}
	
	file->store_buffer(image_data);
	file->close();
	
	// Notify the editor file system so the image is imported and usable immediately.
	if (EditorFileSystem::get_singleton()) {
		EditorFileSystem::get_singleton()->update_file(save_path);
		// Debounce a follow-up scan to avoid re-entrancy issues in editor docks.
		static uint64_t s_last_scan_request_ms = 0;
		s_last_scan_request_ms = OS::get_singleton()->get_ticks_msec();
		uint64_t scheduled_at = s_last_scan_request_ms;
		Ref<SceneTreeTimer> timer = get_tree()->create_timer(0.4, true);
		timer->connect("timeout", callable_mp(this, &AIChatDock::_on_filesystem_debounced_scan).bind(scheduled_at));
		// Force immediate reimport for this file to avoid stale previews until next scan.
		Vector<String> to_reimport;
		to_reimport.push_back(save_path);
		EditorFileSystem::get_singleton()->reimport_files(to_reimport);
		
		// Clear any cached load failure so the resource can be loaded fresh
		if (ResourceCache::has(save_path)) {
			Ref<Resource> cached = ResourceCache::get_ref(save_path);
			if (cached.is_valid()) {
				cached->reload_from_file();
			}
		}
	}
	
	print_line("AI Chat: Image saved successfully to: " + save_path);
	
	// Show success notification to user
	if (EditorNode::get_singleton()) {
		EditorNode::get_singleton()->show_warning("Image saved successfully to: " + save_path.get_file());
	}
	
	// Clear pending data
	pending_save_image_data = "";
	pending_save_image_format = "";
}

void AIChatDock::_update_tool_placeholder_status(const String &p_tool_id, const String &p_tool_name, const String &p_status) {
	if (chat_container == nullptr) {
		return;
	}
	
	// Find the placeholder for this tool
	PanelContainer *placeholder = Object::cast_to<PanelContainer>(chat_container->find_child("tool_placeholder_" + p_tool_id, true, false));
	
	// If no placeholder exists, do NOT create another bubble here, as the
	// assistant message should be responsible for placeholder creation.
	// We avoid duplicates by simply returning and letting the deferred
	// tool result handler populate once placeholders are available.
	if (!placeholder) {
		return;
	}
	
	if (!placeholder) {
		return;
	}
	
	// Find the label in the placeholder and update its text
	HBoxContainer *tool_hbox = Object::cast_to<HBoxContainer>(placeholder->get_child(0));
	if (tool_hbox) {
		Label *tool_label = Object::cast_to<Label>(tool_hbox->get_child(0));
		if (tool_label) {
			if (p_status == "starting") {
				tool_label->set_text("Executing: " + p_tool_name + "...");
				tool_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(0.2, 0.8, 1.0, 1.0)); // Blue color for executing
			}
		}
	}
}

void AIChatDock::_create_assistant_message_for_backend_tool(const String &p_tool_name) {
    // Create a proper assistant message bubble that can later receive content
    _add_message_to_chat("assistant", "Executing: " + p_tool_name + "...");
}



void AIChatDock::_create_backend_tool_placeholder(const String &p_tool_id, const String &p_tool_name) {
	if (chat_container == nullptr) {
		return;
	}
	
	// Create a new message bubble for backend tools
	PanelContainer *bubble_panel = memnew(PanelContainer);
	bubble_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	
	Ref<StyleBoxFlat> bubble_style = memnew(StyleBoxFlat);
	bubble_style->set_bg_color(get_theme_color(SNAME("base_color"), SNAME("Editor")) * Color(0.9, 1.1, 0.9, 1.0)); // Slightly greenish for assistant
	bubble_style->set_content_margin_all(15);
	bubble_style->set_border_width_all(1);
	bubble_style->set_border_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
	bubble_style->set_corner_radius_all(10);
	bubble_panel->add_theme_style_override("panel", bubble_style);
	
	VBoxContainer *message_vbox = memnew(VBoxContainer);
	bubble_panel->add_child(message_vbox);
	
	// Create tool placeholder
	PanelContainer *placeholder = memnew(PanelContainer);
	placeholder->set_name("tool_placeholder_" + p_tool_id);
	message_vbox->add_child(placeholder);

	Ref<StyleBoxFlat> placeholder_style = memnew(StyleBoxFlat);
	placeholder_style->set_bg_color(get_theme_color(SNAME("dark_color_1"), SNAME("Editor")));
	placeholder_style->set_content_margin_all(10);
	placeholder_style->set_border_width_all(1);
	placeholder_style->set_border_color(get_theme_color(SNAME("dark_color_2"), SNAME("Editor")));
	placeholder_style->set_corner_radius_all(5);
	placeholder->add_theme_style_override("panel", placeholder_style);

	HBoxContainer *tool_hbox = memnew(HBoxContainer);
	placeholder->add_child(tool_hbox);

    Label *tool_label = memnew(Label);
    tool_label->set_text("Executing: " + p_tool_name + "...");
	tool_label->add_theme_color_override("font_color", get_theme_color(SNAME("font_color"), SNAME("Editor")) * Color(0.2, 0.8, 1.0, 1.0));
	tool_label->add_theme_icon_override("icon", get_theme_icon(SNAME("Tools"), SNAME("EditorIcons")));
	tool_hbox->add_child(tool_label);
	
	chat_container->add_child(bubble_panel);
	call_deferred("_scroll_to_bottom");
}

bool AIChatDock::_is_label_descendant_of_node(Node *p_label, Node *p_node) {
	if (p_label == nullptr || p_node == nullptr) {
		return false;
	}
	
	// Direct match
	if (p_label == p_node) {
		return true;
	}
	
	// Check if p_label is a child of p_node
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		if (child == p_label) {
			return true;
		}
		
		// Recursively check descendants
		if (_is_label_descendant_of_node(p_label, child)) {
			return true;
		}
	}
	
	return false;
}

// Attachment helper methods

void AIChatDock::_attach_scene_node(Node *p_node) {
	if (!p_node) {
		return;
	}
	
	// Check if node is already attached
	String node_path_str = String(p_node->get_path());
	for (const AttachedFile &existing_file : current_attached_files) {
		if (existing_file.is_node && existing_file.node_path == p_node->get_path()) {
			return; // Already attached
		}
	}
	
	AIChatDock::AttachedFile attached_file;
	attached_file.path = node_path_str;
	attached_file.name = String(p_node->get_name()) + " (" + String(p_node->get_class()) + ")";
    attached_file.content = _truncate_text_for_context(_get_node_info_string(p_node));
	attached_file.is_node = true;
	attached_file.node_path = p_node->get_path();
	attached_file.node_type = p_node->get_class();
	attached_file.mime_type = "application/godot-node";
	
	current_attached_files.push_back(attached_file);
	_update_attached_files_display();
}

void AIChatDock::_attach_current_script() {
	ScriptEditor *script_editor = EditorInterface::get_singleton()->get_script_editor();
	if (!script_editor) {
		return;
	}
	
	ScriptTextEditor *text_editor = Object::cast_to<ScriptTextEditor>(script_editor->get_current_editor());
	if (!text_editor) {
		return;
	}
	
	Ref<Script> script = text_editor->get_edited_resource();
	if (script.is_null()) {
		return;
	}
	
	String script_path = script->get_path();
	if (script_path.is_empty()) {
		return;
	}
	
	// Check if script is already attached
	for (const AttachedFile &existing_file : current_attached_files) {
		if (existing_file.path == script_path) {
			return; // Already attached
		}
	}
	
	AIChatDock::AttachedFile attached_file;
	attached_file.path = script_path;
	attached_file.name = script_path.get_file();
    attached_file.content = _truncate_text_for_context(text_editor->get_code_editor()->get_text_editor()->get_text());
	attached_file.is_image = false;
	attached_file.mime_type = _get_mime_type_from_extension(script_path);
	
	current_attached_files.push_back(attached_file);
	_update_attached_files_display();
}

void AIChatDock::_populate_scene_tree_recursive(Node *p_node, TreeItem *p_parent) {
	if (!p_node || !p_parent) {
		return;
	}
	
	for (int i = 0; i < p_node->get_child_count(); i++) {
		Node *child = p_node->get_child(i);
		TreeItem *item = scene_tree->create_item(p_parent);
		item->set_text(0, String(child->get_name()) + " (" + String(child->get_class()) + ")");
		item->set_metadata(0, child->get_path());
		
		// Set appropriate icon based on node type
		if (child->is_class("Node2D")) {
			item->set_icon(0, get_theme_icon(SNAME("Node2D"), SNAME("EditorIcons")));
		} else if (child->is_class("Node3D")) {
			item->set_icon(0, get_theme_icon(SNAME("Node3D"), SNAME("EditorIcons")));
		} else if (child->is_class("Control")) {
			item->set_icon(0, get_theme_icon(SNAME("Control"), SNAME("EditorIcons")));
		} else if (child->is_class("CanvasItem")) {
			item->set_icon(0, get_theme_icon(SNAME("CanvasItem"), SNAME("EditorIcons")));
		} else {
			item->set_icon(0, get_theme_icon(SNAME("Node"), SNAME("EditorIcons")));
		}
		
		// Recursively populate children
		if (child->get_child_count() > 0) {
			_populate_scene_tree_recursive(child, item);
		}
	}
}
String AIChatDock::_get_node_info_string(Node *p_node) {
	if (!p_node) {
		return "";
	}
	
	String info = "Node Information:\n";
	info += "Name: " + p_node->get_name() + "\n";
	info += "Type: " + p_node->get_class() + "\n";
	info += "Path: " + String(p_node->get_path()) + "\n";
	
	// Add position information for spatial nodes
	if (p_node->is_class("Node2D")) {
		Node2D *node2d = Object::cast_to<Node2D>(p_node);
		if (node2d) {
			info += "Position: " + String(node2d->get_position()) + "\n";
			info += "Rotation: " + String::num(node2d->get_rotation()) + "\n";
			info += "Scale: " + String(node2d->get_scale()) + "\n";
		}
	} else if (p_node->is_class("Node3D")) {
		Node3D *node3d = Object::cast_to<Node3D>(p_node);
		if (node3d) {
			info += "Position: " + String(node3d->get_position()) + "\n";
			info += "Rotation: " + String(node3d->get_rotation()) + "\n";
			info += "Scale: " + String(node3d->get_scale()) + "\n";
		}
	}
	
	// Add basic properties
	List<PropertyInfo> properties;
	p_node->get_property_list(&properties);
	
	info += "\nKey Properties:\n";
	int prop_count = 0;
	for (const PropertyInfo &prop : properties) {
		if (prop_count >= 10) break; // Limit to first 10 properties
		if (prop.usage & PROPERTY_USAGE_EDITOR && prop.type != Variant::OBJECT) {
			Variant value = p_node->get(prop.name);
			info += "  " + prop.name + ": " + String(value) + "\n";
			prop_count++;
		}
	}
	
	// Add children count
	if (p_node->get_child_count() > 0) {
		info += "\nChildren: " + String::num(p_node->get_child_count()) + " nodes\n";
	}
	
	return info;
}

String AIChatDock::get_machine_id() const {
	// Generate a unique machine ID based on hardware characteristics
	String machine_id = OS::get_singleton()->get_unique_id();
	if (machine_id.is_empty()) {
		// Fallback to processor name + OS name
		machine_id = OS::get_singleton()->get_processor_name() + "_" + OS::get_singleton()->get_name();
		machine_id = machine_id.replace(" ", "_").replace("(", "").replace(")", "");
	}
	return machine_id;
}

// ========== EMBEDDING SYSTEM IMPLEMENTATION ==========

void AIChatDock::_initialize_embedding_system() {
    print_line("AI Chat: 🔧 Initializing cloud-based embedding system");

    // Connect to editor file system signals for automatic reindexing
    // This connection does not require authentication; it only sets up callbacks.
    if (EditorFileSystem::get_singleton()) {
        if (!EditorFileSystem::get_singleton()->is_connected("filesystem_changed", callable_mp(this, &AIChatDock::_on_filesystem_changed))) {
            EditorFileSystem::get_singleton()->connect("filesystem_changed", callable_mp(this, &AIChatDock::_on_filesystem_changed));
        }
        if (!EditorFileSystem::get_singleton()->is_connected("sources_changed", callable_mp(this, &AIChatDock::_on_sources_changed))) {
            EditorFileSystem::get_singleton()->connect("sources_changed", callable_mp(this, &AIChatDock::_on_sources_changed));
        }
        print_line("AI Chat: 🔗 Connected to EditorFileSystem change signals (filesystem_changed, sources_changed)");
    } else {
        print_line("AI Chat: ⚠️ EditorFileSystem not ready; change signals not connected");
    }

    // Connect to precise save signals from EditorNode to index only changed files
    if (EditorNode::get_singleton()) {
        if (!EditorNode::get_singleton()->is_connected("resource_saved", callable_mp(this, &AIChatDock::_on_editor_resource_saved))) {
            EditorNode::get_singleton()->connect("resource_saved", callable_mp(this, &AIChatDock::_on_editor_resource_saved), CONNECT_DEFERRED);
        }
        if (!EditorNode::get_singleton()->is_connected("scene_saved", callable_mp(this, &AIChatDock::_on_editor_scene_saved))) {
            EditorNode::get_singleton()->connect("scene_saved", callable_mp(this, &AIChatDock::_on_editor_scene_saved), CONNECT_DEFERRED);
        }
        print_line("AI Chat: 🔗 Connected to EditorNode save signals (resource_saved, scene_saved)");
    }

    // Create HTTPRequest for embedding API calls
    if (!embedding_request) {
        embedding_request = memnew(HTTPRequest);
        add_child(embedding_request);
        embedding_request->connect("request_completed", callable_mp(this, &AIChatDock::_on_embedding_request_completed));
    }

    // Setup status timer for animated dots
    if (!embedding_status_timer) {
        embedding_status_timer = memnew(Timer);
        embedding_status_timer->set_wait_time(0.5);
        embedding_status_timer->set_one_shot(false);
        embedding_status_timer->connect("timeout", callable_mp(this, &AIChatDock::_on_embedding_status_tick));
        add_child(embedding_status_timer);
    }

    // Setup periodic poll timer for background indexing
    if (!embedding_poll_timer) {
        embedding_poll_timer = memnew(Timer);
        embedding_poll_timer->set_wait_time(embedding_poll_seconds);
        embedding_poll_timer->set_one_shot(false);
        embedding_poll_timer->connect("timeout", callable_mp(this, &AIChatDock::_on_embedding_poll_tick));
        add_child(embedding_poll_timer);
        embedding_poll_timer->start();
        print_line("AI Chat: ⏱️ Enabled periodic indexing poll every " + String::num_int64(embedding_poll_seconds) + "s");
    }

    embedding_system_initialized = true;
    _set_embedding_status("Ready to index", false);

    if (!_is_user_authenticated()) {
        current_user_id = "guest:" + get_machine_id();
        current_user_name = "Guest";
        auth_token = "";
        _update_user_status();
        print_line("AI Chat: ℹ️ Embedding system ready; indexing as guest session");
    }

    // Defer status/indexing to avoid overlapping requests right after init
    print_line("AI Chat: ✅ Embedding system initialized successfully");
}
void AIChatDock::_perform_initial_indexing() {
	print_line("AI Chat: 📚 Starting project indexing...");
	print_line("AI Chat: 🔍 DEBUG - embedding_system_initialized: " + String(embedding_system_initialized ? "true" : "false"));
	print_line("AI Chat: 🔍 DEBUG - embedding_in_progress: " + String(embedding_in_progress ? "true" : "false"));
	
	if (!embedding_system_initialized) {
		print_line("AI Chat: ❌ Cannot start indexing - system not initialized");
		return;
	}
	
	if (embedding_in_progress) {
		print_line("AI Chat: ⏳ Indexing already in progress");
		return;
	}
	
	print_line("AI Chat: ✅ All checks passed, starting file scan...");
	_set_embedding_status("Scanning files", true);
	
	// Always use cloud-ready approach: scan files and send content
	// This works both locally and when deployed to cloud
	print_line("AI Chat: 📁 About to call _scan_and_index_project_files...");
	_scan_and_index_project_files();
}

void AIChatDock::_send_embedding_request(const String &p_action, const Dictionary &p_data) {
	if (!embedding_request || embedding_request_busy) {
		print_line("AI Chat: ❌ Cannot send embedding request - busy or not initialized");
		return;
	}
	
	String embed_url = _get_embed_base_url() + "/embed";
	
	Dictionary request_data;
	request_data["action"] = p_action;
	
	// Always include project_root for all embedding requests
	request_data["project_root"] = _get_project_root_path();
	
	if (!p_data.is_empty()) {
		for (const Variant *key = p_data.next(); key; key = p_data.next(key)) {
			request_data[*key] = p_data[*key];
		}
	}
	
	Ref<JSON> json;
	json.instantiate();
	String request_body = json->stringify(request_data);
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	// Add authentication headers
	if (!auth_token.is_empty()) {
		headers.push_back("Authorization: Bearer " + auth_token);
	}
	headers.push_back("X-User-ID: " + current_user_id);
	headers.push_back("X-Machine-ID: " + get_machine_id());
	
    print_line("AI Chat: 📡 Sending embedding request: " + p_action + " to " + embed_url +
        " (project_root=" + _get_project_root_path() + ")");
	
	embedding_request_busy = true;
	Error err = embedding_request->request(embed_url, headers, HTTPClient::METHOD_POST, request_body);
	
	if (err != OK) {
		print_line("AI Chat: ❌ Failed to send embedding request: " + String::num_int64(err));
		embedding_request_busy = false;
		_set_embedding_status("Request failed", false);
	}
}
void AIChatDock::_on_embedding_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	embedding_request_busy = false;
	
	print_line("AI Chat: 📨 Embedding request completed - Result: " + String::num_int64(p_result) + ", Code: " + String::num_int64(p_code));
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		String error_msg = "Request failed (" + String::num_int64(p_code) + ")";
		print_line("AI Chat: ❌ " + error_msg);
		_set_embedding_status(error_msg, false);
		return;
	}
	
	String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
	
	Ref<JSON> json;
	json.instantiate();
	Error parse_err = json->parse(response_text);
	
	if (parse_err != OK) {
		print_line("AI Chat: ❌ Failed to parse embedding response");
		_set_embedding_status("Parse error", false);
		return;
	}
	
	Dictionary response = json->get_data();
	bool success = response.get("success", false);
	String action = response.get("action", "");
	
	if (!success) {
		String error = response.get("error", "Unknown error");
		print_line("AI Chat: ❌ Embedding request failed: " + error);
		_set_embedding_status("Error: " + error, false);
		return;
	}
	
	print_line("AI Chat: ✅ Embedding action '" + action + "' completed successfully");
	
	if (action == "index_project") {
		Dictionary stats = response.get("stats", Dictionary());
		int total = stats.get("total", 0);
		int indexed = stats.get("indexed", 0);
		int skipped = stats.get("skipped", 0);
		
		String status_text = "Indexed " + String::num_int64(indexed) + "/" + String::num_int64(total) + " files";
		if (skipped > 0) {
			status_text += " (" + String::num_int64(skipped) + " skipped)";
		}
		
		_set_embedding_status(status_text, false);
		initial_indexing_done = true;
		
		print_line("AI Chat: 🎉 Project indexing completed - " + status_text);
		
	} else if (action == "index_files") {
		// Handle batch processing response
		Dictionary stats = response.get("stats", Dictionary());
		int batch_indexed = stats.get("indexed", 0);
		int batch_skipped = stats.get("skipped", 0);
		int batch_failed = stats.get("failed", 0);
		
		print_line("AI Chat: ✅ Batch completed - indexed: " + String::num_int64(batch_indexed) + ", skipped: " + String::num_int64(batch_skipped) + ", failed: " + String::num_int64(batch_failed));
		
		// Check if we need to send more batches
		if (current_batch_info.has("current_batch") && current_batch_info.has("total_batches")) {
			int current_batch = current_batch_info["current_batch"];
			int total_batches = current_batch_info["total_batches"];
			
			if (current_batch < total_batches) {
				// Send next batch
				int next_batch = current_batch + 1;
				int start_index = current_batch_info["start_index"];
				int batch_size = current_batch_info["batch_size"];
				Array all_files = current_batch_info["all_files"];
				
				int next_start_index = start_index + batch_size;
				
				_set_embedding_status("Indexing files (batch " + String::num_int64(next_batch) + "/" + String::num_int64(total_batches) + ")", true);
				call_deferred("_send_file_batch", all_files, next_start_index, batch_size, next_batch, total_batches);
			} else {
				// All batches completed
				_set_embedding_status("All files indexed successfully", false);
				initial_indexing_done = true;
				print_line("AI Chat: 🎉 All file batches completed successfully");
			}
		}
		
	} else if (action == "status") {
		Dictionary stats = response.get("stats", Dictionary());
		int files_indexed = stats.get("files_indexed", 0);
		int total_chunks = stats.get("total_chunks", 0);
		
		if (files_indexed > 0) {
			String status_text = String::num_int64(files_indexed) + " files indexed (" + String::num_int64(total_chunks) + " chunks)";
			_set_embedding_status(status_text, false);
			initial_indexing_done = true;
		} else {
			_set_embedding_status("No files indexed", false);
			initial_indexing_done = false;
		}
		
	} else if (action == "clear") {
		_set_embedding_status("Index cleared", false);
		initial_indexing_done = false;
	}
}

String AIChatDock::_get_project_root_path() {
	return ProjectSettings::get_singleton()->globalize_path("res://");
}

String AIChatDock::_get_embed_base_url() {
	// Use same endpoint as chat but for embedding operations
	String base_url = api_endpoint;
	
	// Remove /chat suffix if present and replace with embedding endpoint
	if (base_url.ends_with("/chat")) {
		base_url = base_url.substr(0, base_url.length() - 5);
	}
	
	return base_url;
}

void AIChatDock::_set_embedding_status(const String &p_text, bool p_busy) {
	if (!embedding_status_label) {
		return;
	}
	
	embedding_in_progress = p_busy;
	embedding_status_base = p_text;
	embedding_status_dots = 0;
	
	if (p_busy) {
		embedding_status_label->set_text(p_text + "...");
		embedding_status_label->set_modulate(Color(1.0, 0.8, 0.0)); // Yellow for in-progress
		if (embedding_status_timer) {
			embedding_status_timer->start();
		}
	} else {
		embedding_status_label->set_text(p_text);
		embedding_status_label->set_modulate(Color(0.7, 0.7, 0.7)); // Gray for idle
		if (embedding_status_timer) {
			embedding_status_timer->stop();
		}
	}
}

void AIChatDock::_on_embedding_status_tick() {
	if (!embedding_in_progress || !embedding_status_label) {
		return;
	}
	
	embedding_status_dots = (embedding_status_dots + 1) % 4;
	String dots = "";
	for (int i = 0; i < embedding_status_dots; i++) {
		dots += ".";
	}
	
	embedding_status_label->set_text(embedding_status_base + dots);
}

bool AIChatDock::_should_index_file(const String &p_file_path) {
	// Only allow textual/code formats; backend enforces too
	String ext = p_file_path.get_extension().to_lower();
	static const HashSet<String> allowed_text_ext = {
		"gd", "cs", "c", "cpp", "h", "hpp", "glsl", "shader", "gdshader",
		"tscn", "scn", "tres", "res", "godot", "import",
		"json", "cfg", "ini", "yaml", "yml", "xml", "md", "txt", "rst"
	};
	if (!allowed_text_ext.has(ext)) {
		return false;
	}
	String filename = p_file_path.get_file();
	if (filename.begins_with(".")) {
		return false;
	}
	return true;
}
void AIChatDock::_update_file_embedding(const String &p_file_path) {
	if (!embedding_system_initialized || !_should_index_file(p_file_path)) {
		return;
	}
	
	// Cloud-ready: read file content and send via index_files
	Dictionary file_data = _read_file_for_indexing(p_file_path, _get_project_root_path());
	if (file_data.is_empty()) {
		print_line("AI Chat: ❌ Failed to read file for embedding update: " + p_file_path);
		return;
	}
	
	Array files_arr;
	files_arr.push_back(file_data);
	
	Dictionary payload;
	payload["files"] = files_arr;
	Dictionary batch_info;
	batch_info["current"] = 1;
	batch_info["total"] = 1;
	batch_info["files_in_batch"] = 1;
	payload["batch_info"] = batch_info;
	
	_send_embedding_request("index_files", payload);
}
void AIChatDock::_remove_file_embedding(const String &p_file_path) {
	if (!embedding_system_initialized) {
		return;
	}
	
	Dictionary payload;
	payload["file_path"] = p_file_path;
	payload["project_root"] = _get_project_root_path();
	
	_send_embedding_request("remove_file", payload);
}

void AIChatDock::_on_filesystem_changed() {
    print_line("AI Chat: 📁 filesystem_changed signal received");
    // Only rely on per-file saved signals for accuracy; skip project-wide reindex on generic FS changes.
    pending_fs_changes = true;
}

void AIChatDock::_on_sources_changed(bool p_exist) {
    // Called when source files change
    print_line("AI Chat: 📝 sources_changed signal received (exist=" + String(p_exist ? "true" : "false") + ")");
    // Do nothing here; precise per-file handlers will trigger indexing.
}

void AIChatDock::_on_editor_resource_saved(Object *p_res) {
    if (!p_res) {
        return;
    }
    Ref<Resource> res = Ref<Resource>(Object::cast_to<Resource>(p_res));
    if (res.is_null()) {
        return;
    }
    String path = res->get_path();
    if (path.is_empty()) {
        return;
    }
    print_line("AI Chat: 💾 resource_saved -> " + path);
    // Clear overlay on save to ensure subsequent reads/checks use disk content
    EditorTools::clear_preview_overlay(path);
    // Always queue for periodic batch to avoid immediate requests
    String abs_path_res = ProjectSettings::get_singleton()->globalize_path(path);
    if (_should_index_file(abs_path_res)) {
        pending_changed_files.insert(abs_path_res);
        pending_fs_changes = true;
        // Optional: surface status as queued (non-blocking)
        _set_embedding_status("Queued file for indexing", false);
    }
}

void AIChatDock::_on_editor_scene_saved(const String &p_path) {
    print_line("AI Chat: 💾 scene_saved -> " + p_path);
    // Always queue for periodic batch to avoid immediate requests
    String abs_path = ProjectSettings::get_singleton()->globalize_path(p_path);
    if (_should_index_file(abs_path)) {
        pending_changed_files.insert(abs_path);
        pending_fs_changes = true;
        _set_embedding_status("Queued file for indexing", false);
    }
}

void AIChatDock::_on_embedding_poll_tick() {
    if (!embedding_system_initialized || !_is_user_authenticated() || embedding_request_busy) {
        return;
    }
    // Prefer batching changed files; otherwise do lightweight incremental project scan
    if (!pending_changed_files.is_empty()) {
        // Batch changed files using cloud-ready endpoint to minimize requests
        Array files_arr;
        for (const String &abs_path : pending_changed_files) {
            // Read file contents and hash (same as bulk flow)
            Dictionary file_data = _read_file_for_indexing(abs_path, _get_project_root_path());
            if (!file_data.is_empty()) {
                files_arr.push_back(file_data);
            }
        }
        pending_changed_files.clear();
        if (!files_arr.is_empty()) {
            Dictionary payload;
            payload["files"] = files_arr;
            Dictionary batch_info;
            batch_info["current"] = 1;
            batch_info["total"] = 1;
            batch_info["files_in_batch"] = files_arr.size();
            payload["batch_info"] = batch_info;
            _set_embedding_status("Indexing changed files", true);
            _send_embedding_request("index_files", payload);
            last_index_request_ms = OS::get_singleton()->get_ticks_msec();
            return;
        }
    }
    // If there were FS changes but nothing queued, avoid aggressive full project scan
    if (pending_fs_changes) {
        print_line("AI Chat: 📁 FS changes detected but no specific files queued. Skipping full project scan to avoid unnecessary re-indexing.");
        // Only clear the flag - don't trigger full project indexing for generic FS changes
        // Real changes should be caught by the specific save handlers above
        pending_fs_changes = false;
    }
}

void AIChatDock::_suggest_relevant_files(const String &p_query) {
	// TODO: Implement smart file suggestions based on embedding similarity
	print_line("AI Chat: 🔍 Smart file suggestions not implemented yet for query: " + p_query);
}

void AIChatDock::_auto_attach_relevant_context() {
	// TODO: Implement automatic context attachment based on message content
	print_line("AI Chat: 🤖 Auto context attachment not implemented yet");
}

void AIChatDock::_scan_and_index_project_files() {
	print_line("AI Chat: 📁 Scanning project files for indexing...");
	
	String project_root = _get_project_root_path();
	print_line("AI Chat: 🔍 DEBUG - project_root: " + project_root);
	
	Array file_contents = Array();
	int files_processed = 0;
	int files_skipped = 0;
	
	// Get all files in project recursively
	print_line("AI Chat: 🔄 Starting recursive directory scan...");
	_scan_directory_recursive(project_root, project_root, file_contents, files_processed, files_skipped);
	
	print_line("AI Chat: 📊 Scan complete - " + String::num_int64(files_processed) + " files to index, " + String::num_int64(files_skipped) + " skipped");
	print_line("AI Chat: 🔍 DEBUG - file_contents.size(): " + String::num_int64(file_contents.size()));
	
	if (file_contents.size() == 0) {
		print_line("AI Chat: ❌ No files found to index!");
		_set_embedding_status("No files to index", false);
		return;
	}
	
	// Send files in batches to avoid huge HTTP requests
	int batch_size = 20; // Process 20 files at a time
	int total_batches = (file_contents.size() + batch_size - 1) / batch_size;
	
	print_line("AI Chat: 📦 Preparing " + String::num_int64(total_batches) + " batches of " + String::num_int64(batch_size) + " files each");
	_set_embedding_status("Indexing files (batch 1/" + String::num_int64(total_batches) + ")", true);
	_send_file_batch(file_contents, 0, batch_size, 1, total_batches);
}

void AIChatDock::_scan_directory_recursive(const String &p_dir_path, const String &p_project_root, Array &p_file_contents, int &p_files_processed, int &p_files_skipped) {
	Ref<DirAccess> dir = DirAccess::open(p_dir_path);
	if (dir.is_null()) {
		print_line("AI Chat: ❌ Cannot access directory: " + p_dir_path);
		return;
	}
	
	dir->list_dir_begin();
	String file_name = dir->get_next();
	
	while (!file_name.is_empty()) {
		String full_path = p_dir_path.path_join(file_name);
		
		if (dir->current_is_dir()) {
			// Skip hidden directories and common build/cache dirs
			if (!file_name.begins_with(".") && 
				file_name != "build" && 
				file_name != "bin" && 
				file_name != "obj" && 
				file_name != "__pycache__") {
				_scan_directory_recursive(full_path, p_project_root, p_file_contents, p_files_processed, p_files_skipped);
			} else {
				p_files_skipped++;
			}
		} else {
			// Check if we should index this file
			if (_should_index_file(full_path)) {
				Dictionary file_data = _read_file_for_indexing(full_path, p_project_root);
				if (!file_data.is_empty()) {
					p_file_contents.push_back(file_data);
					p_files_processed++;
				} else {
					p_files_skipped++;
				}
			} else {
				p_files_skipped++;
			}
		}
		
		file_name = dir->get_next();
	}
	
	dir->list_dir_end();
}

Dictionary AIChatDock::_read_file_for_indexing(const String &p_file_path, const String &p_project_root) {
	Ref<FileAccess> file = FileAccess::open(p_file_path, FileAccess::READ);
	if (file.is_null()) {
		print_line("AI Chat: ❌ Cannot read file: " + p_file_path);
		return Dictionary();
	}
	
	String content = file->get_as_text(true); // Skip BOM if present
	file->close();

	// Sanitize content to avoid invalid JSON: strip control chars except whitespace
	{
		String sanitized;
		for (int i = 0; i < content.length(); i++) {
			char32_t ch = content[i];
			if (ch >= 32 || ch == '\n' || ch == '\t' || ch == '\r') {
				sanitized += ch;
			}
		}
		content = sanitized;
	}

	// Cap per-file payload size
	const int64_t max_len = 200000; // ~200 KB
	if (content.length() > max_len) {
		content = content.substr(0, max_len);
	}
	
	// Skip empty files or files with only whitespace
	if (content.strip_edges().is_empty()) {
		return Dictionary();
	}
	
	// Get relative path from project root
	String relative_path = p_file_path.replace(p_project_root, "");
	if (relative_path.begins_with("/") || relative_path.begins_with("\\")) {
		relative_path = relative_path.substr(1);
	}
	
	// Calculate content hash for change detection
	String content_hash = _calculate_content_hash(content);
	
	Dictionary file_data;
	file_data["path"] = relative_path;
	file_data["content"] = content;
	file_data["hash"] = content_hash;
	file_data["size"] = content.length();
	
	return file_data;
}

String AIChatDock::_calculate_content_hash(const String &p_content) {
	// Simple hash calculation using Godot's built-in hash function
	uint32_t hash_value = p_content.hash();
	return String::num_uint64(hash_value, 16);
}

void AIChatDock::_send_file_batch(const Array &p_all_files, int p_start_index, int p_batch_size, int p_current_batch, int p_total_batches) {
	Array batch_files = Array();
	int end_index = MIN(p_start_index + p_batch_size, p_all_files.size());
	
	// Create batch
	for (int i = p_start_index; i < end_index; i++) {
		batch_files.push_back(p_all_files[i]);
	}
	
	// Prepare request data
	Dictionary batch_info_dict;
	batch_info_dict["current"] = p_current_batch;
	batch_info_dict["total"] = p_total_batches;
	batch_info_dict["files_in_batch"] = batch_files.size();
	
	Dictionary payload;
	payload["files"] = batch_files;
	payload["batch_info"] = batch_info_dict;
	
	// Store batch info for handling response
	current_batch_info["start_index"] = p_start_index;
	current_batch_info["batch_size"] = p_batch_size;
	current_batch_info["current_batch"] = p_current_batch;
	current_batch_info["total_batches"] = p_total_batches;
	current_batch_info["all_files"] = p_all_files;
	
	print_line("AI Chat: 📤 Sending batch " + String::num_int64(p_current_batch) + "/" + String::num_int64(p_total_batches) + " (" + String::num_int64(batch_files.size()) + " files)");
	
	_send_embedding_request("index_files", payload);
}

void AIChatDock::_register_pending_edit(const String &p_path, const NodePath &p_btns_path, const NodePath &p_status_label_path) {
	Dictionary info;
	info["btns_path"] = p_btns_path;
	info["status_path"] = p_status_label_path;
	pending_edits[p_path] = info;
	print_line("AI Chat: Registered pending edit for: " + p_path);
	print_line("  - btns_path: " + String(p_btns_path));
	print_line("  - status_path: " + String(p_status_label_path));
	print_line("  - Total pending edits: " + itos(pending_edits.size()));
}

void AIChatDock::_clear_pending_edit(const String &p_path) {
	if (pending_edits.has(p_path)) {
		pending_edits.erase(p_path);
		print_line("Cleared pending edit for: " + p_path);
	}
}

void AIChatDock::_on_script_editor_save(const String &p_path) {
	print_line("AI Chat: _on_script_editor_save called for: " + p_path);
	
	// List all pending edits for debugging
	print_line("AI Chat: Current pending edits:");
	Array keys = pending_edits.keys();
	for (int i = 0; i < keys.size(); i++) {
		print_line("  - " + String(keys[i]));
	}
	
	// Called when a file is saved in the script editor (including diff acceptance)
	// Check if this file has a pending edit in the chat
	// Try both the path as-is and with project:// prefix
	String normalized_path = p_path;
	if (p_path.begins_with("res://")) {
		// Path is already relative to project
	} else if (p_path.is_absolute_path()) {
		// Convert absolute to res:// if it's within the project
		String project_path = ProjectSettings::get_singleton()->get_resource_path();
		if (p_path.begins_with(project_path)) {
			normalized_path = "res://" + p_path.substr(project_path.length()).lstrip("/");
		}
	}
	
	if (!pending_edits.has(normalized_path) && !pending_edits.has(p_path)) {
		print_line("AI Chat: No pending edit found for: " + p_path + " or " + normalized_path);
		return;
	}
	
	// Use whichever path matched
	String matched_path = pending_edits.has(normalized_path) ? normalized_path : p_path;
	
	Dictionary info = pending_edits[matched_path];
	NodePath btns_path = info.get("btns_path", NodePath());
	NodePath status_path = info.get("status_path", NodePath());
	
	// Update the chat bubble UI to show it was accepted
	if (!btns_path.is_empty() && has_node(btns_path)) {
		if (Control *btns = Object::cast_to<Control>(get_node(btns_path))) {
			btns->set_visible(false);
		}
	}
	
	if (!status_path.is_empty() && has_node(status_path)) {
		if (Label *status = Object::cast_to<Label>(get_node(status_path))) {
			String text = status->get_text();
			if (!text.ends_with(" — Accepted") && !text.ends_with(" — Accepted (via Script Editor)")) {
				status->set_text(text + " — Accepted (via Script Editor)");
			}
		}
	}
	
	// Clear the pending edit
	_clear_pending_edit(matched_path);
	
	// Clear any preview overlay since the file was saved
	EditorTools::clear_preview_overlay(matched_path);
	
	print_line("AI Chat: Updated bubble for accepted diff: " + p_path);
}

void AIChatDock::_connect_script_editor_signals() {
	print_line("AI Chat: _connect_script_editor_signals called");
	ScriptEditor *script_editor = ScriptEditor::get_singleton();
	if (script_editor) {
		print_line("AI Chat: ScriptEditor singleton found");
		// Connect to the script saved signal if not already connected
		if (!script_editor->is_connected("script_saved", callable_mp(this, &AIChatDock::_on_script_editor_save))) {
			// First check if the signal exists
			if (!script_editor->has_signal("script_saved")) {
				print_line("AI Chat: WARNING - ScriptEditor does not have script_saved signal!");
				return;
			}
			
			Error err = script_editor->connect("script_saved", callable_mp(this, &AIChatDock::_on_script_editor_save));
			if (err == OK) {
				print_line("AI Chat: Successfully connected to ScriptEditor::script_saved signal");
				
				// Verify the connection
				List<Node::Connection> connections;
				script_editor->get_signal_connection_list("script_saved", &connections);
				print_line("AI Chat: Verified - script_saved now has " + itos(connections.size()) + " connections");
			} else {
				print_line("AI Chat: ERROR - Failed to connect to ScriptEditor::script_saved signal, error: " + itos(err));
			}
		} else {
			print_line("AI Chat: Already connected to ScriptEditor::script_saved signal");
		}
	} else {
		print_line("AI Chat: ERROR - ScriptEditor singleton is null, retrying in 0.5 seconds");
		// Retry after a delay
		get_tree()->create_timer(0.5)->connect("timeout", callable_mp(this, &AIChatDock::_connect_script_editor_signals), CONNECT_ONE_SHOT);
	}
}
void AIChatDock::_show_diff_in_script_editor(const String &p_path, const String &p_original, const String &p_modified) {
	// Ensure signals are connected before showing diff
	_connect_script_editor_signals();
	
	ScriptEditor *script_editor = ScriptEditor::get_singleton();
	if (!script_editor) {
		return;
	}
	
	// Load or create the script resource
	Ref<Resource> resource = ResourceLoader::load(p_path);
	Ref<Script> script = resource;
	
	if (!script.is_valid()) {
		// For new files, create a temporary script
		if (p_path.get_extension() == "gd") {
			Ref<GDScript> gd_script;
			gd_script.instantiate();
			gd_script->set_path(p_path);
			gd_script->set_source_code(p_original.is_empty() ? "extends Node\n\n" : p_original);
			script = gd_script;
		} else {
			// For other script types, we might need to handle them differently
			print_line("AI Chat: Cannot create temporary script for non-GDScript file");
			return;
		}
	}
	
	// Open the script in the editor
	script_editor->edit(script);
	
	// We need to wait a frame for the script editor to be ready
	call_deferred("_show_diff_in_script_editor_deferred", p_path, p_original, p_modified);
}

void AIChatDock::_show_diff_in_script_editor_deferred(const String &p_path, const String &p_original, const String &p_modified) {
	ScriptEditor *script_editor = ScriptEditor::get_singleton();
	if (!script_editor) {
		return;
	}
	
	// Get the current script editor instance (ScriptTextEditor)
	ScriptEditorBase *current = script_editor->get_current_editor();
	if (!current) {
		print_line("AI Chat: No current script editor found after deferral");
		return;
	}
	
	ScriptTextEditor *ste = Object::cast_to<ScriptTextEditor>(current);
	if (!ste) {
		print_line("AI Chat: Current editor is not a ScriptTextEditor");
		return;
	}
	
	// Show the diff in the script editor
	ste->set_diff(p_original, p_modified);
	
	// Switch to Script editor view
	EditorInterface::get_singleton()->set_main_screen_editor("Script");
	
	print_line("AI Chat: Showing diff in script editor for " + p_path);
}

void AIChatDock::_populate_cerebras_models() {
	// Fetch available models from backend
	String base_url = _get_api_base_url();
	String models_url = base_url + "/models";
	
	print_line("AI Chat: Fetching models from " + models_url);
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	Error err = models_http_request->request(models_url, headers, HTTPClient::METHOD_GET);
	if (err != OK) {
		print_line("AI Chat: Failed to request models: " + String::num_int64(err));
	}
}

void AIChatDock::_on_models_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		print_line("AI Chat: Models request failed - Result: " + String::num_int64(p_result) + ", Code: " + String::num_int64(p_code));
		return;
	}
	
	String response_text = String::utf8(reinterpret_cast<const char*>(p_body.ptr()), p_body.size());
	
	Ref<JSON> json;
	json.instantiate();
	Error parse_error = json->parse(response_text);
	
	if (parse_error != OK) {
		print_line("AI Chat: Failed to parse models response: " + response_text);
		return;
	}
	
	Variant result = json->get_data();
	if (result.get_type() != Variant::DICTIONARY) {
		print_line("AI Chat: Invalid models response format");
		return;
	}
	
	Dictionary response = result;
	if (!response.get("success", false)) {
		print_line("AI Chat: Models request was not successful");
		return;
	}
	
	Array models = response.get("models", Array());
	print_line("AI Chat: Received " + String::num_int64(models.size()) + " models from backend");
	
	// Clear existing Cerebras models from dropdown (keep base models)
	int items_to_remove = 0;
	for (int i = model_dropdown->get_item_count() - 1; i >= 0; i--) {
		String item_text = model_dropdown->get_item_text(i);
		if (item_text.begins_with("[FAST] ")) {
			model_dropdown->remove_item(i);
			items_to_remove++;
		}
	}
	if (items_to_remove > 0) {
		print_line("AI Chat: Removed " + String::num_int64(items_to_remove) + " old Cerebras models");
	}
	
	// Add new models to dropdown
	int added_models = 0;
	for (int i = 0; i < models.size(); i++) {
		Dictionary model_info = models[i];
		String model_name = model_info.get("name", "");
		String provider = model_info.get("provider", "");
		
		if (provider == "cerebras") {
			model_dropdown->add_item(model_name);
			added_models++;
		}
	}
	
	print_line("AI Chat: Added " + String::num_int64(added_models) + " Cerebras models to dropdown");
}

String AIChatDock::_get_api_base_url() {
	String base_url;
	String is_dev = OS::get_singleton()->get_environment("IS_DEV");
	if (is_dev.is_empty()) {
		is_dev = OS::get_singleton()->get_environment("DEV_MODE");
	}
	if (!is_dev.is_empty() && is_dev.to_lower() == "true") {
		base_url = "http://127.0.0.1:8000";
	} else {
		base_url = "https://gamechat.simplifine.com";
	}
	
	// Allow override via editor settings or environment variable
	if (EditorSettings::get_singleton() && EditorSettings::get_singleton()->has_setting("ai_chat/base_url")) {
		String override_url = EditorSettings::get_singleton()->get_setting("ai_chat/base_url");
		if (!override_url.is_empty()) {
			base_url = override_url;
		}
	} else if (!OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL").is_empty()) {
		base_url = OS::get_singleton()->get_environment("AI_CHAT_CLOUD_URL");
	}
	
	if (base_url.ends_with("/")) {
		base_url = base_url.substr(0, base_url.length() - 1);
	}
	
	return base_url;
}

AIChatDock::~AIChatDock() {
	// Wait for any background save to complete
	if (save_thread_busy && save_thread) {
		save_thread->wait_to_finish();
		memdelete(save_thread);
	}
	
	// Clean up mutex
	if (save_mutex) {
		memdelete(save_mutex);
	}
}