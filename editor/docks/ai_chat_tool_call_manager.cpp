/*
#include "ai_chat_content_alignment.h"
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */
#include "ai_chat_content_alignment.h"
#include "ai_chat_tool_call_manager.h"
#include "ai_chat_dock.h"
#include "ai_chat_streaming_indicator.h"
#include "ai_chat_tool_styling.h"
#include "ai_chat_streaming_manager.h"
#include "editor/editor_string_names.h"
#include "scene/resources/style_box_flat.h"

void AIChatToolCallManager::create_tool_call_bubbles(
	const Array &p_tool_calls,
	VBoxContainer *p_chat_container,
	RichTextLabel *p_current_assistant_label,
	AIChatDock *p_chat_dock
) {
	print_line("AI Chat: AIChatToolCallManager::create_tool_call_bubbles called with " + String::num_int64(p_tool_calls.size()) + " tool calls");
	
	if (!p_current_assistant_label || !p_chat_dock) {
		print_line("AI Chat: ERROR - create_tool_call_bubbles called but parameters are NULL!");
		return;
	}
	
	if (p_tool_calls.is_empty()) {
		print_line("AI Chat: create_tool_call_bubbles called with empty tool_calls array");
		return;
	}

	Control *bubble_panel = Object::cast_to<Control>(p_current_assistant_label->get_parent()->get_parent());
	if (!bubble_panel) {
		print_line("AI Chat: ERROR - Could not get bubble_panel from current_assistant_message_label hierarchy");
		return;
	}
	
	VBoxContainer *message_vbox = Object::cast_to<VBoxContainer>(bubble_panel->get_child(0));
	if (!message_vbox) {
		print_line("AI Chat: ERROR - Could not get message_vbox from bubble_panel");
		return;
	}
	
	print_line("AI Chat: Successfully found bubble panel and message vbox, creating tool placeholders");

	// Create a single container for all tool calls to group them together
	VBoxContainer *tools_container = memnew(VBoxContainer);
	tools_container->set_name("tools_container");
	// FIXED: Use less aggressive negative spacing to prevent overlap
	tools_container->add_theme_constant_override("separation", TOOLS_CONTAINER_SEPARATION);
	message_vbox->add_child(tools_container);

	// Create individual tool placeholders within the grouped container
	for (int i = 0; i < p_tool_calls.size(); i++) {
		Dictionary tool_call = p_tool_calls[i];
		String tool_call_id = tool_call.get("id", "");
		Dictionary function_dict = tool_call.get("function", Dictionary());
		String func_name = function_dict.get("name", "unknown_tool");
		String arguments_str = function_dict.get("arguments", "{}");
		
		print_line("AI Chat: Creating placeholder for tool: " + func_name + " (ID: " + tool_call_id + ")");

		// Check if placeholder already exists (from tool_starting message)
		PanelContainer *placeholder = nullptr;
		if (p_chat_container) {
			placeholder = Object::cast_to<PanelContainer>(p_chat_container->find_child("tool_placeholder_" + tool_call_id, true, false));
		}
		
		if (placeholder) {
			print_line("AI Chat: Placeholder already exists for " + tool_call_id + ", updating it instead of creating new one");
			// Update existing placeholder
			String descriptive_status = p_chat_dock->_get_immediate_tool_status(func_name, arguments_str);
			if (descriptive_status.is_empty()) {
				descriptive_status = p_chat_dock->_generate_executing_tool_message(func_name, arguments_str);
			}
			update_tool_placeholder_with_description(tool_call_id, func_name, "executing", descriptive_status, p_chat_container, p_chat_dock);
			continue; // Skip creating a new placeholder
		}

		// Create new placeholder
		placeholder = _create_tool_placeholder(tool_call_id, tools_container, p_chat_dock);
		if (placeholder) {
			// Generate descriptive status
			String descriptive_status = p_chat_dock->_get_immediate_tool_status(func_name, arguments_str);
			if (descriptive_status.is_empty()) {
				descriptive_status = p_chat_dock->_generate_executing_tool_message(func_name, arguments_str);
			}
			if (descriptive_status.is_empty()) {
				descriptive_status = func_name + "..."; // Final fallback
			}
			
			_create_tool_executing_ui(placeholder, tool_call_id, func_name, descriptive_status, p_chat_dock);
			print_line("AI Chat: Created placeholder " + String::num_int64(i + 1) + "/" + String::num_int64(p_tool_calls.size()) + " with status: " + descriptive_status);
		}
	}
	
	// Force the parent bubble panel to be visible after adding placeholders
	if (bubble_panel) {
		bubble_panel->set_visible(true);
		bubble_panel->queue_redraw();
		print_line("AI Chat: Forced bubble_panel visible after creating " + String::num_int64(p_tool_calls.size()) + " placeholders");
		
		// Also scroll to make sure it's in view
		p_chat_dock->call_deferred("_scroll_to_bottom");
		
		// Force entire chat UI to update
		if (p_chat_container) {
			p_chat_container->queue_redraw();
		}
		if (p_chat_dock->chat_scroll) {
			p_chat_dock->chat_scroll->queue_redraw();
		}
	}

	// Reposition streaming indicator after tool calls
	_reposition_streaming_indicator(p_current_assistant_label, p_chat_container, p_chat_dock);
	
	print_line("AI Chat: create_tool_call_bubbles completed successfully");
}

PanelContainer *AIChatToolCallManager::_create_tool_placeholder(
	const String &p_tool_call_id,
	VBoxContainer *p_tools_container,
	AIChatDock *p_chat_dock
) {
	if (!p_tools_container || !p_chat_dock) {
		return nullptr;
	}

	PanelContainer *placeholder = memnew(PanelContainer);
	placeholder->set_name("tool_placeholder_" + p_tool_call_id);
	p_tools_container->add_child(placeholder);

	Ref<StyleBoxFlat> placeholder_style = memnew(StyleBoxFlat);
	placeholder_style->set_bg_color(Color(0, 0, 0, 0)); // Transparent background
	placeholder_style->set_content_margin_all(0); // No padding
	placeholder_style->set_border_width_all(0); // No border
	placeholder_style->set_border_color(Color(0, 0, 0, 0)); // Transparent border
	placeholder_style->set_corner_radius_all(CORNER_RADIUS);
	placeholder->add_theme_style_override("panel", placeholder_style);

	return placeholder;
}

void AIChatToolCallManager::_create_tool_executing_ui(
	PanelContainer *p_placeholder,
	const String &p_tool_call_id,
	const String &p_function_name,
	const String &p_descriptive_status,
	AIChatDock *p_chat_dock
) {
	if (!p_placeholder || !p_chat_dock) {
		return;
	}

	VBoxContainer *tool_vbox = memnew(VBoxContainer);
	// FIXED: Use positive spacing instead of negative to prevent overlap
	tool_vbox->add_theme_constant_override("separation", TOOL_VBOX_SEPARATION);
	p_placeholder->add_child(tool_vbox);

	HBoxContainer *tool_hbox = memnew(HBoxContainer);
	// FIXED: Align with AI text (0px margin instead of 8px since AI text now has 0px margin too)
	AIChatContentAlignment::apply_tool_call_margin(tool_hbox);
	tool_vbox->add_child(tool_hbox);

	RichTextLabel *tool_label = memnew(RichTextLabel);
	// Use descriptive status with Cursor-style emphasis (action bright, details faded)
	String formatted_status = AIChatToolStyling::format_tool_status_with_emphasis(p_descriptive_status);
	tool_label->set_use_bbcode(true);
	tool_label->set_fit_content(true);
	tool_label->set_text(formatted_status);
	tool_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tool_label->set_selection_enabled(false);
	
	// Apply transparent background and sizing
	Color base_color = p_chat_dock->get_theme_color(SNAME("font_color"), SNAME("Editor"));
	Color faded_color = base_color * Color(1.0, 1.0, 1.0, 0.7); // 70% opacity for overall tool text
	tool_label->add_theme_color_override("default_color", faded_color);
	tool_label->add_theme_color_override("font_selected_color", Color(1, 1, 1, 1)); // White when selected
	
	int default_size = p_chat_dock->get_theme_font_size(SNAME("font_size"), SNAME("Label"));
	tool_label->add_theme_font_size_override("normal_font_size", default_size - 1); // Slightly smaller
	tool_hbox->add_child(tool_label);
}

void AIChatToolCallManager::_reposition_streaming_indicator(
	RichTextLabel *p_current_assistant_label,
	VBoxContainer *p_chat_container,
	AIChatDock *p_chat_dock
) {
	if (!p_current_assistant_label || !p_chat_dock) {
		return;
	}

	// Find indicator container from meta (stored when message was created)
	HBoxContainer *indicator_container = nullptr;
	StreamingIndicator *found_indicator = nullptr;
	
	if (p_current_assistant_label->has_meta("indicator_container")) {
		Variant meta = p_current_assistant_label->get_meta("indicator_container");
		indicator_container = Object::cast_to<HBoxContainer>(meta);
		if (indicator_container) {
			// Find the StreamingIndicator inside
			for (int j = 0; j < indicator_container->get_child_count(); j++) {
				StreamingIndicator *indicator = Object::cast_to<StreamingIndicator>(indicator_container->get_child(j));
				if (indicator) {
					found_indicator = indicator;
					break;
				}
			}
			print_line("AI Chat: Found indicator_container from meta");
		}
	}

	// Get message vbox for repositioning
	Control *bubble_panel = Object::cast_to<Control>(p_current_assistant_label->get_parent()->get_parent());
	if (!bubble_panel) return;
	
	VBoxContainer *message_vbox = Object::cast_to<VBoxContainer>(bubble_panel->get_child(0));
	if (!message_vbox) return;

	// Reposition indicator to the end (after all tool calls)
	if (indicator_container && found_indicator) {
		// Remove from current parent if it has one
		if (indicator_container->get_parent() != nullptr) {
			print_line("AI Chat: indicator_container already has parent, removing before repositioning");
			Node *parent = indicator_container->get_parent();
			parent->remove_child(indicator_container);
		}
		
		// Add at the end
		message_vbox->add_child(indicator_container);
		print_line("AI Chat: Added indicator_container AFTER tool calls");
		
		// Only show the indicator if appropriate
		bool should_show_indicator = AIChatStreamingManager::should_show_indicator(
			p_chat_dock->is_waiting_for_response, false, p_chat_dock->pending_tool_tasks
		);
		
		if (should_show_indicator && !found_indicator->is_visible()) {
			print_line("AI Chat: Showing streaming indicator at end (waiting for final response, no tool calls executing)");
			found_indicator->start_animation();
		} else if (!should_show_indicator) {
			if (found_indicator->is_visible()) {
				print_line("AI Chat: Hiding streaming indicator (tool calls executing or not waiting for response)");
				found_indicator->stop_animation();
			}
		}
		print_line("AI Chat: Moved indicator_container to end (after all tool calls)");
	}
}

void AIChatToolCallManager::apply_tool_result_deferred(
	const String &p_tool_call_id,
	const String &p_tool_name,
	const String &p_content,
	const Array &p_tool_results,
	VBoxContainer *p_chat_container,
	AIChatDock *p_chat_dock
) {
	if (!p_chat_container || !p_chat_dock) {
		print_line("AI Chat: apply_tool_result_deferred called with null parameters");
		return;
	}

	// Find the placeholder for this tool call ID
	PanelContainer *placeholder = Object::cast_to<PanelContainer>(
		p_chat_container->find_child("tool_placeholder_" + p_tool_call_id, true, false)
	);

	if (!placeholder) {
		print_line("AI Chat: apply_tool_result_deferred - No placeholder found for tool_call_id: " + p_tool_call_id);
		return;
	}

	// Clear the "loading" text immediately
	while (placeholder->get_child_count() > 0) {
		Node *child = placeholder->get_child(0);
		placeholder->remove_child(child);
		child->queue_free();
	}

	// Create the replacement UI using the main dock's existing logic
	// Note: This calls back to the main dock for now, but could be extracted further
	p_chat_dock->_apply_tool_result_deferred(p_tool_call_id, p_tool_name, p_content, p_tool_results);
}

void AIChatToolCallManager::update_tool_placeholder_status(
	const String &p_tool_id,
	const String &p_tool_name,
	const String &p_status,
	VBoxContainer *p_chat_container,
	AIChatDock *p_chat_dock
) {
	if (!p_chat_container || !p_chat_dock) {
		return;
	}

	// Find the placeholder
	PanelContainer *placeholder = Object::cast_to<PanelContainer>(
		p_chat_container->find_child("tool_placeholder_" + p_tool_id, true, false)
	);

	if (!placeholder) {
		print_line("AI Chat: Warning - no placeholder found for tool " + p_tool_id + " to update status");
		return;
	}

	// Update status through main dock (for now - could be extracted)
	p_chat_dock->_update_tool_placeholder_status(p_tool_id, p_tool_name, p_status);
}

void AIChatToolCallManager::update_tool_placeholder_with_description(
	const String &p_tool_id,
	const String &p_tool_name,
	const String &p_status,
	const String &p_description,
	VBoxContainer *p_chat_container,
	AIChatDock *p_chat_dock
) {
	if (!p_chat_container || !p_chat_dock) {
		return;
	}

	// Find the placeholder
	PanelContainer *placeholder = Object::cast_to<PanelContainer>(
		p_chat_container->find_child("tool_placeholder_" + p_tool_id, true, false)
	);

	if (!placeholder) {
		print_line("AI Chat: Warning - no placeholder found for tool " + p_tool_id + " to update description");
		return;
	}

	// Update description through main dock (for now - could be extracted)
	p_chat_dock->_update_tool_placeholder_with_description(p_tool_id, p_tool_name, p_status, p_description);
}

void AIChatToolCallManager::create_backend_tool_placeholder(
	const String &p_tool_id,
	const String &p_tool_name,
	VBoxContainer *p_chat_container,
	AIChatDock *p_chat_dock
) {
	if (!p_chat_container || !p_chat_dock) {
		return;
	}

	// Create a new message bubble for backend tools
	PanelContainer *bubble_panel = memnew(PanelContainer);
	bubble_panel->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	
	Ref<StyleBoxFlat> bubble_style = memnew(StyleBoxFlat);
	bubble_style->set_bg_color(Color(0, 0, 0, 0)); // Transparent background
	bubble_style->set_content_margin_all(15);
	bubble_style->set_border_width_all(0); // No border
	bubble_style->set_border_color(Color(0, 0, 0, 0)); // Transparent border
	bubble_style->set_corner_radius_all(10);
	bubble_panel->add_theme_style_override("panel", bubble_style);
	
	VBoxContainer *message_vbox = memnew(VBoxContainer);
	bubble_panel->add_child(message_vbox);
	
	// Create tool placeholder
	PanelContainer *placeholder = memnew(PanelContainer);
	placeholder->set_name("tool_placeholder_" + p_tool_id);
	message_vbox->add_child(placeholder);

	Ref<StyleBoxFlat> placeholder_style = memnew(StyleBoxFlat);
	placeholder_style->set_bg_color(Color(0, 0, 0, 0)); // Transparent
	placeholder_style->set_content_margin_all(0); // No padding
	placeholder_style->set_border_width_all(0);
	placeholder_style->set_border_color(Color(0, 0, 0, 0)); // Transparent
	placeholder_style->set_corner_radius_all(CORNER_RADIUS);
	placeholder->add_theme_style_override("panel", placeholder_style);

	HBoxContainer *tool_hbox = memnew(HBoxContainer);
	placeholder->add_child(tool_hbox);

	Label *tool_label = memnew(Label);
	String executing_message = p_chat_dock->_generate_executing_tool_message(p_tool_name, "");
	tool_label->set_text(executing_message);
	// Use monochromatic styling for executing tools
	AIChatToolStyling::style_executing_tool_label(tool_label, p_chat_dock);
	tool_hbox->add_child(tool_label);
	
	p_chat_container->add_child(bubble_panel);
	p_chat_dock->call_deferred("_scroll_to_bottom");
}

void AIChatToolCallManager::apply_simplified_tool_result(
	const String &p_tool_call_id,
	const String &p_tool_name,
	const String &p_content,
	VBoxContainer *p_chat_container,
	AIChatDock *p_chat_dock
) {
	if (!p_chat_container || !p_chat_dock) {
		return;
	}

	// Find the placeholder
	PanelContainer *placeholder = Object::cast_to<PanelContainer>(
		p_chat_container->find_child("tool_placeholder_" + p_tool_call_id, true, false)
	);

	if (!placeholder) {
		print_line("AI Chat: apply_simplified_tool_result - No placeholder found for: " + p_tool_call_id);
		return;
	}

	// Delegate to main dock implementation for now
	p_chat_dock->_apply_simplified_tool_result(p_tool_call_id, p_tool_name, p_content);
}

void AIChatToolCallManager::expand_simplified_tool_result(
	const String &p_tool_call_id,
	const String &p_tool_name,
	const String &p_content,
	PanelContainer *p_placeholder,
	AIChatDock *p_chat_dock
) {
	if (!p_placeholder || !p_chat_dock) {
		return;
	}

	// Delegate to main dock implementation for now
	p_chat_dock->_expand_simplified_tool_result(p_tool_call_id, p_tool_name, p_content, p_placeholder);
}

void AIChatToolCallManager::update_tool_placeholder_with_result(
	const Dictionary &p_tool_message_data,
	VBoxContainer *p_chat_container,
	AIChatDock *p_chat_dock
) {
	if (!p_chat_container || !p_chat_dock) {
		return;
	}

	// Extract tool call ID from message data
	String tool_call_id = p_tool_message_data.get("tool_call_id", "");
	if (tool_call_id.is_empty()) {
		return;
	}

	// Find the placeholder
	PanelContainer *placeholder = Object::cast_to<PanelContainer>(
		p_chat_container->find_child("tool_placeholder_" + tool_call_id, true, false)
	);

	if (!placeholder) {
		print_line("AI Chat: Warning - no placeholder found for tool_call_id: " + tool_call_id);
		return;
	}

	// Delegate to main dock implementation for now
	// Note: This would be fully extracted in a complete implementation
	// Convert Dictionary to ChatMessage for compatibility
	AIChatDock::ChatMessage msg;
	msg.tool_call_id = tool_call_id;
	msg.name = p_tool_message_data.get("name", "");
	msg.content = p_tool_message_data.get("content", "");
	msg.tool_results = p_tool_message_data.get("tool_results", Array());
	
	p_chat_dock->_update_tool_placeholder_with_result(msg);
}
