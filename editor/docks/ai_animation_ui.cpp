/* AI Animation UI Implementation */

#include "ai_animation_ui.h"
#include "ai_chat_dock.h"
#include "scene/gui/separator.h"
#include "scene/resources/style_box_flat.h"
#include "scene/main/http_request.h"
#include "core/io/image.h"
#include "core/io/json.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/core_bind.h"
#include "core/os/os.h"
#include "editor/settings/editor_settings.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/animated_texture.h"
#include "editor/editor_node.h"
#include "editor/file_system/editor_file_system.h"
#include "core/config/project_settings.h"

PanelContainer *AIAnimationUI::create_animation_job_panel(
	const String &p_job_id,
	const String &p_user_request,
	const Dictionary &p_result,
	VBoxContainer *p_parent_container,
	AIChatDock *p_dock
) {
	// Create beautiful panel for animation job tracking
	PanelContainer *job_panel = memnew(PanelContainer);
	job_panel->set_name("anim_job_" + p_job_id);
	
	// Modern styling
	Ref<StyleBoxFlat> panel_style = memnew(StyleBoxFlat);
	panel_style->set_bg_color(Color(0.15, 0.15, 0.2, 0.95)); // Dark blue-tinted background
	panel_style->set_border_width_all(2);
	panel_style->set_border_color(Color(0.4, 0.6, 0.9, 0.8)); // Blue accent border
	panel_style->set_corner_radius_all(8);
	panel_style->set_content_margin_all(16);
	job_panel->add_theme_style_override("panel", panel_style);
	
	VBoxContainer *content = memnew(VBoxContainer);
	content->add_theme_constant_override("separation", 12);
	job_panel->add_child(content);
	
	// Header with icon and title
	HBoxContainer *header = memnew(HBoxContainer);
	content->add_child(header);
	
	Label *icon = memnew(Label);
	if (p_dock) {
		icon->add_theme_icon_override("icon", p_dock->get_theme_icon("AnimationPlayer", "EditorIcons"));
	}
	header->add_child(icon);
	
	Label *title = memnew(Label);
	title->set_text("Creating Sprite Animations");
	if (p_dock) {
		title->add_theme_font_override("font", p_dock->get_theme_font("bold", "EditorFonts"));
		title->add_theme_color_override("font_color", Color(0.4, 0.6, 0.9));
	}
	title->add_theme_font_size_override("font_size", 16);
	header->add_child(title);
	
	// User request preview
	Label *request_label = memnew(Label);
	String request_preview = p_user_request;
	if (request_preview.length() > 80) {
		request_preview = request_preview.substr(0, 77) + "...";
	}
	request_label->set_text(request_preview);
	request_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	if (p_dock) {
		request_label->add_theme_color_override("font_color", Color(0.9, 0.9, 0.9, 0.8));
	}
	content->add_child(request_label);
	
	// Status label (will be updated by tracker)
	Label *status_label = memnew(Label);
	status_label->set_name("anim_status_" + p_job_id);
	status_label->set_text("Generating... (2-4 minutes)");
	if (p_dock) {
		status_label->add_theme_color_override("font_color", Color(0.9, 0.8, 0.3));
	}
	content->add_child(status_label);
	
	// Progress bar
	ProgressBar *progress_bar = memnew(ProgressBar);
	progress_bar->set_min(0);
	progress_bar->set_max(100);
	progress_bar->set_value(0);
	progress_bar->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	progress_bar->set_custom_minimum_size(Size2(0, 24));
	content->add_child(progress_bar);
	
	// Placeholder for GIF preview (will be populated when thumbnails are ready)
	VBoxContainer *preview_container = memnew(VBoxContainer);
	preview_container->set_name("preview_container");
	preview_container->set_visible(false); // Hidden until we have data
	content->add_child(preview_container);
	
	// Polling indicator (shows that system is actively checking)
	HBoxContainer *polling_row = memnew(HBoxContainer);
	content->add_child(polling_row);
	
	Label *polling_icon = memnew(Label);
	polling_icon->set_text("[*]");
	polling_row->add_child(polling_icon);
	
	Label *polling_label = memnew(Label);
	polling_label->set_text("Auto-checking progress...");
	polling_label->set_name("polling_indicator");
	polling_label->add_theme_font_size_override("font_size", 11);
	if (p_dock) {
		polling_label->add_theme_color_override("font_color", Color(0.5, 0.7, 1.0, 0.9));
	}
	polling_row->add_child(polling_label);
	
	// Job ID (small, subtle)
	Label *job_id_label = memnew(Label);
	job_id_label->set_text("Job ID: " + p_job_id);
	job_id_label->add_theme_font_size_override("font_size", 10);
	if (p_dock) {
		job_id_label->add_theme_color_override("font_color", Color(0.7, 0.7, 0.7, 0.5));
	}
	content->add_child(job_id_label);
	
	p_parent_container->add_child(job_panel);
	
	print_line("AIAnimationUI: Created job panel for " + p_job_id.substr(0, 8));
	
	return job_panel;
}

void AIAnimationUI::update_animation_job_panel(
	PanelContainer *p_panel,
	const String &p_status,
	const Dictionary &p_progress
) {
	if (!p_panel || !p_panel->is_inside_tree()) {
		print_line("AIAnimationUI: Panel not in tree");
		return;
	}
	
	print_line("AIAnimationUI: Status=" + p_status);
	
	// Find and update status label
	Node *content_node = p_panel->get_child(0);
	VBoxContainer *content = Object::cast_to<VBoxContainer>(content_node);
	if (!content) {
		return;
	}
	
	for (int i = 0; i < content->get_child_count(); i++) {
		Node *child = content->get_child(i);
		
		// Update status label
		Label *status_label = Object::cast_to<Label>(child);
		if (status_label && String(status_label->get_name()).begins_with("anim_status_")) {
			String message = format_progress_message(p_progress);
			status_label->set_text(message);
			
			print_line("   Updated status label: " + message);
			
			// Update color based on status
			if (p_status == "completed") {
				status_label->add_theme_color_override("font_color", Color(0.3, 0.9, 0.3));
			} else if (p_status == "failed") {
				status_label->add_theme_color_override("font_color", Color(0.9, 0.3, 0.3));
			} else {
				status_label->add_theme_color_override("font_color", Color(0.9, 0.8, 0.3));
			}
		}
		
		// Update progress bar
		ProgressBar *progress_bar = Object::cast_to<ProgressBar>(child);
		if (progress_bar) {
			int current_level = p_progress.get("current_level", 0);
			int total_levels = p_progress.get("levels", 1);
			
			if (total_levels > 0) {
				float progress_pct = (float)(current_level + 1) / (float)total_levels * 100.0f;
				progress_bar->set_value(progress_pct);
				print_line("   Updated progress bar: " + String::num(progress_pct, 1) + "%");
			}
		}
		
		// Update polling indicator to show activity
		Label *polling_label = Object::cast_to<Label>(child);
		if (polling_label && String(polling_label->get_name()) == "polling_indicator") {
			// Show last poll time
			polling_label->set_text("Checking... next in 15s");
		}
	}
}

void AIAnimationUI::create_animation_result_panel(
	VBoxContainer *p_parent_container,
	const Dictionary &p_completion_data,
	AIChatDock *p_dock
) {
	// Create beautiful completion panel with GIF previews
	PanelContainer *result_panel = memnew(PanelContainer);
	
	Ref<StyleBoxFlat> result_style = memnew(StyleBoxFlat);
	result_style->set_bg_color(Color(0.1, 0.3, 0.15, 0.9)); // Green-tinted success
	result_style->set_border_width_all(2);
	result_style->set_border_color(Color(0.3, 0.9, 0.4, 0.8));
	result_style->set_corner_radius_all(8);
	result_style->set_content_margin_all(16);
	result_panel->add_theme_style_override("panel", result_style);
	
	VBoxContainer *content = memnew(VBoxContainer);
	content->add_theme_constant_override("separation", 12);
	result_panel->add_child(content);
	
	// Success header
	HBoxContainer *header = memnew(HBoxContainer);
	content->add_child(header);
	
	Label *success_icon = memnew(Label);
	if (p_dock) {
		success_icon->add_theme_icon_override("icon", p_dock->get_theme_icon("StatusSuccess", "EditorIcons"));
	}
	header->add_child(success_icon);
	
	Label *success_title = memnew(Label);
	success_title->set_text("Animations Ready");
	if (p_dock) {
		success_title->add_theme_font_override("font", p_dock->get_theme_font("bold", "EditorFonts"));
		success_title->add_theme_color_override("font_color", Color(0.3, 0.9, 0.4));
	}
	success_title->add_theme_font_size_override("font_size", 16);
	header->add_child(success_title);
	
	// Animation count and project ID (check multiple fields for backwards compatibility)
	Array completed_anims = p_completion_data.get("completed_animations", Array());
	String supabase_project_id = p_completion_data.get("supabase_project_id", "");
	if (supabase_project_id.is_empty()) {
		supabase_project_id = p_completion_data.get("project_id", "");
	}
	
	// Debug: print available fields
	print_line("Animation result panel - project_id: " + supabase_project_id + ", anims: " + itos(completed_anims.size()));
	
	Label *count_label = memnew(Label);
	count_label->set_text(itos(completed_anims.size()) + " animation(s) generated successfully");
	if (p_dock) {
		count_label->add_theme_color_override("font_color", Color(0.9, 0.9, 0.9, 0.9));
	}
	content->add_child(count_label);
	
	// Get thumbnail URLs if available (may include frame_urls for animated thumbnails)
	Array thumbnail_urls = p_completion_data.get("thumbnail_urls", Array());
	HashMap<String, Dictionary> name_to_thumb_info;
	for (int i = 0; i < thumbnail_urls.size(); i++) {
		Dictionary thumb_info = thumbnail_urls[i];
		String name = thumb_info.get("name", "");
		if (!name.is_empty()) {
			name_to_thumb_info[name] = thumb_info;
		}
	}
	
	// List animations - with animated thumbnails if available
	for (int i = 0; i < completed_anims.size(); i++) {
		String anim_name = completed_anims[i];
		
		HBoxContainer *anim_row = memnew(HBoxContainer);
		anim_row->add_theme_constant_override("separation", 8);
		content->add_child(anim_row);
		
		// Check if we have thumbnail info for this animation
		if (name_to_thumb_info.has(anim_name)) {
			Dictionary thumb_info = name_to_thumb_info[anim_name];
			Array frame_urls = thumb_info.get("frame_urls", Array());
			String static_url = thumb_info.get("url", "");
			
			// Create TextureRect for the thumbnail
			TextureRect *thumb_rect = memnew(TextureRect);
			thumb_rect->set_custom_minimum_size(Size2(64, 64));
			thumb_rect->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
			thumb_rect->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
			thumb_rect->set_stretch_mode(TextureRect::STRETCH_KEEP_ASPECT_CENTERED);
			thumb_rect->set_expand_mode(TextureRect::EXPAND_IGNORE_SIZE);  // Keep at 64x64
			thumb_rect->set_name("thumb_" + anim_name);
			anim_row->add_child(thumb_rect);
			
			if (p_dock && !frame_urls.is_empty()) {
				// Load animated thumbnail (multiple frames -> AnimatedTexture)
				print_line("Loading animated thumbnail for " + anim_name + " with " + itos(frame_urls.size()) + " frames");
				_start_animated_thumbnail_load(thumb_rect, frame_urls, p_dock);
			} else if (p_dock && !static_url.is_empty()) {
				// Load static thumbnail
				HTTPRequest *http = memnew(HTTPRequest);
				p_dock->add_child(http);
				http->set_name("thumb_loader_" + anim_name);
				http->set_download_file("");
				http->set_use_threads(true);
				http->connect("request_completed", callable_mp_static(&AIAnimationUI::_on_thumbnail_loaded).bind(thumb_rect, http));
				http->request(static_url);
			}
		} else {
			// No thumbnail - show bullet
			Label *bullet = memnew(Label);
			bullet->set_text("  - ");
			if (p_dock) {
				bullet->add_theme_color_override("font_color", Color(0.5, 0.8, 0.5));
			}
			anim_row->add_child(bullet);
		}
		
		Label *anim_label = memnew(Label);
		anim_label->set_text(anim_name);
		anim_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		if (p_dock) {
			anim_label->add_theme_font_override("font", p_dock->get_theme_font("main", "EditorFonts"));
		}
		anim_row->add_child(anim_label);
		
		// Export button (only if we have a project ID)
		if (!supabase_project_id.is_empty() && p_dock) {
			Button *export_btn = memnew(Button);
			export_btn->set_text("Export");
			export_btn->set_flat(true);
			export_btn->set_custom_minimum_size(Size2(50, 22));
			export_btn->set_tooltip_text("Export as sprite sheet, GIF, or frames");
			
			// Style the button to match the panel aesthetic
			Ref<StyleBoxFlat> btn_normal = memnew(StyleBoxFlat);
			btn_normal->set_bg_color(Color(0.2, 0.5, 0.3, 0.8));  // Green tint
			btn_normal->set_corner_radius_all(4);
			btn_normal->set_content_margin(SIDE_LEFT, 8);
			btn_normal->set_content_margin(SIDE_RIGHT, 8);
			btn_normal->set_content_margin(SIDE_TOP, 2);
			btn_normal->set_content_margin(SIDE_BOTTOM, 2);
			export_btn->add_theme_style_override("normal", btn_normal);
			
			Ref<StyleBoxFlat> btn_hover = memnew(StyleBoxFlat);
			btn_hover->set_bg_color(Color(0.25, 0.6, 0.35, 0.9));  // Lighter on hover
			btn_hover->set_corner_radius_all(4);
			btn_hover->set_content_margin(SIDE_LEFT, 8);
			btn_hover->set_content_margin(SIDE_RIGHT, 8);
			btn_hover->set_content_margin(SIDE_TOP, 2);
			btn_hover->set_content_margin(SIDE_BOTTOM, 2);
			export_btn->add_theme_style_override("hover", btn_hover);
			
			Ref<StyleBoxFlat> btn_pressed = memnew(StyleBoxFlat);
			btn_pressed->set_bg_color(Color(0.15, 0.4, 0.25, 1.0));  // Darker on press
			btn_pressed->set_corner_radius_all(4);
			btn_pressed->set_content_margin(SIDE_LEFT, 8);
			btn_pressed->set_content_margin(SIDE_RIGHT, 8);
			btn_pressed->set_content_margin(SIDE_TOP, 2);
			btn_pressed->set_content_margin(SIDE_BOTTOM, 2);
			export_btn->add_theme_style_override("pressed", btn_pressed);
			
			export_btn->add_theme_font_size_override("font_size", 11);
			export_btn->add_theme_color_override("font_color", Color(0.95, 0.95, 0.95));
			export_btn->add_theme_color_override("font_hover_color", Color(1.0, 1.0, 1.0));
			
			// Store animation info in button metadata for the callback
			export_btn->set_meta("animation_id", anim_name);
			export_btn->set_meta("project_id", supabase_project_id);
			
			export_btn->connect("pressed", callable_mp(p_dock, &AIChatDock::_on_export_animation_pressed).bind(export_btn));
			anim_row->add_child(export_btn);
		}
	}
	
	p_parent_container->add_child(result_panel);
}

void AIAnimationUI::create_numbered_animation_list(
	VBoxContainer *p_parent_container,
	const Array &p_animations_list,
	int p_total_count,
	AIChatDock *p_dock
) {
	// Create beautiful numbered list of animations
	PanelContainer *list_panel = memnew(PanelContainer);
	
	Ref<StyleBoxFlat> list_style = memnew(StyleBoxFlat);
	list_style->set_bg_color(Color(0.12, 0.12, 0.15, 0.95));
	list_style->set_border_width_all(1);
	list_style->set_border_color(Color(0.3, 0.3, 0.35, 0.8));
	list_style->set_corner_radius_all(6);
	list_style->set_content_margin_all(12);
	list_panel->add_theme_style_override("panel", list_style);
	
	VBoxContainer *content = memnew(VBoxContainer);
	content->add_theme_constant_override("separation", 6);
	list_panel->add_child(content);
	
	// Header
	HBoxContainer *header = memnew(HBoxContainer);
	content->add_child(header);
	
	Label *icon = memnew(Label);
	if (p_dock) {
		icon->add_theme_icon_override("icon", p_dock->get_theme_icon("AnimationPlayer", "EditorIcons"));
	}
	header->add_child(icon);
	
	Label *title = memnew(Label);
	title->set_text("Your Sprite Animations (" + itos(p_total_count) + ")");
	if (p_dock) {
		title->add_theme_font_override("font", p_dock->get_theme_font("bold", "EditorFonts"));
		title->add_theme_color_override("font_color", Color(0.5, 0.7, 1.0));
	}
	header->add_child(title);
	
	content->add_child(memnew(HSeparator));
	
	// Animation list
	for (int i = 0; i < p_animations_list.size(); i++) {
		String anim_entry = p_animations_list[i];
		
		HBoxContainer *row = memnew(HBoxContainer);
		content->add_child(row);
		
		Label *number = memnew(Label);
		// Extract number from entry (e.g., "#1: idle_knight...")
		int colon_pos = anim_entry.find(":");
		String number_part = colon_pos > 0 ? anim_entry.substr(0, colon_pos + 1) : "";
		number->set_text(number_part);
		if (p_dock) {
			number->add_theme_font_override("font", p_dock->get_theme_font("bold", "EditorFonts"));
			number->add_theme_color_override("font_color", Color(0.5, 0.7, 1.0));
		}
		number->set_custom_minimum_size(Size2(40, 0));
		row->add_child(number);
		
		Label *anim_label = memnew(Label);
		String desc = colon_pos > 0 ? anim_entry.substr(colon_pos + 1).strip_edges() : anim_entry;
		anim_label->set_text(desc);
		anim_label->set_h_size_flags(Control::SIZE_EXPAND_FILL);
		if (p_dock) {
			anim_label->add_theme_color_override("font_color", Color(0.95, 0.95, 0.95));
		}
		row->add_child(anim_label);
	}
	
	// Usage hint
	content->add_child(memnew(HSeparator));
	
	Label *hint = memnew(Label);
	hint->set_text("💡 Reference by number: \"Download #1\" or \"Make #2 faster\"");
	hint->add_theme_font_size_override("font_size", 11);
	hint->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	if (p_dock) {
		hint->add_theme_color_override("font_color", Color(0.7, 0.7, 0.8, 0.9));
	}
	content->add_child(hint);
	
	p_parent_container->add_child(list_panel);
}

void AIAnimationUI::download_and_display_animation_gif(
	const String &p_thumbnail_url,
	TextureRect *p_texture_rect,
	Label *p_status_label
) {
	// This would download GIF from Supabase and display
	// For now, just show a placeholder
	if (p_status_label) {
		p_status_label->set_text("GIF preview: " + p_thumbnail_url);
	}
	
	// TODO: Implement actual GIF download and display
	// Would need HTTPRequest to fetch from Supabase storage
	// Then decode GIF and create animated texture
}

String AIAnimationUI::format_time_estimate(int p_seconds_elapsed, const String &p_status) {
	if (p_status == "completed") {
		return "Completed in " + itos(p_seconds_elapsed) + "s";
	} else if (p_status == "failed") {
		return "Failed after " + itos(p_seconds_elapsed) + "s";
	}
	
	// Estimate remaining time (rough)
	int avg_time = 240; // 4 minutes average
	int remaining = avg_time - p_seconds_elapsed;
	if (remaining < 0) remaining = 30; // At least 30s
	
	int minutes = remaining / 60;
	int seconds = remaining % 60;
	
	String estimate = "~";
	if (minutes > 0) {
		estimate += itos(minutes) + "m ";
	}
	estimate += itos(seconds) + "s remaining";
	
	return estimate;
}

Label *AIAnimationUI::create_status_badge(const String &p_status, Control *p_parent) {
	Label *badge = memnew(Label);
	
	if (p_status == "pending" || p_status == "starting") {
		badge->set_text("PENDING");
		badge->add_theme_color_override("font_color", Color(0.9, 0.8, 0.3));
	} else if (p_status == "processing" || p_status == "generating") {
		badge->set_text("GENERATING");
		badge->add_theme_color_override("font_color", Color(0.4, 0.6, 0.9));
	} else if (p_status == "completed") {
		badge->set_text("DONE");
		badge->add_theme_color_override("font_color", Color(0.3, 0.9, 0.4));
	} else if (p_status == "failed") {
		badge->set_text("FAILED");
		badge->add_theme_color_override("font_color", Color(0.9, 0.3, 0.3));
	} else {
		badge->set_text(p_status.to_upper());
		badge->add_theme_color_override("font_color", Color(0.7, 0.7, 0.7));
	}
	
	badge->add_theme_font_size_override("font_size", 12);
	
	return badge;
}

String AIAnimationUI::format_progress_message(const Dictionary &p_progress) {
	// Simple message - just show stage or "Generating..."
	String stage = p_progress.get("stage", "");
	String msg = p_progress.get("message", "");
	
	if (stage == "completed") {
		return "Done";
	} else if (stage == "failed") {
		return "Failed";
	} else if (!msg.is_empty()) {
		return msg;
	} else if (!stage.is_empty()) {
		return stage.replace("_", " ").capitalize();
	}
	
	return "Generating...";
}

void AIAnimationUI::create_animation_failure_panel(
	VBoxContainer *p_parent_container,
	const String &p_job_id,
	const String &p_error,
	AIChatDock *p_dock
) {
	// Create failure panel with clear error display
	PanelContainer *failure_panel = memnew(PanelContainer);
	
	Ref<StyleBoxFlat> failure_style = memnew(StyleBoxFlat);
	failure_style->set_bg_color(Color(0.25, 0.1, 0.1, 0.95)); // Red-tinted error
	failure_style->set_border_width_all(2);
	failure_style->set_border_color(Color(0.9, 0.3, 0.3, 0.9)); // Red border
	failure_style->set_corner_radius_all(8);
	failure_style->set_content_margin_all(16);
	failure_panel->add_theme_style_override("panel", failure_style);
	
	VBoxContainer *content = memnew(VBoxContainer);
	content->add_theme_constant_override("separation", 12);
	failure_panel->add_child(content);
	
	// Error header
	HBoxContainer *header = memnew(HBoxContainer);
	content->add_child(header);
	
	Label *error_icon = memnew(Label);
	error_icon->set_text("❌");
	error_icon->add_theme_font_size_override("font_size", 18);
	header->add_child(error_icon);
	
	Label *error_title = memnew(Label);
	error_title->set_text("Animation Generation Failed");
	if (p_dock) {
		error_title->add_theme_font_override("font", p_dock->get_theme_font("bold", "EditorFonts"));
	}
	error_title->add_theme_color_override("font_color", Color(0.95, 0.4, 0.4));
	error_title->add_theme_font_size_override("font_size", 16);
	header->add_child(error_title);
	
	// Error message
	Label *error_label = memnew(Label);
	error_label->set_text(p_error);
	error_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	error_label->add_theme_color_override("font_color", Color(0.95, 0.85, 0.85));
	content->add_child(error_label);
	
	// Helpful hint based on error type
	String hint_text = "";
	if (p_error.contains("content safety") || p_error.contains("safety filter")) {
		hint_text = "💡 Tip: Try rephrasing your request. Avoid explicit violence, weapons, or harmful content.";
	} else if (p_error.contains("timeout")) {
		hint_text = "💡 Tip: The generation took too long. Try a simpler animation or retry.";
	} else if (p_error.contains("connection") || p_error.contains("network")) {
		hint_text = "💡 Tip: Check your internet connection and try again.";
	} else {
		hint_text = "💡 Tip: Try again or describe your animation differently.";
	}
	
	Label *hint_label = memnew(Label);
	hint_label->set_text(hint_text);
	hint_label->set_autowrap_mode(TextServer::AUTOWRAP_WORD_SMART);
	hint_label->add_theme_font_size_override("font_size", 12);
	hint_label->add_theme_color_override("font_color", Color(0.8, 0.8, 0.6, 0.9));
	content->add_child(hint_label);
	
	// Job ID (small, subtle)
	Label *job_id_label = memnew(Label);
	job_id_label->set_text("Job ID: " + p_job_id);
	job_id_label->add_theme_font_size_override("font_size", 10);
	job_id_label->add_theme_color_override("font_color", Color(0.6, 0.5, 0.5, 0.6));
	content->add_child(job_id_label);
	
	p_parent_container->add_child(failure_panel);
	
	print_line("AIAnimationUI: Created failure panel for " + p_job_id.substr(0, 8) + ": " + p_error);
}

void AIAnimationUI::_on_thumbnail_loaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, TextureRect *p_rect, HTTPRequest *p_http) {
	// Clean up HTTP request
	if (p_http && p_http->get_parent()) {
		p_http->queue_free();
	}
	
	if (!p_rect || !p_rect->is_inside_tree()) {
		return;
	}
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
		print_line("Thumbnail load failed: result=" + itos(p_result) + " code=" + itos(p_code));
		return;
	}
	
	if (p_body.is_empty()) {
		print_line("Thumbnail body is empty");
		return;
	}
	
	// Try to load the image from the body (PNG/JPG/WebP - GIF not directly supported)
	Ref<Image> img;
	img.instantiate();
	
	// Try PNG first (most common for thumbnails)
	Error err = img->load_png_from_buffer(p_body);
	if (err != OK) {
		// Try JPG
		err = img->load_jpg_from_buffer(p_body);
	}
	if (err != OK) {
		// Try WebP
		err = img->load_webp_from_buffer(p_body);
	}
	
	if (err != OK) {
		// For GIFs, we can't load directly - they would need AnimatedTexture
		// For now, just log and skip
		print_line("Failed to decode thumbnail image (format may be GIF which requires special handling)");
		return;
	}
	
	// Create texture from image
	Ref<ImageTexture> tex = ImageTexture::create_from_image(img);
	if (tex.is_valid()) {
		p_rect->set_texture(tex);
		print_line("Thumbnail loaded successfully: " + itos(img->get_width()) + "x" + itos(img->get_height()));
	}
}

// Static storage for animated thumbnail loading state
// We use a simple approach: store loaded textures in meta data on the TextureRect
static void _on_animated_frame_loaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, TextureRect *p_rect, int p_frame_index, int p_total_frames, HTTPRequest *p_http) {
	// Clean up HTTP request
	if (p_http && p_http->get_parent()) {
		p_http->queue_free();
	}
	
	if (!p_rect || !p_rect->is_inside_tree()) {
		return;
	}
	
	// Load the frame
	Ref<ImageTexture> tex;
	if (p_result == HTTPRequest::RESULT_SUCCESS && p_code == 200 && !p_body.is_empty()) {
		Ref<Image> img;
		img.instantiate();
		
		Error err = img->load_png_from_buffer(p_body);
		if (err != OK) {
			err = img->load_jpg_from_buffer(p_body);
		}
		if (err != OK) {
			err = img->load_webp_from_buffer(p_body);
		}
		
		if (err == OK) {
			tex = ImageTexture::create_from_image(img);
		}
	}
	
	// Store frame in meta data
	String frame_key = "anim_frame_" + itos(p_frame_index);
	p_rect->set_meta(frame_key, tex);
	
	// Increment loaded count
	int loaded = p_rect->get_meta("anim_loaded_count", 0);
	loaded++;
	p_rect->set_meta("anim_loaded_count", loaded);
	
	// Check if all frames loaded
	if (loaded >= p_total_frames) {
		// Collect all frames IN ORDER
		Vector<Ref<ImageTexture>> frames;
		int missing = 0;
		for (int i = 0; i < p_total_frames; i++) {
			String key = "anim_frame_" + itos(i);
			Ref<ImageTexture> frame_tex = p_rect->get_meta(key, Ref<ImageTexture>());
			if (frame_tex.is_valid()) {
				frames.push_back(frame_tex);
			} else {
				missing++;
				print_line("AnimatedThumbnail: Frame " + itos(i) + " failed to load");
			}
		}
		
		if (missing > 0) {
			print_line("AnimatedThumbnail: " + itos(missing) + " frames missing");
		}
		
		if (frames.size() > 0) {
			// Create AnimatedTexture
			Ref<AnimatedTexture> anim_tex;
			anim_tex.instantiate();
			anim_tex->set_frames(frames.size());
			
			// Set frame textures and durations
			for (int i = 0; i < frames.size(); i++) {
				anim_tex->set_frame_texture(i, frames[i]);
				anim_tex->set_frame_duration(i, 0.12);  // 120ms per frame = ~8 FPS (smoother)
			}
			
			// Configure animation playback
			anim_tex->set_pause(false);  // Ensure animation plays
			anim_tex->set_one_shot(false);  // Loop continuously
			anim_tex->set_speed_scale(1.0);  // Normal speed
			
			p_rect->set_texture(anim_tex);
			print_line("AnimatedThumbnail: Created with " + itos(frames.size()) + " frames (looping)");
		}
		
		// Clean up meta data
		for (int i = 0; i < p_total_frames; i++) {
			String key = "anim_frame_" + itos(i);
			p_rect->remove_meta(key);
		}
		p_rect->remove_meta("anim_loaded_count");
	}
}

void AIAnimationUI::_start_animated_thumbnail_load(TextureRect *p_rect, const Array &p_frame_urls, AIChatDock *p_dock) {
	if (p_frame_urls.is_empty() || !p_dock || !p_rect) {
		return;
	}
	
	// Limit to first 8 frames for performance (128x128 images)
	int frame_count = MIN((int)p_frame_urls.size(), 8);
	
	// Initialize tracking
	p_rect->set_meta("anim_loaded_count", 0);
	
	// Start loading each frame (only valid URLs)
	int valid_urls = 0;
	for (int i = 0; i < frame_count; i++) {
		String url = p_frame_urls[i];
		
		// Validate URL - must start with http:// or https://
		if (url.is_empty() || (!url.begins_with("http://") && !url.begins_with("https://"))) {
			print_line("Skipping invalid frame URL " + itos(i) + ": " + url);
			continue;
		}
		
		valid_urls++;
		
		HTTPRequest *http = memnew(HTTPRequest);
		p_dock->add_child(http);
		http->set_name("frame_loader_" + itos(i));
		http->set_download_file("");
		http->set_use_threads(true);
		
		// Connect with frame index
		http->connect("request_completed", callable_mp_static(&_on_animated_frame_loaded).bind(p_rect, i, frame_count, http));
		
		Error err = http->request(url);
		if (err != OK) {
			print_line("Failed to start frame request " + itos(i) + ": " + url);
		}
	}
	
	if (valid_urls == 0) {
		print_line("No valid frame URLs found for animated thumbnail");
	}
}

// ==================== Auto-Export ====================

String AIAnimationUI::_get_api_base_url() {
	// Use the Flask proxy URL (same as the main backend)
	// This matches AIChatDock::_get_api_base_url() exactly
	String base_url;
	String is_dev = OS::get_singleton()->get_environment("IS_DEV");
	if (is_dev.is_empty()) {
		is_dev = OS::get_singleton()->get_environment("DEV_MODE");
	}
	if (!is_dev.is_empty() && is_dev.to_lower() == "true") {
		base_url = "http://127.0.0.1:5050";
	} else {
		base_url = "https://api.orcaengine.ai";
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
	
	return base_url;
}

void AIAnimationUI::trigger_auto_export(
	const String &p_anim_id,
	const String &p_project_id,
	const String &p_save_path,
	int p_resolution,
	const String &p_format,
	Control *p_parent,
	const String &p_template_type,
	const String &p_resource_name,
	int p_fps
) {
	print_line("AUTO_EXPORT: Starting export for " + p_anim_id + " to " + p_save_path + " (format: " + p_format + ")");
	
	if (!p_parent) {
		print_line("AUTO_EXPORT: No parent node for HTTP request");
		return;
	}
	
	// Create HTTP request
	HTTPRequest *export_req = memnew(HTTPRequest);
	p_parent->add_child(export_req);
	export_req->set_use_threads(true);
	export_req->set_meta("save_path", p_save_path);
	export_req->set_meta("format", p_format);
	
	String url;
	Dictionary body;
	body["project_id"] = p_project_id;
	body["resolution"] = p_resolution;
	
	// Choose endpoint based on format
	// NOTE: URLs use /animation/ prefix because they go through Flask proxy on port 5050
	// which forwards /animation/* requests to the animation server on port 8001
	if (p_format == "godot_template") {
		// Use the Godot template export endpoint
		url = _get_api_base_url() + "/animation/export/godot_template";
		
		// For template export, don't filter by animation_ids - export ALL animations in the project
		// The backend will include all animations when animation_ids is empty/not provided
		// This creates a combined sprite sheet with all animations as rows
		body["template_type"] = p_template_type.is_empty() ? "character" : p_template_type;
		body["resource_name"] = p_resource_name.is_empty() ? "sprite" : p_resource_name;
		body["fps"] = p_fps > 0 ? p_fps : 10;
		
		export_req->set_meta("is_template", true);
		export_req->connect("request_completed", callable_mp_static(&_on_template_export_completed).bind(export_req));
	} else {
		// Use simple export endpoint
		url = _get_api_base_url() + "/animation/export";
		body["animation_id"] = p_anim_id;
		body["format"] = p_format;
		
		export_req->set_meta("is_template", false);
		export_req->connect("request_completed", callable_mp_static(&_on_auto_export_completed).bind(export_req));
	}
	
	String json_body = JSON::stringify(body);
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	Error err = export_req->request(url, headers, HTTPClient::METHOD_POST, json_body);
	if (err != OK) {
		print_line("AUTO_EXPORT: Failed to start request: " + itos(err));
		export_req->queue_free();
	}
}

void AIAnimationUI::_on_auto_export_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, HTTPRequest *p_request) {
	if (!p_request) return;
	
	String save_path = p_request->get_meta("save_path", "");
	String format = p_request->get_meta("format", "sprite_sheet");
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_response_code != 200) {
		print_line("AUTO_EXPORT: Failed with HTTP " + itos(p_response_code));
		p_request->queue_free();
		return;
	}
	
	// Parse response
	String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
	Ref<JSON> json;
	json.instantiate();
	Error parse_err = json->parse(response_text);
	if (parse_err != OK) {
		print_line("AUTO_EXPORT: Failed to parse response");
		p_request->queue_free();
		return;
	}
	
	Dictionary result = json->get_data();
	
	// Ensure directory exists
	String dir_path = save_path.get_base_dir();
	DirAccess::make_dir_recursive_absolute(ProjectSettings::get_singleton()->globalize_path(dir_path));
	
	// Save based on format
	if (format == "sprite_sheet" || format == "all") {
		String base64_data = result.get("sprite_sheet_base64", "");
		if (!base64_data.is_empty()) {
			Vector<uint8_t> image_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(base64_data);
			String actual_path = (format == "all") ? save_path.get_basename() + "_sheet.png" : save_path;
			Ref<FileAccess> file = FileAccess::open(actual_path, FileAccess::WRITE);
			if (file.is_valid()) {
				file->store_buffer(image_data);
				file->close();
				print_line("AUTO_EXPORT: Saved sprite sheet to: " + actual_path);
			}
		}
	}
	
	if (format == "gif" || format == "all") {
		String base64_data = result.get("gif_base64", "");
		if (!base64_data.is_empty()) {
			Vector<uint8_t> gif_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(base64_data);
			String actual_path = (format == "all") ? save_path.get_basename() + ".gif" : save_path;
			Ref<FileAccess> file = FileAccess::open(actual_path, FileAccess::WRITE);
			if (file.is_valid()) {
				file->store_buffer(gif_data);
				file->close();
				print_line("AUTO_EXPORT: Saved GIF to: " + actual_path);
			}
		}
	}
	
	if (format == "frames" || format == "all") {
		Array frames = result.get("frames_base64", Array());
		String base_path = save_path.get_base_dir();
		String base_name = save_path.get_file().get_basename();
		
		for (int i = 0; i < frames.size(); i++) {
			String frame_base64 = frames[i];
			Vector<uint8_t> frame_data = CoreBind::Marshalls::get_singleton()->base64_to_raw(frame_base64);
			String frame_path = base_path + "/" + base_name + "_" + itos(i + 1).pad_zeros(3) + ".png";
			
			Ref<FileAccess> file = FileAccess::open(frame_path, FileAccess::WRITE);
			if (file.is_valid()) {
				file->store_buffer(frame_data);
				file->close();
			}
		}
		print_line("AUTO_EXPORT: Saved " + itos(frames.size()) + " frames to: " + base_path);
	}
	
	// Refresh filesystem
	if (EditorFileSystem::get_singleton()) {
		EditorFileSystem::get_singleton()->scan();
	}
	
	p_request->queue_free();
}

void AIAnimationUI::_on_template_export_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, HTTPRequest *p_request) {
	if (!p_request) return;
	
	String save_path = p_request->get_meta("save_path", "");
	
	if (p_result != HTTPRequest::RESULT_SUCCESS || p_response_code != 200) {
		print_line("TEMPLATE_EXPORT: Failed with HTTP " + itos(p_response_code));
		p_request->queue_free();
		return;
	}
	
	String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
	Ref<JSON> json;
	json.instantiate();
	Error parse_err = json->parse(response_text);
	if (parse_err != OK) {
		print_line("TEMPLATE_EXPORT: Failed to parse response");
		p_request->queue_free();
		return;
	}
	
	Dictionary result = json->get_data();
	Dictionary files = result.get("files", Dictionary());
	
	if (files.is_empty()) {
		print_line("TEMPLATE_EXPORT: No files in response");
		p_request->queue_free();
		return;
	}
	
	// Ensure directory exists (save_path should be a folder for templates)
	String folder_path = save_path.ends_with("/") ? save_path : save_path.get_base_dir();
	DirAccess::make_dir_recursive_absolute(ProjectSettings::get_singleton()->globalize_path(folder_path));
	
	// Save all files
	Array file_names = files.keys();
	int saved_count = 0;
	PackedStringArray saved_files;
	
	for (int i = 0; i < file_names.size(); i++) {
		String filename = file_names[i];
		String content = files[filename];
		String filepath = folder_path.path_join(filename);
		String global_path = ProjectSettings::get_singleton()->globalize_path(filepath);
		
		bool is_binary = filename.ends_with(".png");
		
		Ref<FileAccess> file = FileAccess::open(global_path, FileAccess::WRITE);
		if (file.is_valid()) {
			if (is_binary) {
				Vector<uint8_t> data = CoreBind::Marshalls::get_singleton()->base64_to_raw(content);
				file->store_buffer(data);
			} else {
				file->store_string(content);
			}
			file->close();
			saved_count++;
			saved_files.push_back(filepath);
		}
	}
	
	// Print clear file paths for user
	print_line("");
	print_line("============================================================");
	print_line("ANIMATION EXPORT COMPLETE");
	print_line("============================================================");
	print_line("Folder: " + folder_path);
	print_line("Files saved (" + itos(saved_count) + "):");
	for (int i = 0; i < saved_files.size(); i++) {
		print_line("  📄 " + saved_files[i]);
	}
	
	// Get template type and scene type from response
	String template_type = result.get("template_type", "character");
	String scene_type = result.get("scene_type", "");
	if (!scene_type.is_empty()) {
		print_line("Scene type: " + scene_type);
	}
	print_line("============================================================");
	
	// Refresh filesystem
	if (EditorFileSystem::get_singleton()) {
		EditorFileSystem::get_singleton()->scan();
	}
	
	p_request->queue_free();
}

