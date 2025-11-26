/* AI Animation UI Components
 * © 2025 Simplifine Corp.
 * Beautiful UI for displaying sprite animations with auto-tracking
 */

#ifndef AI_ANIMATION_UI_H
#define AI_ANIMATION_UI_H

#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/texture_rect.h"
#include "core/io/http_client.h"

class AIChatDock; // Forward declaration
class HTTPRequest; // Forward declaration

class AIAnimationUI {
public:
	// Create a beautiful animation job tracker panel with auto-updating progress
	static PanelContainer *create_animation_job_panel(
		const String &p_job_id,
		const String &p_user_request,
		const Dictionary &p_result,
		VBoxContainer *p_parent_container,
		AIChatDock *p_dock
	);
	
	// Update an existing animation job panel with new status/progress
	static void update_animation_job_panel(
		PanelContainer *p_panel,
		const String &p_status,
		const Dictionary &p_progress
	);
	
	// Create completed animation display with GIF preview and download buttons
	static void create_animation_result_panel(
		VBoxContainer *p_parent_container,
		const Dictionary &p_completion_data,
		AIChatDock *p_dock
	);
	
	// Create numbered animation list display
	static void create_numbered_animation_list(
		VBoxContainer *p_parent_container,
		const Array &p_animations_list,
		int p_total_count,
		AIChatDock *p_dock
	);
	
	// Download animation GIF from URL and display
	static void download_and_display_animation_gif(
		const String &p_thumbnail_url,
		TextureRect *p_texture_rect,
		Label *p_status_label
	);
	
	// Helper to create status badge
	static Label *create_status_badge(const String &p_status, Control *p_parent);
	
	// Helper to format progress message
	static String format_progress_message(const Dictionary &p_progress);
	
	// Thumbnail loading callback (static PNG)
	static void _on_thumbnail_loaded(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, TextureRect *p_rect, HTTPRequest *p_http);
	
	// Start loading animated thumbnail (multiple frames)
	static void _start_animated_thumbnail_load(TextureRect *p_rect, const Array &p_frame_urls, AIChatDock *p_dock);
	
	// Auto-export animation when job completes
	static void trigger_auto_export(
		const String &p_anim_id,
		const String &p_project_id, 
		const String &p_save_path,
		int p_resolution,
		const String &p_format,
		Control *p_parent  // Node to attach HTTP request to
	);
	
	// Callback for auto-export completion
	static void _on_auto_export_completed(int p_result, int p_response_code, const PackedStringArray &p_headers, const PackedByteArray &p_body, HTTPRequest *p_request);

private:
	// Helper to format time remaining
	static String format_time_estimate(int p_seconds_elapsed, const String &p_status);
	
	// Helper to get API base URL
	static String _get_api_base_url();
};

#endif // AI_ANIMATION_UI_H

