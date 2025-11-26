/* AI Animation Status Tracker
 * © 2025 Simplifine Corp.
 * Tracks long-running 2D animation generation jobs with polling and UI updates
 */

#ifndef AI_ANIMATION_TRACKER_H
#define AI_ANIMATION_TRACKER_H

#include "scene/gui/control.h"
#include "scene/gui/label.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/progress_bar.h"
#include "scene/main/http_request.h"
#include "scene/main/timer.h"

class AIAnimationTracker : public RefCounted {
	GDCLASS(AIAnimationTracker, RefCounted);

public:
	struct AnimationJob {
		String job_id;
		String tool_call_id;
		String user_request;
		String status; // pending, processing, completed, failed
		Dictionary progress;
		String supabase_project_id;
		PanelContainer *ui_panel = nullptr;
		Label *status_label = nullptr;
		ProgressBar *progress_bar = nullptr;
		uint64_t start_time_ms = 0;
		int poll_count = 0;
		// Auto-export settings (set when job is created)
		String export_destination;  // e.g., "res://sprites/hero/"
		int export_resolution = 128;
		String export_format = "sprite_sheet";  // sprite_sheet, frames, gif, all
	};

private:
	HashMap<String, AnimationJob> active_jobs; // job_id -> AnimationJob
	HTTPRequest *poll_request = nullptr;
	Timer *poll_timer = nullptr;
	String api_base_url;
	Control *chat_container = nullptr;
	
	// Callbacks for when jobs complete
	Callable on_job_completed_callback;
	Callable on_job_failed_callback;

protected:
	static void _bind_methods();

public:
	AIAnimationTracker();
	~AIAnimationTracker();

	void initialize(const String &p_api_base, Control *p_chat_container);
	
	// Start tracking a new animation job
	void track_job(const String &p_job_id, const String &p_tool_call_id, const String &p_user_request, PanelContainer *p_ui_panel,
		const String &p_export_destination = "", int p_export_resolution = 128, const String &p_export_format = "sprite_sheet");
	
	// Stop tracking a job
	void stop_tracking(const String &p_job_id);
	
	// Update UI for a specific job
	void update_job_ui(const String &p_job_id, const String &p_status, const Dictionary &p_progress);
	
	// Set callbacks
	void set_on_job_completed(const Callable &p_callback) { on_job_completed_callback = p_callback; }
	void set_on_job_failed(const Callable &p_callback) { on_job_failed_callback = p_callback; }
	
	// Get job info
	bool has_job(const String &p_job_id) const { return active_jobs.has(p_job_id); }
	String get_job_status(const String &p_job_id) const;
	Dictionary get_job_data(const String &p_job_id) const;

private:
	void _on_poll_timer_timeout();
	void _poll_next_job();
	void _on_poll_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body);
	void _handle_job_completed(const String &p_job_id, const Dictionary &p_result);
	void _handle_job_failed(const String &p_job_id, const String &p_error);
	String _format_progress_message(const Dictionary &p_progress) const;
};

#endif // AI_ANIMATION_TRACKER_H

