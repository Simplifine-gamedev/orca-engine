/* AI Animation Status Tracker Implementation */

#include "ai_animation_tracker.h"
#include "core/io/json.h"
#include "core/os/time.h"
#include "scene/gui/box_container.h"
#include "scene/gui/separator.h"
#include "scene/resources/style_box_flat.h"

void AIAnimationTracker::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_poll_timer_timeout"), &AIAnimationTracker::_on_poll_timer_timeout);
	ClassDB::bind_method(D_METHOD("_on_poll_request_completed"), &AIAnimationTracker::_on_poll_request_completed);
}

AIAnimationTracker::AIAnimationTracker() {
	// Nodes will be created when initialized with chat_container
	poll_request = nullptr;
	poll_timer = nullptr;
}

AIAnimationTracker::~AIAnimationTracker() {
	if (poll_timer && poll_timer->is_inside_tree()) {
		poll_timer->stop();
	}
}

void AIAnimationTracker::initialize(const String &p_api_base, Control *p_chat_container) {
	api_base_url = p_api_base;
	chat_container = p_chat_container;
	
	if (!chat_container) {
		print_line("ANIM_TRACKER: Can't initialize - no chat container");
		return;
	}
	
	// Create and add HTTP request to chat container
	if (!poll_request) {
		poll_request = memnew(HTTPRequest);
		poll_request->connect("request_completed", callable_mp(this, &AIAnimationTracker::_on_poll_request_completed));
		chat_container->add_child(poll_request);
	}
	
	// Create and add timer to chat container
	if (!poll_timer) {
		poll_timer = memnew(Timer);
		poll_timer->set_wait_time(15.0);
		poll_timer->set_one_shot(false);
		poll_timer->connect("timeout", callable_mp(this, &AIAnimationTracker::_on_poll_timer_timeout));
		chat_container->add_child(poll_timer);
	}
}

void AIAnimationTracker::track_job(const String &p_job_id, const String &p_tool_call_id, const String &p_user_request, PanelContainer *p_ui_panel,
		const String &p_export_destination, int p_export_resolution, const String &p_export_format) {
	AnimationJob job;
	job.job_id = p_job_id;
	job.tool_call_id = p_tool_call_id;
	job.user_request = p_user_request;
	job.status = "pending";
	job.ui_panel = p_ui_panel;
	job.start_time_ms = Time::get_singleton()->get_ticks_msec();
	job.poll_count = 0;
	job.export_destination = p_export_destination;
	job.export_resolution = p_export_resolution;
	job.export_format = p_export_format;
	
	if (!p_export_destination.is_empty()) {
		print_line("TRACKER: Job " + p_job_id + " will auto-export to: " + p_export_destination + " at " + itos(p_export_resolution) + "px (" + p_export_format + ")");
	}
	
	// Find status label and progress bar in the UI panel
	if (p_ui_panel && p_ui_panel->get_child_count() > 0) {
		VBoxContainer *vbox = Object::cast_to<VBoxContainer>(p_ui_panel->get_child(0));
		if (vbox) {
			for (int i = 0; i < vbox->get_child_count(); i++) {
				Node *child = vbox->get_child(i);
				
				// Look for status label
				if (!job.status_label) {
					Label *label = Object::cast_to<Label>(child);
					if (label && String(label->get_name()).begins_with("anim_status_")) {
						job.status_label = label;
					}
				}
				
				// Look for progress bar
				if (!job.progress_bar) {
					ProgressBar *bar = Object::cast_to<ProgressBar>(child);
					if (bar) {
						job.progress_bar = bar;
					}
				}
			}
		}
	}
	
	active_jobs[p_job_id] = job;
	
	print_line("ANIM_TRACKER: Tracking job " + p_job_id.substr(0, 8));
	
	// Start polling if not already running
	if (poll_timer && poll_timer->is_stopped()) {
		poll_timer->start();
	}
	
	// Immediate first poll
	_poll_next_job();
}

void AIAnimationTracker::stop_tracking(const String &p_job_id) {
	if (active_jobs.has(p_job_id)) {
		active_jobs.erase(p_job_id);
		print_line("ANIM_TRACKER: Stopped tracking job " + p_job_id);
	}
	
	// Stop timer if no more jobs
	if (active_jobs.is_empty() && poll_timer) {
		poll_timer->stop();
	}
}

void AIAnimationTracker::update_job_ui(const String &p_job_id, const String &p_status, const Dictionary &p_progress) {
	if (!active_jobs.has(p_job_id)) {
		return;
	}
	
	AnimationJob &job = active_jobs[p_job_id];
	job.status = p_status;
	job.progress = p_progress;
	
	// Update status label
	if (job.status_label && job.status_label->is_inside_tree()) {
		String status_text = _format_progress_message(p_progress);
		job.status_label->set_text(status_text);
		
		// Update color based on status
		if (p_status == "completed") {
			job.status_label->add_theme_color_override("font_color", Color(0.3, 0.9, 0.3)); // Green
		} else if (p_status == "failed") {
			job.status_label->add_theme_color_override("font_color", Color(0.9, 0.3, 0.3)); // Red
		} else {
			job.status_label->add_theme_color_override("font_color", Color(0.9, 0.8, 0.3)); // Yellow
		}
	}
	
	// Update progress bar
	if (job.progress_bar && job.progress_bar->is_inside_tree()) {
		int current_level = p_progress.get("current_level", 0);
		int total_levels = p_progress.get("levels", 1);
		
		if (total_levels > 0) {
			float progress_pct = (float)current_level / (float)total_levels * 100.0f;
			job.progress_bar->set_value(progress_pct);
		}
	}
}

String AIAnimationTracker::get_job_status(const String &p_job_id) const {
	if (active_jobs.has(p_job_id)) {
		return active_jobs[p_job_id].status;
	}
	return "";
}

Dictionary AIAnimationTracker::get_job_data(const String &p_job_id) const {
	Dictionary result;
	if (active_jobs.has(p_job_id)) {
		const AnimationJob &job = active_jobs[p_job_id];
		result["job_id"] = job.job_id;
		result["tool_call_id"] = job.tool_call_id;
		result["status"] = job.status;
		result["progress"] = job.progress;
		result["supabase_project_id"] = job.supabase_project_id;
		result["user_request"] = job.user_request;
	}
	return result;
}

void AIAnimationTracker::_on_poll_timer_timeout() {
	if (active_jobs.is_empty()) {
		if (poll_timer) {
			poll_timer->stop();
		}
		return;
	}
	
	_poll_next_job();
}

void AIAnimationTracker::_poll_next_job() {
	if (active_jobs.is_empty() || !poll_request) {
		return;
	}
	
	// Constants for timeout
	const uint64_t MAX_POLL_TIME_MS = 10 * 60 * 1000; // 10 minutes
	const int MAX_POLL_COUNT = 50; // Safety limit (~12.5 minutes at 15s intervals)
	
	uint64_t current_time = Time::get_singleton()->get_ticks_msec();
	
	// Find the oldest pending/processing job to poll
	String job_to_poll;
	uint64_t oldest_time = UINT64_MAX;
	Vector<String> jobs_to_remove;
	
	for (const KeyValue<String, AnimationJob> &E : active_jobs) {
		const AnimationJob &job = E.value;
		// Continue polling for any non-terminal status
		if (job.status == "pending" || job.status == "processing" || job.status == "starting" || 
		    job.status == "creating_graph" || job.status == "uploading_to_supabase" || job.status.is_empty()) {
			// Check for timeout
			uint64_t elapsed = current_time - job.start_time_ms;
			if (elapsed > MAX_POLL_TIME_MS || job.poll_count >= MAX_POLL_COUNT) {
				print_line("ANIM_TRACKER: Job " + job.job_id + " timed out after " + itos(elapsed / 1000) + "s (" + itos(job.poll_count) + " polls)");
				jobs_to_remove.push_back(job.job_id);
				continue;
			}
			
			if (job.start_time_ms < oldest_time) {
				oldest_time = job.start_time_ms;
				job_to_poll = job.job_id;
			}
		}
	}
	
	// Remove timed out jobs
	for (const String &job_id : jobs_to_remove) {
		_handle_job_failed(job_id, "Timed out waiting for animation generation (10 minute limit)");
	}
	
	if (job_to_poll.is_empty()) {
		// No jobs left to poll, stop timer
		if (poll_timer) {
			poll_timer->stop();
			print_line("ANIM_TRACKER: No active jobs, stopping poll timer");
		}
		return;
	}
	
	// Poll this job
	String poll_url = api_base_url + "/animation/status/" + job_to_poll;
	
	PackedStringArray headers;
	headers.push_back("Content-Type: application/json");
	
	Error err = poll_request->request(poll_url, headers, HTTPClient::METHOD_GET);
	if (err == OK) {
		active_jobs[job_to_poll].poll_count++;
		print_line("ANIM_TRACKER: Polling job " + job_to_poll + " (poll #" + itos(active_jobs[job_to_poll].poll_count) + ") - URL: " + poll_url);
	} else {
		print_line("ANIM_TRACKER: Failed to send poll request (error: " + itos(err) + ")");
	}
}

void AIAnimationTracker::_on_poll_request_completed(int p_result, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_body) {
	String response_text = String::utf8((const char *)p_body.ptr(), p_body.size());
	
	// Handle connection failures
	if (p_result != HTTPRequest::RESULT_SUCCESS) {
		print_line("ANIM_TRACKER: Poll connection failed (result: " + itos(p_result) + ")");
		return; // Keep polling, might be temporary network issue
	}
	
	// Handle HTTP errors
	if (p_code == 404) {
		// Job not found - might still be initializing or server restarted
		// Keep polling but don't treat as error
		print_line("ANIM_TRACKER: Job not found (404) - may still be initializing, continuing polls...");
		return;
	}
	
	if (p_code == 202) {
		// 202 Accepted = still processing (server busy but job exists)
		print_line("ANIM_TRACKER: Server busy (202) - job still processing");
		return;
	}
	
	if (p_code == 500) {
		// Server error - log but keep polling (might recover)
		print_line("ANIM_TRACKER: Server error (500) - retrying...");
		if (!response_text.is_empty()) {
			print_line("   Response: " + response_text.substr(0, 200));
		}
		return;
	}
	
	if (p_code != 200) {
		print_line("ANIM_TRACKER: Poll request failed (code: " + itos(p_code) + ")");
		return;
	}
	
	Ref<JSON> json;
	json.instantiate();
	Error parse_err = json->parse(response_text);
	if (parse_err != OK) {
		print_line("ANIM_TRACKER: Failed to parse poll response: " + response_text.substr(0, 100));
		return;
	}
	
	Dictionary result = json->get_data();
	String job_id = result.get("job_id", "");
	String status = result.get("status", "unknown");
	Dictionary progress = result.get("progress", Dictionary());
	
	// Try to find job - might be missing job_id in response, use currently polling job
	if (job_id.is_empty()) {
		// Find any job we're actively polling
		for (const KeyValue<String, AnimationJob> &E : active_jobs) {
			if (E.value.status == "pending" || E.value.status == "processing" || E.value.status == "starting") {
				job_id = E.key;
				break;
			}
		}
	}
	
	if (job_id.is_empty() || !active_jobs.has(job_id)) {
		print_line("ANIM_TRACKER: Could not match response to active job");
		return;
	}
	
	print_line("ANIM_TRACKER: Job " + job_id + " status: " + status);
	
	// Update UI
	update_job_ui(job_id, status, progress);
	
	// Handle completion
	if (status == "completed") {
		String supabase_project_id = result.get("supabase_project_id", result.get("project_id", ""));
		active_jobs[job_id].supabase_project_id = supabase_project_id;
		
		_handle_job_completed(job_id, result);
	} else if (status == "failed") {
		String error = result.get("error", "Unknown error");
		_handle_job_failed(job_id, error);
	}
}

void AIAnimationTracker::_handle_job_completed(const String &p_job_id, const Dictionary &p_result) {
	if (!active_jobs.has(p_job_id)) {
		return;
	}
	
	const AnimationJob &job = active_jobs[p_job_id];
	
	print_line("ANIM_TRACKER: Job " + p_job_id.substr(0, 8) + " completed");
	
	// Include export settings in result for auto-export
	Dictionary result_with_export = p_result.duplicate();
	if (!job.export_destination.is_empty()) {
		result_with_export["auto_export"] = true;
		result_with_export["export_destination"] = job.export_destination;
		result_with_export["export_resolution"] = job.export_resolution;
		result_with_export["export_format"] = job.export_format;
		print_line("ANIM_TRACKER: Including auto-export settings: " + job.export_destination);
	}
	
	// Call completion callback if set
	if (on_job_completed_callback.is_valid()) {
		on_job_completed_callback.call(p_job_id, job.tool_call_id, result_with_export);
	}
	
	// Stop tracking this job
	stop_tracking(p_job_id);
}

void AIAnimationTracker::_handle_job_failed(const String &p_job_id, const String &p_error) {
	if (!active_jobs.has(p_job_id)) {
		return;
	}
	
	const AnimationJob &job = active_jobs[p_job_id];
	
	print_line("ANIM_TRACKER: Job " + p_job_id.substr(0, 8) + " failed: " + p_error);
	
	// Call failure callback if set
	if (on_job_failed_callback.is_valid()) {
		on_job_failed_callback.call(p_job_id, job.tool_call_id, p_error);
	}
	
	// Stop tracking this job
	stop_tracking(p_job_id);
}

String AIAnimationTracker::_format_progress_message(const Dictionary &p_progress) const {
	// Simple status message - no percentages
	String stage = p_progress.get("stage", "");
	String msg = p_progress.get("message", "");
	
	if (stage == "completed") {
		return "Done";
	} else if (stage == "failed") {
		return "Failed";
	} else if (!msg.is_empty()) {
		return msg;
	} else if (!stage.is_empty()) {
		return stage;
	}
	
	return "Generating...";
}

