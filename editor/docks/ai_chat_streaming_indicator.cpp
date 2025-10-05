/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_streaming_indicator.h"

void StreamingIndicator::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_on_animation_timer_timeout"), &StreamingIndicator::_on_animation_timer_timeout);
}

StreamingIndicator::StreamingIndicator() {
	animation_timer = memnew(Timer);
	animation_timer->set_wait_time(0.4); // Update every 400ms for smoother animation
	animation_timer->set_one_shot(false);
	animation_timer->connect("timeout", callable_mp(this, &StreamingIndicator::_on_animation_timer_timeout));
	add_child(animation_timer);
	
	// Initial state
	set_text("");
	set_visible(false);
	dot_count = 0;
	
	// Set modulate to a dimmed color so it's visible but subtle
	set_modulate(Color(0.7, 0.7, 0.7, 1.0));
}

StreamingIndicator::~StreamingIndicator() {
	if (animation_timer && animation_timer->is_inside_tree()) {
		animation_timer->queue_free();
	}
}

void StreamingIndicator::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			// Ensure timer is setup correctly
			if (animation_timer && !animation_timer->is_inside_tree()) {
				add_child(animation_timer);
			}
		} break;
	}
}

void StreamingIndicator::start_animation() {
	set_visible(true);
	dot_count = 1;
	set_text(".");
	
	// Show the container if it exists
	if (has_meta("indicator_container")) {
		Control *container = Object::cast_to<Control>(get_meta("indicator_container"));
		if (container) {
			container->set_visible(true);
		}
	}
	
	if (animation_timer) {
		animation_timer->start();
	}
	print_line("AI Chat: Streaming indicator animation started");
}

void StreamingIndicator::stop_animation() {
	set_visible(false);
	set_text("");
	
	// Hide the container if it exists
	if (has_meta("indicator_container")) {
		Control *container = Object::cast_to<Control>(get_meta("indicator_container"));
		if (container) {
			container->set_visible(false);
		}
	}
	
	if (animation_timer) {
		animation_timer->stop();
	}
	dot_count = 0;
	print_line("AI Chat: Streaming indicator animation stopped");
}

void StreamingIndicator::_on_animation_timer_timeout() {
	dot_count++;
	if (dot_count > 3) {
		dot_count = 1;
	}
	
	String dots;
	for (int i = 0; i < dot_count; i++) {
		dots += ".";
	}
	
	set_text(dots);
}
