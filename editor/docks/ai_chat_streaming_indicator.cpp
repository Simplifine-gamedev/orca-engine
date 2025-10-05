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
	animation_timer->set_wait_time(0.5); // Update every 500ms
	animation_timer->set_one_shot(false);
	animation_timer->connect("timeout", callable_mp(this, &StreamingIndicator::_on_animation_timer_timeout));
	add_child(animation_timer);
	
	// Initial state
	set_text("");
	set_visible(false);
	dot_count = 0;
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
	dot_count = 0;
	set_text(".");
	if (animation_timer) {
		animation_timer->start();
	}
}

void StreamingIndicator::stop_animation() {
	set_visible(false);
	set_text("");
	if (animation_timer) {
		animation_timer->stop();
	}
	dot_count = 0;
}

void StreamingIndicator::_on_animation_timer_timeout() {
	dot_count = (dot_count + 1) % 4; // Cycle through 0, 1, 2, 3
	
	String dots;
	for (int i = 0; i < dot_count; i++) {
		dots += ".";
	}
	
	// Show at least one space to prevent flicker when empty
	if (dots.is_empty()) {
		dots = " ";
	}
	
	set_text(dots);
}
