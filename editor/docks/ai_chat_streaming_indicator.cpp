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
	// CRITICAL FIX: Do NOT access child nodes in destructor - they may already be destroyed
	// All cleanup is now handled in _notification(NOTIFICATION_PREDELETE)
}

void StreamingIndicator::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			// Ensure timer is setup correctly with validity checks  
			Timer *timer = Object::cast_to<Timer>(animation_timer);
			if (timer && !timer->is_inside_tree()) {
				add_child(timer);
			}
		} break;
		case NOTIFICATION_PREDELETE: {
			// CRITICAL FIX: Clean up timer reference before destruction
			if (animation_timer) {
				Timer *timer = Object::cast_to<Timer>(animation_timer);
				if (timer && is_animating) {
					timer->stop();
				}
				// Clear the pointer to prevent accessing freed memory in destructor
				animation_timer = nullptr;
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			// CRITICAL: Stop animation when removed from tree to prevent orphaned timers
			if (is_animating) {
				stop_animation();
			}
		} break;
	}
}

void StreamingIndicator::start_animation() {
	
	is_animating = true;
	set_visible(true);
	dot_count = 1;
	set_text(".");
	
	// Show the container if it exists
	if (has_meta("indicator_container")) {
		Control *container = Object::cast_to<Control>(get_meta("indicator_container"));
		if (container) {
			container->set_visible(true);
		} else {
		}
	} else {
	}
	
	if (animation_timer) {
		Timer *timer = Object::cast_to<Timer>(animation_timer);
		if (timer) {
			timer->start();
		} else {
		}
	} else {
	}
	
}

void StreamingIndicator::stop_animation() {
	
	is_animating = false;
	
	// CRITICAL FIX: Stop timer FIRST before hiding - with null check after predelete cleanup
	if (animation_timer && ObjectDB::get_instance(animation_timer->get_instance_id())) {
		Timer *timer = Object::cast_to<Timer>(animation_timer);
		if (timer && !timer->is_stopped()) {
			timer->stop();
		} else if (!timer) {
			// Timer became invalid - null the pointer to prevent future crashes
			animation_timer = nullptr;
		}
	} else {
	}
	
	set_visible(false);
	set_text("");
	dot_count = 0;
	
	// Hide the container if it exists
	if (has_meta("indicator_container")) {
		Control *container = Object::cast_to<Control>(get_meta("indicator_container"));
		if (container) {
			container->set_visible(false);
		}
	}
	
}

void StreamingIndicator::_on_animation_timer_timeout() {
	// CRITICAL FIX: Safety check with null check after predelete cleanup
	if (!animation_timer) {
		is_animating = false;
		return;
	}
	
	Timer *timer = Object::cast_to<Timer>(animation_timer);
	if (!is_animating || !timer || timer->is_stopped()) {
		if (timer && !timer->is_stopped()) {
			timer->stop();
		} else if (!timer) {
			// Timer became invalid - null the pointer to prevent future crashes
			animation_timer = nullptr;
		}
		return;
	}
	
	// Safety check: don't animate if not visible or not in tree
	if (!is_visible() || !is_inside_tree()) {
		stop_animation();
		return;
	}
	
	dot_count++;
	if (dot_count > 3) {
		dot_count = 1;
	}
	
	String dots;
	for (int i = 0; i < dot_count; i++) {
		dots += ".";
	}
	
	set_text(dots);
	// Removed verbose logging - indicator is working correctly
}
