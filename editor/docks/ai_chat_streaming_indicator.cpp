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
	print_line("AI Chat: StreamingIndicator constructor called");
	animation_timer = memnew(Timer);
	animation_timer->set_wait_time(0.4); // Update every 400ms for smoother animation
	animation_timer->set_one_shot(false);
	animation_timer->connect("timeout", callable_mp(this, &StreamingIndicator::_on_animation_timer_timeout));
	add_child(animation_timer);
	print_line("AI Chat: StreamingIndicator timer created and connected");
	
	// Initial state
	set_text("");
	set_visible(false);
	dot_count = 0;
	
	// Set modulate to a dimmed color so it's visible but subtle
	set_modulate(Color(0.7, 0.7, 0.7, 1.0));
	print_line("AI Chat: StreamingIndicator constructor complete - initial state: hidden, text: ''");
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
		case NOTIFICATION_EXIT_TREE: {
			// CRITICAL: Stop animation when removed from tree to prevent orphaned timers
			if (is_animating) {
				print_line("AI Chat: StreamingIndicator exiting tree while animating - stopping");
				stop_animation();
			}
		} break;
	}
}

void StreamingIndicator::start_animation() {
	print_line("AI Chat: StreamingIndicator::start_animation() called");
	print_line("AI Chat: - is_inside_tree: " + String(is_inside_tree() ? "YES" : "NO"));
	print_line("AI Chat: - is_visible_in_tree (before): " + String(is_visible_in_tree() ? "YES" : "NO"));
	print_line("AI Chat: - has_meta(indicator_container): " + String(has_meta("indicator_container") ? "YES" : "NO"));
	
	is_animating = true;
	set_visible(true);
	dot_count = 1;
	set_text(".");
	
	// Show the container if it exists
	if (has_meta("indicator_container")) {
		Control *container = Object::cast_to<Control>(get_meta("indicator_container"));
		if (container) {
			print_line("AI Chat: - Found indicator_container, setting visible");
			print_line("AI Chat: - Container was visible: " + String(container->is_visible() ? "YES" : "NO"));
			print_line("AI Chat: - Container is_inside_tree: " + String(container->is_inside_tree() ? "YES" : "NO"));
			container->set_visible(true);
			print_line("AI Chat: - Container now visible: " + String(container->is_visible() ? "YES" : "NO"));
		} else {
			print_line("AI Chat: - WARNING: indicator_container meta exists but cast failed!");
		}
	} else {
		print_line("AI Chat: - WARNING: No indicator_container meta found!");
	}
	
	if (animation_timer) {
		animation_timer->start();
		print_line("AI Chat: - Animation timer started, active: " + String(animation_timer->is_stopped() ? "NO" : "YES"));
	} else {
		print_line("AI Chat: - ERROR: No animation_timer!");
	}
	
	print_line("AI Chat: - is_visible (after): " + String(is_visible() ? "YES" : "NO"));
	print_line("AI Chat: - is_visible_in_tree (after): " + String(is_visible_in_tree() ? "YES" : "NO"));
	print_line("AI Chat: - text set to: '" + get_text() + "'");
	print_line("AI Chat: Streaming indicator animation started");
}

void StreamingIndicator::stop_animation() {
	print_line("AI Chat: StreamingIndicator::stop_animation() called");
	print_line("AI Chat: - was visible: " + String(is_visible() ? "YES" : "NO"));
	print_line("AI Chat: - was visible_in_tree: " + String(is_visible_in_tree() ? "YES" : "NO"));
	
	is_animating = false;
	
	// Stop timer FIRST before hiding
	if (animation_timer && !animation_timer->is_stopped()) {
		animation_timer->stop();
		print_line("AI Chat: - Stopped animation timer");
	}
	
	set_visible(false);
	set_text("");
	dot_count = 0;
	
	// Hide the container if it exists
	if (has_meta("indicator_container")) {
		Control *container = Object::cast_to<Control>(get_meta("indicator_container"));
		if (container) {
			print_line("AI Chat: - Hiding indicator_container");
			container->set_visible(false);
		}
	}
	
	print_line("AI Chat: Streaming indicator animation stopped");
}

void StreamingIndicator::_on_animation_timer_timeout() {
	// Safety check: verify animation should still be running
	if (!is_animating || !animation_timer || animation_timer->is_stopped()) {
		print_line("AI Chat: StreamingIndicator tick but not animating - stopping timer");
		if (animation_timer && !animation_timer->is_stopped()) {
			animation_timer->stop();
		}
		return;
	}
	
	// Safety check: don't animate if not visible or not in tree
	if (!is_visible() || !is_inside_tree()) {
		print_line("AI Chat: StreamingIndicator tick while invisible/not in tree - stopping animation");
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
	print_line("AI Chat: StreamingIndicator animation tick - dots: '" + dots + "', visible: " + String(is_visible_in_tree() ? "YES" : "NO"));
}
