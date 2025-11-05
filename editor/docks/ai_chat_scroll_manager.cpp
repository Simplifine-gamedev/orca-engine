/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_scroll_manager.h"
#include "ai_chat_dock.h"

void AIChatScrollManager::scroll_to_bottom(ScrollContainer *p_chat_scroll) {
	if (!p_chat_scroll) {
		return;
	}
	
	// Use deferred scroll for better performance
	p_chat_scroll->call_deferred("queue_redraw");
	_perform_scroll(p_chat_scroll);
}

void AIChatScrollManager::scroll_to_bottom_smooth(ScrollContainer *p_chat_scroll) {
	// For now, just use regular scroll_to_bottom
	// TODO: Implement smooth scrolling animation with tweening
	scroll_to_bottom(p_chat_scroll);
}

bool AIChatScrollManager::is_at_bottom(ScrollContainer *p_chat_scroll) {
	if (!p_chat_scroll) {
		return true;
	}
	
	VScrollBar *vbar = p_chat_scroll->get_v_scroll_bar();
	if (!vbar) {
		return true;
	}
	
	// Consider user at bottom if within a small threshold of the end.
	// End is when value + page >= max.
	const float value = vbar->get_value();
	const float page = vbar->get_page();
	const float max_value = vbar->get_max();
	
	return (max_value - (value + page)) <= BOTTOM_THRESHOLD;
}

void AIChatScrollManager::scroll_to_position_after_load_more(
	ScrollContainer *p_chat_scroll,
	int p_loaded_messages_count
) {
	if (!p_chat_scroll) {
		return;
	}

	VScrollBar *vbar = p_chat_scroll->get_v_scroll_bar();
	if (!vbar) {
		return;
	}
	
	// Estimate scroll position based on loaded messages
	float estimated_position = _estimate_scroll_position(p_loaded_messages_count, p_chat_scroll);
	vbar->set_value(estimated_position);
}

void AIChatScrollManager::on_chat_scroll_changed(
	float p_value,
	bool &p_auto_scroll_at_bottom,
	ScrollContainer *p_chat_scroll
) {
	// Update auto-scroll state based on user's scroll position
	p_auto_scroll_at_bottom = is_at_bottom(p_chat_scroll);
}

void AIChatScrollManager::on_chat_content_min_size_changed(
	bool p_auto_scroll_at_bottom,
	ScrollContainer *p_chat_scroll
) {
	// Only auto-scroll when the user is already at the bottom
	if (p_auto_scroll_at_bottom) {
		scroll_to_bottom(p_chat_scroll);
	}
}

int AIChatScrollManager::calculate_initial_message_start_index(
	int p_total_messages,
	int p_initial_load_count
) {
	// Calculate starting index for pagination
	// Load recent messages first, older ones on demand
	
	if (p_total_messages <= p_initial_load_count) {
		return 0; // Load all messages if count is small
	}
	
	// Start from recent messages
	return p_total_messages - p_initial_load_count;
}

void AIChatScrollManager::_perform_scroll(ScrollContainer *p_chat_scroll) {
	if (!p_chat_scroll) {
		return;
	}

	VScrollBar *vbar = p_chat_scroll->get_v_scroll_bar();
	if (vbar) {
		vbar->set_value(vbar->get_max());
	}
}

float AIChatScrollManager::_estimate_scroll_position(int p_loaded_messages_count, ScrollContainer *p_chat_scroll) {
	if (!p_chat_scroll) {
		return 0.0f;
	}

	VScrollBar *vbar = p_chat_scroll->get_v_scroll_bar();
	if (!vbar) {
		return 0.0f;
	}

	// Estimate scroll position based on loaded messages
	// This is a heuristic - actual message heights vary
	float estimated_position = (float)p_loaded_messages_count / MESSAGE_HEIGHT_ESTIMATE * vbar->get_max() * POSITION_ESTIMATE_FACTOR;
	return estimated_position;
}
