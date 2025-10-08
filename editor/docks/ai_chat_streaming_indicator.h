/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "scene/gui/label.h"
#include "scene/main/timer.h"
#include "core/object/class_db.h"

// Animated streaming indicator for AI chat
class StreamingIndicator : public Label {
	GDCLASS(StreamingIndicator, Label);
	
private:
	Timer *animation_timer = nullptr;
	int dot_count = 0;
	bool is_animating = false; // Track animation state
	
	void _on_animation_timer_timeout();
	
protected:
	static void _bind_methods();
	
public:
	StreamingIndicator();
	~StreamingIndicator();
	
	void start_animation();
	void stop_animation();
	bool is_animation_active() const { return is_animating; }
	
	void _notification(int p_what);
};
