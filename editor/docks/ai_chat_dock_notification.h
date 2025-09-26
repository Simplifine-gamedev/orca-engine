/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "core/variant/dictionary.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/progress_bar.h"
#include "scene/main/timer.h"

class AIChatDock;

class AIChatDockNotification {
public:
	// Status notification system
	static void show_status_notification(AIChatDock *p_dock, const String &p_type, const String &p_message, const String &p_icon = "", float p_duration = 3.0);
	static void show_summarization_notification(AIChatDock *p_dock, int p_original_count, int p_summary_tokens);
	static void show_connection_status_notification(AIChatDock *p_dock, const String &p_status, const String &p_message = "");
	static void show_rate_limit_notification(AIChatDock *p_dock, const String &p_provider, const String &p_message);
	static void show_model_switch_notification(AIChatDock *p_dock, const String &p_from_provider, const String &p_to_provider, const String &p_reason);
	static void hide_status_notification(AIChatDock *p_dock);
	static void on_status_notification_timer_timeout(AIChatDock *p_dock);
	
	// Rate limit popups
	static void show_rate_limit_popup(AIChatDock *p_dock, const String &p_provider, const String &p_message);
	static void show_provider_switch_popup(AIChatDock *p_dock, const String &p_from_provider, const String &p_to_provider, const String &p_reason);
	static void hide_popup_after_delay(AIChatDock *p_dock, float p_delay_seconds);
	static void hide_rate_limit_popup(AIChatDock *p_dock);
	
	// Loading screen for chunked conversation loading
	static void show_loading_screen(AIChatDock *p_dock, const String &p_message, int p_total_items);
	static void hide_loading_screen(AIChatDock *p_dock);
	static void update_loading_progress(AIChatDock *p_dock, const String &p_message, int p_current, int p_total);
	static void start_chunked_conversation_loading(AIChatDock *p_dock, int p_conversation_index);
	static void process_conversation_loading_chunk(AIChatDock *p_dock);
	static void process_tool_results_chunk(AIChatDock *p_dock);
	static void finish_chunked_conversation_loading(AIChatDock *p_dock);
	
	// Popup cleanup
	static void cleanup_popup(AcceptDialog *p_popup);
};
