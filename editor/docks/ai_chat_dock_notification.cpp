/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#include "ai_chat_dock_notification.h"
#include "ai_chat_dock.h"
#include "core/os/os.h"
#include "core/os/time.h"
#include "scene/gui/label.h"
#include "scene/gui/button.h"
#include "scene/gui/panel_container.h"
#include "scene/resources/style_box_flat.h"
#include "scene/main/scene_tree.h"
#include "editor/editor_node.h"

// ========== NOTIFICATION SYSTEM IMPLEMENTATION ==========

void AIChatDockNotification::show_status_notification(AIChatDock *p_dock, const String &p_type, const String &p_message, const String &p_icon, float p_duration) {
	// Implementation moved from main file - simplified for now
	print_line("AI Chat: Status notification: [" + p_type + "] " + p_message);
	// TODO: Restore full notification UI implementation
}

void AIChatDockNotification::show_summarization_notification(AIChatDock *p_dock, int p_original_count, int p_summary_tokens) {
	print_line("AI Chat: Summarization notification: " + String::num_int64(p_original_count) + " -> " + String::num_int64(p_summary_tokens) + " tokens");
}

void AIChatDockNotification::show_connection_status_notification(AIChatDock *p_dock, const String &p_status, const String &p_message) {
	print_line("AI Chat: Connection status: " + p_status + " - " + p_message);
}

void AIChatDockNotification::show_rate_limit_notification(AIChatDock *p_dock, const String &p_provider, const String &p_message) {
	print_line("AI Chat: Rate limit: " + p_provider + " - " + p_message);
}

void AIChatDockNotification::show_model_switch_notification(AIChatDock *p_dock, const String &p_from_provider, const String &p_to_provider, const String &p_reason) {
	print_line("AI Chat: Model switch: " + p_from_provider + " -> " + p_to_provider + " (" + p_reason + ")");
}

void AIChatDockNotification::hide_status_notification(AIChatDock *p_dock) {
	// TODO: Implement
}

void AIChatDockNotification::on_status_notification_timer_timeout(AIChatDock *p_dock) {
	// TODO: Implement
}

void AIChatDockNotification::show_rate_limit_popup(AIChatDock *p_dock, const String &p_provider, const String &p_message) {
	print_line("AI Chat: Rate limit popup: " + p_provider + " - " + p_message);
}

void AIChatDockNotification::show_provider_switch_popup(AIChatDock *p_dock, const String &p_from_provider, const String &p_to_provider, const String &p_reason) {
	print_line("AI Chat: Provider switch popup: " + p_from_provider + " -> " + p_to_provider + " (" + p_reason + ")");
}

void AIChatDockNotification::hide_popup_after_delay(AIChatDock *p_dock, float p_delay_seconds) {
	// TODO: Implement
}

void AIChatDockNotification::hide_rate_limit_popup(AIChatDock *p_dock) {
	// TODO: Implement
}
