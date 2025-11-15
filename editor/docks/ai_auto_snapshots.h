/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "core/string/ustring.h"
#include "core/object/ref_counted.h"
#include "scene/gui/tree.h"
#include "scene/gui/label.h"
#include "scene/gui/button.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/split_container.h"
#include "scene/gui/dialogs.h"
#include "scene/main/window.h"

class AIChatDock;

/**
 * AIAutoSnapshots - Read‑only viewer for AI automatic checkpoints.
 *
 * These checkpoints are created for each user message (via AICheckpointManager)
 * and stored as Git tags in the isolated `.ai-checkpoints` repository.
 *
 * This helper:
 *  - Lists all `msg_*` tags (auto checkpoints)
 *  - Shows when each checkpoint was taken and for which message index
 *  - When a checkpoint is selected, shows a folder tree of the snapshot contents
 *
 * NOTE: This does not perform restore operations. Restores are still handled
 *       by the existing per‑message checkpoint UI in the chat history.
 */
class AIAutoSnapshots : public RefCounted {
	GDCLASS(AIAutoSnapshots, RefCounted);

public:
	struct AutoSnapshot {
		String tag_name;           // Git tag name (msg_<index>_<timestamp>)
		int64_t message_index;     // Chat message index this checkpoint was taken for
		String created_timestamp;  // Human‑readable timestamp
		int64_t created_unix_time; // Unix timestamp for sorting / debugging
		String description;        // Tag message / reason (e.g. AI Chat checkpoint...)
	};

	AIAutoSnapshots();
	~AIAutoSnapshots();

	void initialize(AIChatDock *p_chat_dock);

	// Show the "AI Auto Snapshots" dialog.
	void show_auto_snapshots_dialog();

private:
	static void _bind_methods();

	AIChatDock *chat_dock = nullptr;

	// Dialog and UI elements.
	Window *auto_snapshots_window = nullptr;
	Tree *snapshot_list_tree = nullptr;
	Tree *folder_tree = nullptr;
	Label *details_label = nullptr;
	Button *restore_button = nullptr;

	// Currently selected snapshot info (for restore).
	String selected_tag_name;
	int64_t selected_message_index = -1;

	// Internal helpers.
	void _ensure_dialog_created();
	void _refresh_auto_snapshots_list();
	Vector<AutoSnapshot> _get_all_auto_snapshots();
	void _on_snapshot_item_selected();
	void _on_restore_button_pressed();
	void _on_restore_selected_snapshot_confirmed();
	void _populate_folder_tree(const String &p_tag_name);
	void _build_folder_tree_from_paths(const Vector<String> &p_paths);

	// Utility helpers.
	String _get_checkpoint_directory() const;
	String _format_timestamp_from_unix(int64_t p_unix_time) const;
	int64_t _parse_message_index_from_tag(const String &p_tag_name) const;
};


