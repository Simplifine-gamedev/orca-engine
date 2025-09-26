/*
 * © 2025 Simplifine Corp.
 * Personal Non‑Commercial License applies to this file as an original contribution to this Godot fork.
 * See LICENSES/COMPANY-NONCOMMERCIAL.md for terms. Commercial use requires a separate license from Simplifine.
 */

#pragma once

#include "core/variant/array.h"
#include "core/string/ustring.h"
#include "core/math/vector2i.h"
#include "scene/main/node.h"

namespace AIChatDockTypes {
	
	struct AttachedFile {
		String path;
		String name;
		String content;
		bool is_image = false;
		String mime_type;
		String base64_data; // For images encoded for API
		Vector2i original_size = Vector2i(0, 0);
		Vector2i display_size = Vector2i(0, 0);
		bool was_downsampled = false;
		// Node support
		bool is_node = false;
		NodePath node_path;
		String node_type;
	};

	struct ChatMessage {
		String role; // "user", "assistant", or "tool"
		String content;
		String timestamp;
		// For assistant tool calls.
		Array tool_calls;
		// For tool responses.
		String tool_call_id;
		String name;
		// For attached files
		Vector<AttachedFile> attached_files;
		// For storing tool execution results (like generated images)
		Array tool_results;
		// For automatic project context injection (not displayed in UI)
		String project_context;
		// For thinking mode (reasoning content)
		String reasoning_content;
		Array thinking_blocks;
	};

	struct Conversation {
		String id;
		String title;
		String created_timestamp;
		String last_modified_timestamp;
		Vector<ChatMessage> messages;
		// Persistent pending edits - only for edits that haven't been accepted/rejected
		HashMap<String, String> pending_apply_edits; // tool_call_id -> file_path
	};
}

// Type aliases for backward compatibility
using AIChatAttachedFile = AIChatDockTypes::AttachedFile;
using AIChatMessage = AIChatDockTypes::ChatMessage;
using AIChatConversation = AIChatDockTypes::Conversation;
