/**************************************************************************/
/*  ai_chat_streaming_manager.cpp                                         */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.         */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "ai_chat_streaming_manager.h"

bool AIChatStreamingManager::should_show_indicator(bool p_is_waiting_for_response, bool p_is_content_streaming, int p_pending_tool_tasks) {
	// Indicator should ONLY show when:
	// 1. We're waiting for a response
	// 2. Content is NOT currently streaming
	// 3. No tool tasks are pending
	// This ensures the indicator appears AFTER all outputs (text + tool calls) are complete
	return p_is_waiting_for_response && !p_is_content_streaming && p_pending_tool_tasks == 0;
}

bool AIChatStreamingManager::should_hide_during_content(bool p_is_content_streaming) {
	// Always hide when content is actively streaming
	return p_is_content_streaming;
}

bool AIChatStreamingManager::should_show_after_completion(bool p_is_waiting_for_response, int p_pending_tool_tasks) {
	// Show only after all tool tasks complete and we're still waiting
	return p_is_waiting_for_response && p_pending_tool_tasks == 0;
}


