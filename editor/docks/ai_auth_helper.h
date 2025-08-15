/**************************************************************************/
/*  ai_auth_helper.h                                                      */
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
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/string/ustring.h"

namespace AILoginHelper {

static inline String build_auth_login_url(const String &p_api_endpoint, const String &p_machine_id, const String &p_provider) {
	String login;
	// Prefer cloud endpoint if configured via env var in editor process
	const char *cloud_env = getenv("AI_CHAT_CLOUD_URL");
	if (cloud_env && String(cloud_env).length() > 0) {
		String cloud = String(cloud_env).strip_edges();
		if (!cloud.begins_with("http://") && !cloud.begins_with("https://")) {
			cloud = String("https://") + cloud;
		}
		// Ensure /auth/login path
		if (cloud.find("/auth/login") == -1) {
			cloud = cloud.rstrip("/") + String("/auth/login");
		}
		login = cloud;
	} else {
		String base = p_api_endpoint;
		login = base.replace("/chat", "/auth/login");
	}
	String url = login + "?machine_id=" + p_machine_id + "&provider=" + p_provider;
	return url;
}

}


