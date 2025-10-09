/**************************************************************************/
/*  crash_handler_linuxbsd.cpp                                            */
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

#include "crash_handler_linuxbsd.h"

#include "core/config/crash_report_config.h"
#include "core/config/project_settings.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/version.h"
#include "main/main.h"

// For HTTP crash reporting
#include <curl/curl.h>

#ifndef DEBUG_ENABLED
#undef CRASH_HANDLER_ENABLED
#endif

#ifdef CRASH_HANDLER_ENABLED
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <link.h>
#include <csignal>
#include <cstdlib>

static void send_crash_report_to_backend(const String &p_crash_dump) {
	// Determine endpoint based on environment
	const char *crash_url = CRASH_REPORT_URL;
	
	// Check if running in dev mode (DEV_MODE or IS_DEV)
	const char *dev_mode_env = getenv("DEV_MODE");
	const char *is_dev_env = getenv("IS_DEV");
	if ((dev_mode_env && strcmp(dev_mode_env, "true") == 0) || 
	    (is_dev_env && strcmp(is_dev_env, "true") == 0)) {
		crash_url = CRASH_REPORT_URL_DEV;
		fprintf(stderr, "CRASH_REPORTER: Using dev endpoint: %s\n", crash_url);
	} else {
		fprintf(stderr, "CRASH_REPORTER: Using production endpoint: %s\n", crash_url);
	}
	
	// Get project name
	String project_name = "Unknown";
	if (ProjectSettings::get_singleton()) {
		project_name = GLOBAL_GET("application/config/name");
	}
	
	String engine_version = GODOT_VERSION_FULL_NAME;
	if (!String(GODOT_VERSION_HASH).is_empty()) {
		engine_version = vformat("%s (%s)", GODOT_VERSION_FULL_NAME, GODOT_VERSION_HASH);
	}
	
	String machine_id = OS::get_singleton() ? OS::get_singleton()->get_unique_id() : "unknown";
	if (machine_id.is_empty()) {
		machine_id = "unknown";
	}
	
	// Build JSON payload (manual construction)
	String json_payload = "{";
	json_payload += vformat("\"crash_dump\": \"%s\",", p_crash_dump.c_escape());
	json_payload += "\"platform\": \"linux\",";
	json_payload += vformat("\"engine_version\": \"%s\",", engine_version.c_escape());
	json_payload += vformat("\"project_name\": \"%s\",", project_name.c_escape());
	json_payload += vformat("\"machine_id\": \"%s\",", machine_id.c_escape());
	json_payload += "\"user_id\": \"crash_reporter\",";
	json_payload += vformat("\"timestamp\": %lld", (int64_t)time(nullptr));
	json_payload += "}";
	
	// Send via libcurl (simple synchronous request)
	CURL *curl = curl_easy_init();
	if (curl) {
		struct curl_slist *headers = nullptr;
		headers = curl_slist_append(headers, "Content-Type: application/json");
		
		curl_easy_setopt(curl, CURLOPT_URL, crash_url);
		curl_easy_setopt(curl, CURLOPT_POST, 1L);
		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.utf8().get_data());
		curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(json_payload.utf8().get_data()));
		curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
		curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L); // 5 second timeout
		curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L); // Thread-safe
		
		CURLcode res = curl_easy_perform(curl);
		
		if (res == CURLE_OK) {
			long response_code;
			curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
			if (response_code == 200) {
				fprintf(stderr, "CRASH_REPORTER: Crash report sent successfully to %s\n", crash_url);
			} else {
				fprintf(stderr, "CRASH_REPORTER: Server returned status %ld\n", response_code);
			}
		} else {
			fprintf(stderr, "CRASH_REPORTER: Failed to send crash report: %s\n", curl_easy_strerror(res));
		}
		
		curl_slist_free_all(headers);
		curl_easy_cleanup(curl);
	}
}

static void handle_crash(int sig) {
	signal(SIGSEGV, SIG_DFL);
	signal(SIGFPE, SIG_DFL);
	signal(SIGILL, SIG_DFL);

	if (OS::get_singleton() == nullptr) {
		abort();
	}

	if (OS::get_singleton()->is_crash_handler_silent()) {
		std::_Exit(0);
	}

	void *bt_buffer[256];
	size_t size = backtrace(bt_buffer, 256);
	String _execpath = OS::get_singleton()->get_executable_path();

	String msg;
	if (ProjectSettings::get_singleton()) {
		msg = GLOBAL_GET("debug/settings/crash_handler/message");
	}

	// Tell MainLoop about the crash. This can be handled by users too in Node.
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_CRASH);
	}

	// Build crash dump string for backend reporting
	String crash_dump;
	crash_dump += "\n================================================================\n";
	crash_dump += vformat("%s: Program crashed with signal %d\n", __FUNCTION__, sig);
	
	// Dump the backtrace to stderr with a message to the user
	print_error("\n================================================================");
	print_error(vformat("%s: Program crashed with signal %d", __FUNCTION__, sig));

	// Print the engine version just before, so that people are reminded to include the version in backtrace reports.
	String version_line;
	if (String(GODOT_VERSION_HASH).is_empty()) {
		version_line = vformat("Engine version: %s", GODOT_VERSION_FULL_NAME);
	} else {
		version_line = vformat("Engine version: %s (%s)", GODOT_VERSION_FULL_NAME, GODOT_VERSION_HASH);
	}
	print_error(version_line);
	crash_dump += version_line + "\n";
	
	String backtrace_header = vformat("Dumping the backtrace. %s", msg);
	print_error(backtrace_header);
	crash_dump += backtrace_header + "\n";
	char **strings = backtrace_symbols(bt_buffer, size);
	// PIE executable relocation, zero for non-PIE executables
#ifdef __GLIBC__
	// This is a glibc only thing apparently.
	uintptr_t relocation = _r_debug.r_map->l_addr;
#else
	// Non glibc systems apparently don't give PIE relocation info.
	uintptr_t relocation = 0;
#endif //__GLIBC__
	if (strings) {
		List<String> args;
		for (size_t i = 0; i < size; i++) {
			char str[1024];
			snprintf(str, 1024, "%p", (void *)((uintptr_t)bt_buffer[i] - relocation));
			args.push_back(str);
		}
		args.push_back("-e");
		args.push_back(_execpath);

		// Try to get the file/line number using addr2line
		int ret;
		String output = "";
		Error err = OS::get_singleton()->execute(String("addr2line"), args, &output, &ret);
		Vector<String> addr2line_results;
		if (err == OK) {
			addr2line_results = output.substr(0, output.length() - 1).split("\n", false);
		}

		for (size_t i = 1; i < size; i++) {
			char fname[1024];
			Dl_info info;

			snprintf(fname, 1024, "%s", strings[i]);

			// Try to demangle the function name to provide a more readable one
			if (dladdr(bt_buffer[i], &info) && info.dli_sname) {
				if (info.dli_sname[0] == '_') {
					int status = 0;
					char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);

					if (status == 0 && demangled) {
						snprintf(fname, 1024, "%s", demangled);
					}

					if (demangled) {
						free(demangled);
					}
				}
			}

			// Simplify printed file paths to remove redundant `/./` sections (e.g. `/opt/godot/./core` -> `/opt/godot/core`).
			String frame_line = vformat("[%d] %s (%s)", (int64_t)i, fname, err == OK ? addr2line_results[i].replace("/./", "/") : "");
			print_error(frame_line);
			crash_dump += frame_line + "\n";
		}

		free(strings);
	}
	print_error("-- END OF C++ BACKTRACE --");
	crash_dump += "-- END OF C++ BACKTRACE --\n";
	print_error("================================================================");
	crash_dump += "================================================================\n";

	for (const Ref<ScriptBacktrace> &backtrace : ScriptServer::capture_script_backtraces(false)) {
		if (!backtrace->is_empty()) {
			String script_trace = backtrace->format();
			print_error(script_trace);
			crash_dump += script_trace + "\n";
			
			String script_end = vformat("-- END OF %s BACKTRACE --", backtrace->get_language_name().to_upper());
			print_error(script_end);
			crash_dump += script_end + "\n";
			
			print_error("================================================================");
			crash_dump += "================================================================\n";
		}
	}

	// Send crash report to backend before aborting
	send_crash_report_to_backend(crash_dump);

	// Abort to pass the error to the OS
	abort();
}
#endif

CrashHandler::CrashHandler() {
	disabled = false;
}

CrashHandler::~CrashHandler() {
	disable();
}

void CrashHandler::disable() {
	if (disabled) {
		return;
	}

#ifdef CRASH_HANDLER_ENABLED
	signal(SIGSEGV, SIG_DFL);
	signal(SIGFPE, SIG_DFL);
	signal(SIGILL, SIG_DFL);
#endif

	disabled = true;
}

void CrashHandler::initialize() {
#ifdef CRASH_HANDLER_ENABLED
	signal(SIGSEGV, handle_crash);
	signal(SIGFPE, handle_crash);
	signal(SIGILL, handle_crash);
#endif
}
