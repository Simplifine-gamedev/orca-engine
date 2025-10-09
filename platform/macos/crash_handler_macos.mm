/**************************************************************************/
/*  crash_handler_macos.mm                                                */
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

#import "crash_handler_macos.h"

#include "core/config/crash_report_config.h"
#include "core/config/project_settings.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/version.h"
#include "main/main.h"

#include <unistd.h>

// For HTTP crash reporting
#import <Foundation/Foundation.h>

#if defined(DEBUG_ENABLED)
#define CRASH_HANDLER_ENABLED 1
#endif

#ifdef CRASH_HANDLER_ENABLED
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <csignal>
#include <cstdlib>

#import <mach-o/dyld.h>
#import <mach-o/getsect.h>

static uint64_t load_address() {
	const struct segment_command_64 *cmd = getsegbyname("__TEXT");
	char full_path[1024];
	uint32_t size = sizeof(full_path);

	if (cmd && !_NSGetExecutablePath(full_path, &size)) {
		uint32_t dyld_count = _dyld_image_count();
		for (uint32_t i = 0; i < dyld_count; i++) {
			const char *image_name = _dyld_get_image_name(i);
			if (image_name && strncmp(image_name, full_path, 1024) == 0) {
				return cmd->vmaddr + _dyld_get_image_vmaddr_slide(i);
			}
		}
	}

	return 0;
}

static void send_crash_report_to_backend(const String &p_crash_dump) {
	// Determine endpoint based on environment
	const char *crash_url = CRASH_REPORT_URL;
	
	// Check if running in dev mode (check for DEV_MODE or IS_DEV environment variables)
	const char *dev_mode_env = getenv("DEV_MODE");
	const char *is_dev_env = getenv("IS_DEV");
	if ((dev_mode_env && strcmp(dev_mode_env, "true") == 0) || 
	    (is_dev_env && strcmp(is_dev_env, "true") == 0)) {
		crash_url = CRASH_REPORT_URL_DEV;
		fprintf(stderr, "CRASH_REPORTER: Using dev endpoint: %s\n", crash_url);
	} else {
		fprintf(stderr, "CRASH_REPORTER: Using production endpoint: %s\n", crash_url);
	}
	
	// Get project name and version
	String project_name = "Unknown";
	if (ProjectSettings::get_singleton()) {
		project_name = GLOBAL_GET("application/config/name");
	}
	
	String engine_version = GODOT_VERSION_FULL_NAME;
	if (!String(GODOT_VERSION_HASH).is_empty()) {
		engine_version = vformat("%s (%s)", GODOT_VERSION_FULL_NAME, GODOT_VERSION_HASH);
	}
	
	// Get machine/user identifiers if available
	String machine_id = OS::get_singleton()->get_unique_id();
	if (machine_id.is_empty()) {
		machine_id = "unknown";
	}
	
	@autoreleasepool {
		// Create JSON payload
		NSDictionary *payload = @{
			@"crash_dump": [NSString stringWithUTF8String:p_crash_dump.utf8().get_data()],
			@"platform": @"macos",
			@"engine_version": [NSString stringWithUTF8String:engine_version.utf8().get_data()],
			@"project_name": [NSString stringWithUTF8String:project_name.utf8().get_data()],
			@"machine_id": [NSString stringWithUTF8String:machine_id.utf8().get_data()],
			@"user_id": @"crash_reporter",
			@"timestamp": @((int64_t)[[NSDate date] timeIntervalSince1970])
		};
		
		NSError *json_error = nil;
		NSData *json_data = [NSJSONSerialization dataWithJSONObject:payload options:0 error:&json_error];
		
		if (json_error || !json_data) {
			fprintf(stderr, "CRASH_REPORTER: Failed to serialize crash report JSON\n");
			return;
		}
		
		// Create request
		NSURL *url = [NSURL URLWithString:[NSString stringWithUTF8String:crash_url]];
		NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
		[request setHTTPMethod:@"POST"];
		[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
		[request setHTTPBody:json_data];
		[request setTimeoutInterval:5.0]; // 5 second timeout (we're about to crash anyway)
		
		// Send synchronously (we're in crash handler, need to complete before abort)
		NSHTTPURLResponse *response = nil;
		NSError *error = nil;
		[NSURLConnection sendSynchronousRequest:request returningResponse:&response error:&error];
		
		if (error) {
			fprintf(stderr, "CRASH_REPORTER: Failed to send crash report: %s\n", [[error localizedDescription] UTF8String]);
		} else if (response && [response statusCode] == 200) {
			fprintf(stderr, "CRASH_REPORTER: Crash report sent successfully to %s\n", crash_url);
		} else {
			fprintf(stderr, "CRASH_REPORTER: Server returned status %ld\n", (long)[response statusCode]);
		}
	}
}

static void handle_crash(int sig) {
	signal(SIGSEGV, SIG_DFL);
	signal(SIGFPE, SIG_DFL);
	signal(SIGILL, SIG_DFL);
	signal(SIGTRAP, SIG_DFL);

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
	
	// Also print to stderr for local debugging
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
	if (strings) {
		void *load_addr = (void *)load_address();

		for (size_t i = 1; i < size; i++) {
			char fname[1024];
			Dl_info info;

			snprintf(fname, 1024, "%s", strings[i]);

			// Try to demangle the function name to provide a more readable one
			if (dladdr(bt_buffer[i], &info) && info.dli_sname) {
				if (info.dli_sname[0] == '_') {
					int status;
					char *demangled = abi::__cxa_demangle(info.dli_sname, nullptr, 0, &status);

					if (status == 0 && demangled) {
						snprintf(fname, 1024, "%s", demangled);
					}

					if (demangled) {
						free(demangled);
					}
				}
			}

			String output = fname;

			// Try to get the file/line number using atos
			if (bt_buffer[i] > (void *)0x0 && OS::get_singleton()) {
				List<String> args;
				char str[1024];

				args.push_back("-o");
				args.push_back(_execpath);
#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__)
				args.push_back("-arch");
				args.push_back("x86_64");
#elif defined(__aarch64__)
				args.push_back("-arch");
				args.push_back("arm64");
#endif
				args.push_back("--fullPath");
				args.push_back("-l");
				snprintf(str, 1024, "%p", load_addr);
				args.push_back(str);
				snprintf(str, 1024, "%p", bt_buffer[i]);
				args.push_back(str);

				int ret;
				String out = "";
				Error err = OS::get_singleton()->execute(String("atos"), args, &out, &ret);
				if (err == OK && out.substr(0, 2) != "0x") {
					out = out.substr(0, out.length() - 1);
					output = out;
				}
			}

			String frame_line = vformat("[%d] %s", (int64_t)i, output);
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
	signal(SIGTRAP, SIG_DFL);
#endif

	disabled = true;
}

void CrashHandler::initialize() {
#ifdef CRASH_HANDLER_ENABLED
	signal(SIGSEGV, handle_crash);
	signal(SIGFPE, handle_crash);
	signal(SIGILL, handle_crash);
	signal(SIGTRAP, handle_crash);
#endif
}
