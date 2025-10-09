/**************************************************************************/
/*  crash_handler_windows_seh.cpp                                         */
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

#include "crash_handler_windows.h"

#include "core/config/crash_report_config.h"
#include "core/config/project_settings.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/version.h"
#include "main/main.h"

#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")

#ifdef CRASH_HANDLER_EXCEPTION

// Backtrace code based on: https://stackoverflow.com/questions/6205981/windows-c-stack-trace-from-a-running-app

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <string>
#include <vector>

#include <psapi.h>

// Some versions of imagehlp.dll lack the proper packing directives themselves
// so we need to do it.
#pragma pack(push, before_imagehlp, 8)
#include <imagehlp.h>
#pragma pack(pop, before_imagehlp)

struct module_data {
	std::string image_name;
	std::string module_name;
	void *base_address = nullptr;
	DWORD load_size;
};

class symbol {
	typedef IMAGEHLP_SYMBOL64 sym_type;
	sym_type *sym;
	static const int max_name_len = 1024;

public:
	symbol(HANDLE process, DWORD64 address) :
			sym((sym_type *)::operator new(sizeof(*sym) + max_name_len)) {
		memset(sym, '\0', sizeof(*sym) + max_name_len);
		sym->SizeOfStruct = sizeof(*sym);
		sym->MaxNameLength = max_name_len;
		DWORD64 displacement;

		SymGetSymFromAddr64(process, address, &displacement, sym);
	}

	std::string name() { return std::string(sym->Name); }
	std::string undecorated_name() {
		if (*sym->Name == '\0') {
			return "<couldn't map PC to fn name>";
		}
		std::vector<char> und_name(max_name_len);
		UnDecorateSymbolName(sym->Name, &und_name[0], max_name_len, UNDNAME_COMPLETE);
		return std::string(&und_name[0], strlen(&und_name[0]));
	}
};

static void send_crash_report_to_backend(const String &p_crash_dump) {
	// Determine endpoint based on environment
	const char *crash_url = CRASH_REPORT_URL;
	
	// Check if running in dev mode (DEV_MODE or IS_DEV)
	char dev_mode_env[256];
	char is_dev_env[256];
	bool is_dev = false;
	
	if (GetEnvironmentVariableA("DEV_MODE", dev_mode_env, sizeof(dev_mode_env)) > 0) {
		is_dev = strcmp(dev_mode_env, "true") == 0;
	}
	if (!is_dev && GetEnvironmentVariableA("IS_DEV", is_dev_env, sizeof(is_dev_env)) > 0) {
		is_dev = strcmp(is_dev_env, "true") == 0;
	}
	
	if (is_dev) {
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
	
	// Build JSON payload (manual construction to avoid dependencies in crash handler)
	String json_payload = "{";
	json_payload += vformat("\"crash_dump\": \"%s\",", p_crash_dump.c_escape());
	json_payload += "\"platform\": \"windows\",";
	json_payload += vformat("\"engine_version\": \"%s\",", engine_version.c_escape());
	json_payload += vformat("\"project_name\": \"%s\",", project_name.c_escape());
	json_payload += vformat("\"machine_id\": \"%s\",", machine_id.c_escape());
	json_payload += "\"user_id\": \"crash_reporter\",";
	json_payload += vformat("\"timestamp\": %lld", (int64_t)time(nullptr));
	json_payload += "}";
	
	// Parse URL components
	String url_str(crash_url);
	bool use_https = url_str.begins_with("https://");
	String domain = url_str.substr(use_https ? 8 : 7);
	int path_start = domain.find("/");
	String path = "/crash_report";
	if (path_start >= 0) {
		path = domain.substr(path_start);
		domain = domain.substr(0, path_start);
	}
	
	// Send via WinHTTP (synchronous, simple, built into Windows)
	HINTERNET hSession = WinHttpOpen(L"Orca Engine Crash Reporter/1.0",
		WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
		WINHTTP_NO_PROXY_NAME,
		WINHTTP_NO_PROXY_BYPASS, 0);
	
	if (hSession) {
		HINTERNET hConnect = WinHttpConnect(hSession,
			(LPCWSTR)domain.utf16().get_data(),
			use_https ? INTERNET_DEFAULT_HTTPS_PORT : INTERNET_DEFAULT_HTTP_PORT, 0);
		
		if (hConnect) {
			HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST",
				(LPCWSTR)path.utf16().get_data(),
				nullptr, WINHTTP_NO_REFERER,
				WINHTTP_DEFAULT_ACCEPT_TYPES,
				use_https ? WINHTTP_FLAG_SECURE : 0);
			
			if (hRequest) {
				LPCWSTR headers = L"Content-Type: application/json\r\n";
				const char *data = json_payload.utf8().get_data();
				DWORD data_len = (DWORD)strlen(data);
				
				if (WinHttpSendRequest(hRequest, headers, -1, (LPVOID)data, data_len, data_len, 0) &&
					WinHttpReceiveResponse(hRequest, nullptr)) {
					fprintf(stderr, "CRASH_REPORTER: Crash report sent successfully to %s\n", crash_url);
				} else {
					fprintf(stderr, "CRASH_REPORTER: Failed to send crash report (error: %lu)\n", GetLastError());
				}
				
				WinHttpCloseHandle(hRequest);
			}
			WinHttpCloseHandle(hConnect);
		}
		WinHttpCloseHandle(hSession);
	}
}

class get_mod_info {
	HANDLE process;

public:
	get_mod_info(HANDLE h) :
			process(h) {}

	module_data operator()(HMODULE module) {
		module_data ret;
		char temp[4096];
		MODULEINFO mi;

		GetModuleInformation(process, module, &mi, sizeof(mi));
		ret.base_address = mi.lpBaseOfDll;
		ret.load_size = mi.SizeOfImage;

		GetModuleFileNameEx(process, module, temp, sizeof(temp));
		ret.image_name = temp;
		GetModuleBaseName(process, module, temp, sizeof(temp));
		ret.module_name = temp;
		std::vector<char> img(ret.image_name.begin(), ret.image_name.end());
		std::vector<char> mod(ret.module_name.begin(), ret.module_name.end());
		SymLoadModule64(process, nullptr, &img[0], &mod[0], (DWORD64)ret.base_address, ret.load_size);
		return ret;
	}
};

DWORD CrashHandlerException(EXCEPTION_POINTERS *ep) {
	HANDLE process = GetCurrentProcess();
	HANDLE hThread = GetCurrentThread();
	DWORD offset_from_symbol = 0;
	IMAGEHLP_LINE64 line = {};
	std::vector<module_data> modules;
	DWORD cbNeeded;
	std::vector<HMODULE> module_handles(1);

	if (OS::get_singleton() == nullptr || OS::get_singleton()->is_disable_crash_handler() || IsDebuggerPresent()) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	if (OS::get_singleton()->is_crash_handler_silent()) {
		std::_Exit(0);
	}

	String msg;
	if (ProjectSettings::get_singleton()) {
		msg = GLOBAL_GET("debug/settings/crash_handler/message");
	}

	// Tell MainLoop about the crash. This can be handled by users too in Node.
	if (OS::get_singleton()->get_main_loop()) {
		OS::get_singleton()->get_main_loop()->notification(MainLoop::NOTIFICATION_CRASH);
	}

	// REAL-TIME CRASH DUMP FILE: Write to disk immediately in case of force quit
	String crash_file_path;
	FILE *crash_file = nullptr;
	
	if (OS::get_singleton()) {
		String user_data_dir = OS::get_singleton()->get_user_data_dir();
		String crash_dir = user_data_dir.path_join("crashes");
		
		// Create directory
		CreateDirectoryW((LPCWSTR)crash_dir.utf16().get_data(), nullptr);
		
		// Create timestamped crash file
		__time64_t now = _time64(nullptr);
		crash_file_path = crash_dir.path_join(vformat("crash_%lld.txt", (int64_t)now));
		crash_file = _wfopen((LPCWSTR)crash_file_path.utf16().get_data(), L"w");
		
		if (crash_file) {
			fprintf(stderr, "CRASH_REPORTER: Writing crash dump to: %s\n", crash_file_path.utf8().get_data());
		}
	}
	
	auto write_crash_line = [&](const String &line) {
		if (crash_file) {
			fprintf(crash_file, "%s\n", line.utf8().get_data());
			fflush(crash_file);
		}
	};

	// Build crash dump string for backend reporting
	String crash_dump;
	crash_dump += "\n================================================================\n";
	write_crash_line("\n================================================================");
	
	String crash_header = vformat("%s: Program crashed", __FUNCTION__);
	crash_dump += crash_header + "\n";
	write_crash_line(crash_header);
	
	print_error("\n================================================================");
	print_error(vformat("%s: Program crashed", __FUNCTION__));

	// Print the engine version just before, so that people are reminded to include the version in backtrace reports.
	String version_line;
	if (String(GODOT_VERSION_HASH).is_empty()) {
		version_line = vformat("Engine version: %s", GODOT_VERSION_FULL_NAME);
	} else {
		version_line = vformat("Engine version: %s (%s)", GODOT_VERSION_FULL_NAME, GODOT_VERSION_HASH);
	}
	print_error(version_line);
	crash_dump += version_line + "\n";
	write_crash_line(version_line);
	
	String backtrace_header = vformat("Dumping the backtrace. %s", msg);
	print_error(backtrace_header);
	crash_dump += backtrace_header + "\n";
	write_crash_line(backtrace_header);

	// Load the symbols:
	if (!SymInitialize(process, nullptr, false)) {
		return EXCEPTION_CONTINUE_SEARCH;
	}

	SymSetOptions(SymGetOptions() | SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_EXACT_SYMBOLS);
	EnumProcessModules(process, &module_handles[0], module_handles.size() * sizeof(HMODULE), &cbNeeded);
	module_handles.resize(cbNeeded / sizeof(HMODULE));
	EnumProcessModules(process, &module_handles[0], module_handles.size() * sizeof(HMODULE), &cbNeeded);
	std::transform(module_handles.begin(), module_handles.end(), std::back_inserter(modules), get_mod_info(process));
	void *base = modules[0].base_address;

	// Setup stuff:
	CONTEXT *context = ep->ContextRecord;
	STACKFRAME64 frame;
	bool skip_first = false;

	frame.AddrPC.Mode = AddrModeFlat;
	frame.AddrStack.Mode = AddrModeFlat;
	frame.AddrFrame.Mode = AddrModeFlat;

#if defined(_M_X64)
	frame.AddrPC.Offset = context->Rip;
	frame.AddrStack.Offset = context->Rsp;
	frame.AddrFrame.Offset = context->Rbp;
#elif defined(_M_ARM64) || defined(_M_ARM64EC)
	frame.AddrPC.Offset = context->Pc;
	frame.AddrStack.Offset = context->Sp;
	frame.AddrFrame.Offset = context->Fp;
#elif defined(_M_ARM)
	frame.AddrPC.Offset = context->Pc;
	frame.AddrStack.Offset = context->Sp;
	frame.AddrFrame.Offset = context->R11;
#else
	frame.AddrPC.Offset = context->Eip;
	frame.AddrStack.Offset = context->Esp;
	frame.AddrFrame.Offset = context->Ebp;

	// Skip the first one to avoid a duplicate on 32-bit mode
	skip_first = true;
#endif

	line.SizeOfStruct = sizeof(line);
	IMAGE_NT_HEADERS *h = ImageNtHeader(base);
	DWORD image_type = h->FileHeader.Machine;

	int n = 0;
	do {
		if (skip_first) {
			skip_first = false;
		} else {
			if (frame.AddrPC.Offset != 0) {
				std::string fnName = symbol(process, frame.AddrPC.Offset).undecorated_name();

				String frame_line;
				if (SymGetLineFromAddr64(process, frame.AddrPC.Offset, &offset_from_symbol, &line)) {
					frame_line = vformat("[%d] %s (%s:%d)", n, fnName.c_str(), (char *)line.FileName, (int)line.LineNumber);
				} else {
					frame_line = vformat("[%d] %s", n, fnName.c_str());
				}
				print_error(frame_line);
				crash_dump += frame_line + "\n";
				write_crash_line(frame_line);
			} else {
				String frame_line = vformat("[%d] ???", n);
				print_error(frame_line);
				crash_dump += frame_line + "\n";
				write_crash_line(frame_line);
			}

			n++;
		}

		if (!StackWalk64(image_type, process, hThread, &frame, context, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) {
			break;
		}
	} while (frame.AddrReturn.Offset != 0 && n < 256);

	String cpp_end = "-- END OF C++ BACKTRACE --";
	print_error(cpp_end);
	crash_dump += cpp_end + "\n";
	write_crash_line(cpp_end);
	
	String separator = "================================================================";
	print_error(separator);
	crash_dump += separator + "\n";
	write_crash_line(separator);

	SymCleanup(process);

	for (const Ref<ScriptBacktrace> &backtrace : ScriptServer::capture_script_backtraces(false)) {
		if (!backtrace->is_empty()) {
			String script_trace = backtrace->format();
			print_error(script_trace);
			crash_dump += script_trace + "\n";
			Vector<String> trace_lines = script_trace.split("\n");
			for (const String &trace_line : trace_lines) {
				write_crash_line(trace_line);
			}
			
			String script_end = vformat("-- END OF %s BACKTRACE --", backtrace->get_language_name().to_upper());
			print_error(script_end);
			crash_dump += script_end + "\n";
			write_crash_line(script_end);
			
			print_error("================================================================");
			crash_dump += "================================================================\n";
			write_crash_line("================================================================");
		}
	}

	// Close crash file before HTTP POST
	if (crash_file) {
		fclose(crash_file);
		fprintf(stderr, "CRASH_REPORTER: Crash dump saved to: %s\n", crash_file_path.utf8().get_data());
	}

	// Send crash report to backend before passing to OS
	send_crash_report_to_backend(crash_dump);

	// Pass the exception to the OS
	return EXCEPTION_CONTINUE_SEARCH;
}
#endif

CrashHandler::CrashHandler() {
	disabled = false;
}

CrashHandler::~CrashHandler() {
}

void CrashHandler::disable() {
	if (disabled) {
		return;
	}

	disabled = true;
}

void CrashHandler::initialize() {
}
