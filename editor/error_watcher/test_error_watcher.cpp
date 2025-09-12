/**************************************************************************/
/*  test_error_watcher.cpp                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "error_watcher.h"
#include "core/string/print_string.h"

void test_error_watcher() {
	print_line("Testing ErrorWatcher functionality...");
	
	// Create ErrorWatcher instance
	ErrorWatcher *watcher = memnew(ErrorWatcher);
	ErrorWatcherPanel *panel = memnew(ErrorWatcherPanel);
	watcher->initialize(panel);
	
	// Test error parsing
	print_line("\n--- Testing Error Classification ---");
	
	// Test duplicate variable error
	String duplicate_error = "Variable 'player' is already declared";
	watcher->process_runtime_error("test.gd", 5, 1, duplicate_error);
	
	// Test undefined variable error
	String undefined_error = "Identifier 'unknown_var' is not defined";
	watcher->process_runtime_error("test.gd", 8, 10, undefined_error);
	
	// Test syntax error
	String syntax_error = "Expected ')' after expression";
	watcher->process_runtime_error("test.gd", 12, 15, syntax_error);
	
	// Test missing import error
	String import_error = "Class 'CustomNode' not found";
	watcher->process_runtime_error("test.gd", 1, 1, import_error);
	
	// Display results
	Vector<ErrorWatcherError> errors = watcher->get_all_errors();
	print_line(vformat("Detected %d errors:", errors.size()));
	
	for (int i = 0; i < errors.size(); i++) {
		const ErrorWatcherError &error = errors[i];
		String type_str;
		switch (error.type) {
			case ErrorWatcherError::DUPLICATE_VARIABLE:
				type_str = "DUPLICATE_VARIABLE";
				break;
			case ErrorWatcherError::UNDEFINED_VARIABLE:
				type_str = "UNDEFINED_VARIABLE";
				break;
			case ErrorWatcherError::SYNTAX_ERROR:
				type_str = "SYNTAX_ERROR";
				break;
			case ErrorWatcherError::MISSING_IMPORT:
				type_str = "MISSING_IMPORT";
				break;
			default:
				type_str = "UNKNOWN";
				break;
		}
		
		print_line(vformat("  %d. [%s] Line %d: %s", i + 1, type_str, error.line, error.message));
		
		if (error.has_quick_fix) {
			Vector<QuickFixAction> fixes = watcher->get_quick_fixes(error);
			for (const QuickFixAction &fix : fixes) {
				print_line(vformat("    → Quick Fix: %s", fix.description));
			}
		}
	}
	
	// Test telemetry
	print_line("\n--- Testing Telemetry ---");
	Dictionary telemetry = watcher->get_telemetry_data();
	print_line(vformat("Auto fixes shown: %d", telemetry.get("auto_fix_shown", 0)));
	print_line(vformat("Auto fixes applied: %d", telemetry.get("auto_fix_applied", 0)));
	
	// Cleanup
	memdelete(panel);
	memdelete(watcher);
	
	print_line("\n✅ ErrorWatcher test completed!");
}

// Example of how to integrate with existing error systems
void demonstrate_integration() {
	print_line("\n--- Demonstrating Integration ---");
	
	// Simulate script validation errors (like from ScriptTextEditor)
	List<ScriptLanguage::ScriptError> script_errors;
	
	ScriptLanguage::ScriptError error1;
	error1.line = 4;
	error1.column = 20;
	error1.message = "Missing closing parenthesis ')'";
	error1.path = "res://test_script.gd";
	script_errors.push_back(error1);
	
	ScriptLanguage::ScriptError error2;
	error2.line = 8;
	error2.column = 1;
	error2.message = "Expected parameter name";
	error2.path = "res://test_script.gd";
	script_errors.push_back(error2);
	
	// Process with ErrorWatcher
	ErrorWatcher *watcher = memnew(ErrorWatcher);
	ErrorWatcherPanel *panel = memnew(ErrorWatcherPanel);
	watcher->initialize(panel);
	
	watcher->process_script_errors(script_errors, "res://test_script.gd");
	
	Vector<ErrorWatcherError> processed_errors = watcher->get_errors_for_file("res://test_script.gd");
	print_line(vformat("Processed %d script errors for file", processed_errors.size()));
	
	// Simulate compiler output parsing
	String compiler_output = R"(res://player.gd:5:10: error: Variable 'speed' is already declared
res://enemy.gd:12:1: error: Identifier 'player_ref' is not defined
res://main.gd:3:15: error: Expected ':' after function declaration)";
	
	watcher->process_compiler_output(compiler_output);
	
	Vector<ErrorWatcherError> all_errors = watcher->get_all_errors();
	print_line(vformat("Total errors after compiler output: %d", all_errors.size()));
	
	// Cleanup
	memdelete(panel);
	memdelete(watcher);
}