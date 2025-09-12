# Error Watcher System

The Error Watcher system provides automatic error detection and one-click fixes for common script/compiler errors in the Godot Editor.

## Features

### ✅ Implemented

1. **Core ErrorWatcher Class** - Parses and classifies compiler/runtime errors
2. **Inline Gutter Markers** - Visual error indicators in the script editor with clickable icons
3. **Error Panel** - Dedicated dock showing all detected errors with navigation
4. **Quick Fix Actions** - Automated fixes for common error types
5. **Telemetry Integration** - Tracks usage of auto-fix features

### Error Types Supported

- **Duplicate Variable** - Detects and suggests renaming for duplicate variable declarations
- **Undefined Variable** - Suggests imports or variable declarations for undefined identifiers  
- **Missing Import** - Recommends import statements for unknown classes
- **Syntax Errors** - Basic fixes for common syntax issues (missing parentheses, colons, etc.)
- **Type Mismatch** - Detection of type conversion issues

### Quick Fix Actions

1. **Rename Variable** - Automatically renames duplicate variables with suggested names
2. **Add Import** - Inserts appropriate import/preload statements
3. **Add Declaration** - Creates variable declarations for undefined variables
4. **Fix Syntax** - Corrects common syntax mistakes

## Architecture

### Core Components

```
ErrorWatcher (Singleton)
├── ErrorWatcherPanel (UI Dock)
├── QuickFixPopup (Fix Selection Dialog)  
├── Error Classification Engine
├── Quick Fix Generator
└── Telemetry System
```

### Integration Points

- **ScriptTextEditor** - Gutter integration and error processing
- **EditorNode** - Dock registration and initialization
- **EditorLog** - Runtime error capture
- **Build System** - Compiler output parsing

## Usage

### For Users

1. **View Errors**: Errors appear as icons in the script editor gutter
2. **Quick Fixes**: Click error icons to see available fixes
3. **Error Panel**: View all errors in the dedicated "Error Watcher" dock
4. **Apply Fixes**: Preview and apply suggested fixes with one click

### For Developers

```cpp
// Get ErrorWatcher instance
ErrorWatcher *watcher = ErrorWatcher::get_singleton();

// Process script errors
watcher->process_script_errors(error_list, file_path);

// Get quick fixes for an error
Vector<QuickFixAction> fixes = watcher->get_quick_fixes(error);

// Apply a fix
bool success = watcher->apply_quick_fix(fix_action);
```

## Error Classification

The system uses pattern matching to classify errors:

```cpp
ErrorWatcherError::Type _classify_error(const String &p_message) {
    if (_is_duplicate_variable_error(p_message)) 
        return DUPLICATE_VARIABLE;
    if (_is_undefined_variable_error(p_message))
        return UNDEFINED_VARIABLE;
    // ... more classifications
}
```

## Telemetry Data

The system tracks:
- `auto_fix_shown` - Number of times quick fixes were displayed
- `auto_fix_applied` - Number of fixes successfully applied  
- `auto_fix_undone` - Number of fixes that were undone
- `error_type_counts` - Frequency of each error type

## File Structure

```
editor/error_watcher/
├── error_watcher.h          # Main ErrorWatcher class
├── error_watcher.cpp        # Implementation
├── quick_fix_popup.cpp      # Quick fix dialog
├── test_error_watcher.cpp   # Test harness
├── SCsub                    # Build configuration
└── README.md                # This file
```

## Configuration

Error detection is enabled by default. Settings can be configured in:
- Editor Settings → Interface → Error Watcher

## Future Enhancements

- Support for more language-specific errors (C#, Visual Script)
- Custom error patterns via plugins
- Machine learning-based error prediction
- Integration with external linters
- Batch error fixing
- Error suppression/filtering

## Testing

Run the test harness to validate functionality:

```cpp
#include "editor/error_watcher/test_error_watcher.cpp"

void test_all() {
    test_error_watcher();
    demonstrate_integration();
}
```

## Acceptance Criteria Status

- [x] Errors are surfaced inline with file/line and explanation
- [x] 3+ common error types have working one-click fixes with preview
- [x] Telemetry for auto_fix_shown / auto_fix_applied / auto_fix_undone
- [x] Minimally invasive edits with confirmation
- [x] Integration with existing error handling systems

## Performance Considerations

- Error processing is done asynchronously to avoid blocking the UI
- Gutter updates are batched to minimize redraws
- Quick fix generation is lazy-loaded when needed
- Telemetry data is stored efficiently in memory

## Known Limitations

- Currently focused on GDScript errors (C# support planned)
- Quick fixes are pattern-based, not semantically aware
- Some complex syntax errors may not have automated fixes
- Undo integration requires additional work for complex multi-file fixes