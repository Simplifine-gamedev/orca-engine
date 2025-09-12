extends Node
# Test script to demonstrate ErrorWatcher functionality
# This script intentionally contains various types of errors

# Missing import error - CustomNode is not defined
var custom_node: CustomNode

# Duplicate variable error 
var player_speed = 100
var player_speed = 200  # This should trigger duplicate variable detection

func _ready():
    print("Hello World"  # Missing closing parenthesis - syntax error
    
    # Undefined variable error
    print(unknown_variable)  # This variable is not defined
    
    # Another syntax error - missing colon after function declaration
    func broken_function()
        return 42
    
    # Type mismatch error (if type checking is enabled)
    var number: int = "string"  # String assigned to int variable

# Function with parameter syntax error
func another_broken_function(
    # Missing parameter name and closing parenthesis
    return "test"

# Missing import for Vector3
func use_vector3():
    var pos = Vector3(1, 2, 3)  # Should work, Vector3 is built-in
    var custom_vec = CustomVector3(1, 2, 3)  # This would need an import

# Duplicate function names
func duplicate_function():
    pass

func duplicate_function():  # Duplicate function name
    pass

# Common GDScript errors that ErrorWatcher should detect:
# 1. Missing closing parenthesis/brackets
# 2. Undefined variables/functions
# 3. Duplicate declarations
# 4. Missing import statements
# 5. Type mismatches
# 6. Missing colons after function/if/for statements
# 7. Incorrect indentation (though this is more complex)

# Expected ErrorWatcher behavior:
# - Detect each error type and classify correctly
# - Provide appropriate quick fixes where possible
# - Show inline gutter markers for each error
# - Allow one-click application of fixes with undo support
# - Track telemetry for fix usage