extends Node

func _ready():
    print("Hello World"
    # Missing closing parenthesis - this should cause a parser error
    var x = 10

func broken_function(
    # Missing parameter and closing parenthesis - another error
    return 42