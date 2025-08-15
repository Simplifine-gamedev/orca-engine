extends Node

func _ready():
    print("Hello World")
    var x = 10
    var y = 20
    var sum = x + y
    print("Sum is: ", sum)

func calculate(a, b):
    return a + b