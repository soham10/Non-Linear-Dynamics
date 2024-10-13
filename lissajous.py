import turtle
from math import sin, pi

# Initialize the screen and turtle
window = turtle.Screen()
window.bgcolor("#FFFFFF")

myPen = turtle.Turtle()
myPen.hideturtle()
myPen.speed(0)
myPen.pensize(2)
myPen.color("#AA00AA")

# Parameters for the Lissajous curve
A = 100
B = 100
a = 1
b = 4
delta = pi / 2
t = 0

# Draw Lissajous curve
myPen.penup()
myPen.goto(A * sin(a * t + delta), B * sin(b * t))
myPen.pendown()

for i in range(2000):
    t += 0.01
    x = A * sin(a * t + delta)
    y = B * sin(b * t)
    
    myPen.goto(x, y)
    window.update()  # Update the screen manually

# Complete the drawing
myPen.hideturtle()

turtle.done()
