# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:33:18 2024

@author: User
"""
a = 10
b = 12
def calcululator(a, b):
    if a>b :
        return "12345"
    else: 
        return [1,3,4]
    
print(calcululator(a, b))

def add(x:int, y:int)->int:
    return x+y
print(add(2,4))

F=32

def c(F):
    return (F-32)*9/5
print(c(F))

#list
list_a = [23,23,456,12,12,12,12,98, 1,7,4,1222]
value_remove = 23
while value_remove in list_a:
    list_a.remove(value_remove)
print(list_a)

value_2 = 12
list_a = [x for x in list_a if x != value_2]
print(list_a)
list_a.reverse()
print(list_a)
print(sorted(list_a))

H = "Hello world"
print(H.lower())
print(H.upper())
print(H.swapcase())
print(H.title())
print(H+H)
print(H*3)

def say_hello(x):
    out = f"I WANT TO SAY HA TO {x}"
    return out
print(say_hello("AMY"))