print("\nThis is my output\n")

for i in "This is my output":
    print(i)

dict1 = {1:"one",2:"two"}

print(dict1[1])

name = "Sam"
age = 26

print(f'My name is {name} and my age is {age}')

print(name[0:])

list1 = [1,2,3,4]
list1.append(5)
print(list1)

nest = [1,2,[3,4]]
print(nest[2][1])

dict2 = {'word':{'innerkey':[1,2,3]}}
print(dict2['word']['innerkey'][2])

# Tuples
t = (1,2,3)
print(t[0])

# Tuples are not mutable (you cannot reassign the values of the items in the tuple)

# Sets
s1 = {1,1,1,1,1,2,2,2,2}
print(s1) # will print only distinct values

list2 = list(range(11))
print(list2)

# List Comprehension

x = [1,2,3,4]
out = []
for num in x: # normal appending method
    out.append(num**2)

out = [num**2 for num in x] # this saves lines of codes
print(out)

# DOCSTRINGS
def my_function():
    """
    THIS IS A DOCSTRING
    CAN GO MULTIPLE LINES
    """

my_function # hover over this function so we can view the docstring in the function

#########################################

## Map() Fucntion, Filter() Function, and Lambda Expressions ##

def times2(var):
    return var*2

# map() function

seq = [1,2,3,4,5]

map1 = list(map(times2,seq)) # applies the times2 function to each element in the list

print(map1)

# lambda expressions (also known as anonymous function)

def times3(var):return var*3

lambda var:var*3 # basically the same expression as the times3() function above

map2 = list(map(lambda var:var*3,seq)) # helps write out and use a function in the same line of code

print(map2)

# filters

filter1 = list(filter(lambda num:num%2 == 0,seq))

print(filter1)

# Other Built-in Functions

s2 = 'hello my name is Sam'

print(s2.lower())

print(s2.upper())

print(s2.split()) # splits on all whitespaces

print(s2.split('a')) # splits at every 'a'

# dictionary functions

dict3 = {'k1':1,'k2':2}

print(dict3.keys())
print(dict3.items()) # prints both keys and values
print(dict3.values())

# list fucntions

list3 = [1,2,3]
list3.pop()
print(list3)

# tuple unpacking

y = [(1,2),(3,4),(5,6)] 

for a,b in y: 
    print(a)
    print(b)


