#!/usr/bin/python3

class exceptions(Exception):
    def __init__(self, message=""):
        self.message = message

    def __str__(self):
        if len(self.message) != 0:
            print("Good")
        else:
            print("Bad")

class classa():
    def __init__(self):
        pass

    def simple(self, check):
        if check:
            pass
        else:
            raise exceptions()

a = classa()
print(a.simple(False))