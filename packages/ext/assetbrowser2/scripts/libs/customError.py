# -*- coding: utf-8 -*-

class CustomError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return "Error: {}".format(self.value)
