# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:35:38 2018

@author: Elisabeth
"""

class Affixes:
    def __init__(self):
        self.affixes=[]
        with open('utils/greek_and_latin_affixes.txt') as f:
            for line in f:
                self.affixes.append(line.replace("\n", ""))
                
#    def get_suffixes(self):
#        return self.suffixes
#                
        
    