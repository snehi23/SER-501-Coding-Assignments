# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 18:31:27 2011

@author: jfemiani
"""
import os
import sys
from glob import glob
import cProfile

#Your implementation goes into ch15.py
from ch15 import print_neatly

log = open('output.log', 'w')

print "Running...."
#This tests the code against my own text
maxline = 80
for source in glob('*.txt'):
    with open(source) as f:
        fulltext = f.read()
    
    words = fulltext.split()
    cProfile.run('print_neatly(words, maxline)')
    (cost, text) = print_neatly(words, maxline)
    
    #double check the cost
    lines = text.split('\n')
    truecost = 0
    badlines = 0
    for line in lines[0:-1]:
        if len(line) > maxline: 
            badlines = badlines + 1
        elif line[-1] == ' ':
            badlines = badlines + 1
            line = line.rstrip().strip()
        truecost += (maxline - len(line))**3
        
        
    #print the output and cost
    print >>log, '----------------------'
    print >>log, source
    print >>log, '----------------------'
    print >>log, text
    print >>log, '----------------------'
    print >>log, 'cost = ', cost
    print >>log, 'true cost = ', truecost
    print >>log, 'bad lines = ', badlines
    print >>log, '----------------------'


log.close()

print "Done"
