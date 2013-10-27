# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 10:45:15 2013

@author: alexandersedgwick
"""

# Import required libraries

import sys

# Start a counter and store the textfile in memory

ages=[]

clicks_m_si = []
clicks_m_so = []
clicks_f_si = []
clicks_f_so = []

impressions_m_si =[]
impressions_m_so =[]
impressions_f_si =[]
impressions_f_so =[]


lines = sys.stdin.readlines()

#Remember pop removes the element from lines
lines.pop(0)

#For each line find the sum of the index 2 in the list.


