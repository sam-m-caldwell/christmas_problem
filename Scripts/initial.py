# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:54:49 2017

@author: Sam Caldwell
"""

#==============================================================================
# Libraries
#==============================================================================

import pandas as pd
import numpy as np
import datetime as dt
import os
import pulp as pp
os.chdir('C:\\Users\\Sam Caldwell\\Desktop\\kaggle')

test = False
#==============================================================================
# Read data
#==============================================================================

if test:
    int_gifts_max = 2
    
    ar_child_wishlist = pd.read_csv(
        "Data/child_wishlist_test.csv",
        #"Data/child_wishlist.csv",
        header=None
    ).drop(0, 1).values
    
    ar_gift_wishlist = pd.read_csv(
        "Data/child_wishlist_test.csv",
        #"Data/gift_goodkids.csv",
        header=None
    ).drop(0, 1).values
            
    ls_twins = [(0,1)]
else:
    int_gifts_max = 1000
    
    ar_child_wishlist = pd.read_csv(
        #"Data/child_wishlist_test.csv",
        "Data/child_wishlist.csv",
        header=None
    ).drop(0, 1).values
    
    ar_gift_wishlist = pd.read_csv(
        #"Data/child_wishlist_test.csv",
        "Data/gift_goodkids.csv",
        header=None
    ).drop(0, 1).values
            
    ls_twins = [(2*k,2*k+1) for k in range(int(0.004*len(ar_child_wishlist)/2))]

#==============================================================================
# Assign parameters
#==============================================================================
int_gifts = len(ar_gift_wishlist)
int_kids = len(ar_child_wishlist)
int_kid_pref = ar_child_wishlist.shape[1]
int_gift_pref = ar_gift_wishlist.shape[1]
int_rel_happiness = (int_kids/int_gifts)**2

int_ratio = 2
int_rel_happiness = (int_kids/int_gifts)**2

ls_kids = range(int_kids)
#ls_twins = range(int(0.004*int_kids))

ls_gifts = range(int_gifts)
int_max_kid_happy = int_kids * int_ratio
int_max_gift_happy = int_gifts * int_ratio

int_gift_pref = ar_gift_wishlist.shape[0]
int_child_pref = ar_gift_wishlist.shape[1]

#==============================================================================
# Enumerate benefits
#==============================================================================

#dict_obj_child = {
#    (k, g) : 
#    2*(10 - sum(ar_child_wishlist[k] == g))/int_max_kid_happy
#    for g in ls_gifts
#    for k in ls_kids}
#    
#dict_obj_gifts = {
#    (k, g) : 
#    -1 if g not in ar_gift_wishlist[k] 
#    else 2*(10 - sum(ar_gift_wishlist[k] == g))
#    for g in ls_gifts
#    for k in ls_kids}
    
def obj_child(k, g) :
    child_happiness = (int_gift_pref - np.where(ar_child_wishlist[k]==g)[0])
    if not child_happiness:
        child_happiness = -1
    return int(child_happiness)

def obj_gifts(k, g) :
    gift_happiness = (int_child_pref - np.where(ar_gift_wishlist[g]==k)[0]) * int_rel_happiness
    if not gift_happiness:
        gift_happiness = -1 * int_rel_happiness
    return int(gift_happiness)

ls_set = [(k, g) for k in ls_kids for g in ar_child_wishlist[k]]

dict_obj = {
    (k,g) :
    int(int_gift_pref - np.where(ar_child_wishlist[k]==g)[0]) + \
    int(int_child_pref - np.where(ar_gift_wishlist[g]==k)[0]) * int_rel_happiness
    for (k, g) in ls_set
}
    
#==============================================================================
# Define problem
#==============================================================================
    
prob = pp.LpProblem("Santa optimisation", pp.LpMaximize)

# Variable
cat = pp.LpVariable.dicts(
        'sel', ls_set, 
        lowBound = 0, upBound = 1, cat = 'Integer')

# Objective
#prob += pp.lpSum([
#        (obj_child(k, g)+obj_gifts(k, g))*cat[k,g]  
#        #for g in ls_gifts for k in ls_kids])
#        for k in ls_kids for g in ar_child_wishlist[k]])
prob += pp.lpSum([dict_obj[k,g]*cat[k,g] for (k, g) in ls_set])

# Constraint 1 - one gift per child
for k in ls_kids: 
    prob += pp.lpSum([cat[k, g] for g in ls_gifts if (k,g) in ls_set]) == 1
    
# Constraint 2 - Only 1000 of each gift
for g in ls_gifts: 
    prob += pp.lpSum([cat[k, g] for k in ls_kids if (k,g) in ls_set]) <= int_gifts_max
    
# Constraint 3 - twins have same gifts
for k in ls_twins:
    for g in ls_gifts:
        if (k,g) in ls_set:
            prob += cat[k[0], g] - cat[k[1], g] == 0

prob.solve()
print pp.LpStatus[prob.status]
yo = {(k, g): cat[k,g].varValue for (k, g) in ls_set}
#var_obj = sum([
#    (obj_child(k, g) +  obj_gifts(k, g))*cat[k,g].varValue
#    for g in ls_gifts for k in ls_kids])
var_obj = sum([
    (dict_obj(k,g))*cat[k,g].varValue
    for (k, g) in ls_set])