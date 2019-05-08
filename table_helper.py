# -*- coding: utf-8 -*-
"""
@author: bknightw
"""
#table_helper
import numpy as np

def vectorize_card(table):
    new = np.asarray([],dtype = np.float32)
    for m in range(table.shape[0]):
        for n in range(table.shape[1]):
            temp = table[m,n].vectorize()
            new = np.append(temp,new)
    new.shape =(new.size,1)
    return new

def vectorize(table):
    table.shape =(table.shape[0]*table.shape[1],1)
    return table

def turn_over(table, all_face_down = True):
    count=0
    for f in range(table.size):
        if all_face_down:
            turn =1
        else:
            turn = np.random.uniform(0,1)
        if turn>0.05:
            table[f].turn_over()
        else:
            count+=1
            
def show_table(table_shape, table):
    m,n = table_shape
    elements= []
    for i in range(m):
        for j in range(n):
            elements.append(table[i,j].show())
        print( elements[i*j:i*j+j+1],)
        print()
    val =np.asarray(elements).reshape(m,n,table[0][0].shape[0],table[0][0].shape[1])
    print(np.shape(val))
    return np.asarray(elements).reshape(m,n,table[0][0].shape[0],table[0][0].shape[1])

def classify_table(table,m,n):
    classify = np.zeros(np.shape(table),dtype = np.float32)     
    count =1
    for a in range(table.size):
        if table[a].average() == 0:
            classify[a]=0
            continue
        elif classify[a] != 0:
            continue
        else:
            classify[a]=count
            for b in range(a+1,table.size):
                if table[a].equals(table[b]) and table[a].average()!=0:
                    classify[b] = classify[a]
                    count+=1
                    continue
    classify.shape = (m,n)
    return classify