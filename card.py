# -*- coding: utf-8 -*-
"""
Created on Wed May  1 22:53:51 2019

@author: Benjamin
"""
import numpy as np
import random
import torch as tr
import functions
import table_helper as th

class card:
    def __init__(self,shape):
        start = np.random.uniform(0.1,0.9,shape)
        
        start.dtype  = np.float32
        #start = start.reshape(shape)
        self.face_up = start
        self.face_down = np.ones(shape,dtype=np.float32)
        self.current = self.face_up
        self.shape = shape
    
    def turn_over(self):
        if(self.is_face_up):
            self.current = np.copy(self.face_down)
        else:
            self.current = np.copy(self.face_up)

    def is_face_up(self):
        if self.average()==0 or self.average() ==1:
            return False
        for m in range(self.shape[0]*self.shape[1]):
            if self.current[m] != self.face_up[m]:
                return False
        return True
    
    def is_face_down(self):
        return self.average()==1
    
    def vectorize(self):
        self.current.shape = (self.current.shape[0]*self.current.shape[1],1)
        return self.current
    
    def show(self):
        return self.current
    
    
    def average(self):
        return np.average(self.current)
    
    def copy(self,other):
        self.current = np.copy(other.current)
        self.face_down = np.copy(other.face_down)
        self.face_up = np.copy(other.face_up)
    
    def equals(self,other):
        for m in range(other.shape[0]):
            for n in range(other.shape[1]):
                if self.current[m][n] != other.current[m][n]:
                    return False
        return True
    
    def remove_card(self):
        self.face_up = np.zeros(self.shape,dtype=np.float32)
        self.face_down = np.zeros(self.shape,dtype=np.float32)
        self.current = self.face_up
    
    def is_empty(self):
        return self.average()==0
   

def Load_M():
    M_table = []
    M_table.append(functions.move_ptr)
    M_table.append(functions.move_col_right)
    M_table.append(functions.move_row_down)
    M_table.append(functions.compare)
    M_table.append(functions.increment)
    M_table.append(functions.match)
    M_table.append(functions.card_matching)
    M_k = np.identity(len(M_table))
    return M_table, M_k
            
def create_matching_table(table_shape,card_shape):
    empty_table = np.random.uniform(0,1)
    if empty_table>.95:
        return create_empty_table(table_shape,card_shape)
    m,n = table_shape
    #we need m*n/2 cards
    table = []
    for l in range(int(m*n/2)):
        empty_slot = np.random.uniform(0,1)
        #print(empty_slot)
        new_card =card(card_shape)
        if empty_slot>0.9:
            new_card.face_down = np.zeros(card_shape,dtype=np.float32)
            new_card.face_up = np.zeros(card_shape,dtype=np.float32)
            new_card.current = np.zeros(card_shape,dtype=np.float32)
        a= card(card_shape)
        a.copy(new_card)
        table.append(new_card)
        table.append(a)#The matched card
    card_table = table
    table = np.asarray(table)
    random.shuffle(table)
    classify = th.classify_table(table,m,n)
    th.turn_over(table)
    table.shape = (m,n)
    return table,classify,card_table

def create_empty_table(table_shape,card_shape):
        #we need m*n/2 cards
    m,n = table_shape
    table = []
    for l in range(int(m*n/2)):
        new_card =card(card_shape)
        new_card.remove_card()
        a= card(card_shape)
        a.copy(new_card)
        table.append(new_card)
        table.append(a)#The matched card
    card_table = table
    table = np.asarray(table)
    classify = np.zeros(np.shape(table),dtype=np.float32)     
    table.shape = (m,n)
    classify.shape = (m,n)
    return table,classify,card_table

def generate_training_data(table_size,card_size,batch_size):
    '''
    Assumptions:
        All Card Tables are valid. This means that every combination will solvable
        All Card Tables start Face Down. This allows the algorithm to start in a uniform spot
    This will create training examples with classifications for:
        States:
            empty table vs cards on table
            card match vs card not match
            card match vs empty space
            card match vs face down
        Functions
            card match:
                ptr1 = position0
                while ptr1 != table end:
                    ptr1 = match ptr1
                ###For Solvability###
                !increment ptr1
                
            match ptr1:
                increment ptr1
                ptr2 = ptr1 position
                if ptr1 not at table end:
                    do:
                        ###For Solvability###
                        valid = increment ptr2
                        if not valid:
                            return ptr1
                        #####################
                        increment ptr2
                        removed = compare (ptr1.card, ptr2.card)
                    while not removed
                return ptr1
                
            move_row_down ptr
                move ptr down
                move ptr left

            move_col_right ptr
                if not at end of row
                    move ptr right
                else
                    move_row_down ptr
                    
            increment ptr
                while ptr is empty and not at end of table
                   move position ptr right
                  ###For Solvability###
                   if end of table == empty
                       return False
                   return True
            
                   
            compare (ptr1 card,ptr2 card)
                flip cards up (ptr1, ptr2)
                if ptr1 card == ptr2 card
                    remove cards (ptr1 card)
                    return True
                else
                    flip cards down (ptr1,ptr2)
                    return False
            
        Actions:
            flip (ptr1,ptr2)
            move ptr (ptr, direction)
            remove cards(ptr1,ptr2)
    '''
    table = table_size[0]*table_size[1]
    a, _, _= create_matching_table(table_size,card_size)
    e = tr.from_numpy(th.vectorize_card(a))
    M_prog,_ = Load_M()
    generate = []
    #print(len(M_prog))
    p = np.zeros((len(M_prog),1),dtype = np.float32)
    #train cases
    for i in range(batch_size):
        #move ptr
        args=np.zeros(shape= (table+10,1))
        p[0] =1
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        generate.append((p,p_2))
    ###################
    p = np.zeros((len(M_prog),1),dtype = np.float32)
    for i in range(int(batch_size/2)):
        p = np.zeros((len(M_prog),1),dtype = np.float32)
        #move col right
        args=np.zeros(shape= (table+10,1))
        p[1] = 1
        args[table+4] = int(np.random.uniform(0,table_size[0]-1))
        args[table+6] = int(np.random.uniform(0,table_size[0]-1))
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        if args[table+4] == args[table+6]:  #end of row
            p_2[2] = 1
        else:
            p_2[0] = 1
        generate.append((p,p_2))
    p = np.zeros((len(M_prog),1),dtype = np.float32)
    for i in range(int(batch_size/2)):
        #move col right
        args=np.zeros(shape= (table+10,1))
        p[1] = 1
        args[table+2] = int(np.random.uniform(0,table_size[0]-1))
        args[table+6] = int(np.random.uniform(0,table_size[0]-1))
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        if args[table+4] == args[table+6]:  #end of row
            p_2[2] = 1
        else:
            p_2[0] = 1
        generate.append((p,p_2))    
    ######################
    for i in range(int(batch_size/2)):
        #move row down
        p[2] = 1
        args=np.zeros(shape= (table+10,1))
        args[table+1] = int(np.random.uniform(0,table_size[1]-1))
        args[table+5] = int(np.random.uniform(0,table_size[1]-1))
        args[table+8] = 1
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        if args[table+1] != args[table+5]: #end of column
            p_2[0] = 1
        generate.append((p,p_2))
        args=np.zeros(shape= (table+10,1))
        args[table+3] = int(np.random.uniform(0,table_size[1]-1))
        args[table+5] = int(np.random.uniform(0,table_size[1]-1))
        args[table+8] = 1
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        if args[table+3] != args[table+5]: #end of column
            p_2[0] = 1
        generate.append((p,p_2))
    p = np.zeros((len(M_prog),1),dtype = np.float32)
    ##########################
    args=np.zeros(shape= (table+10,1))
    p = np.zeros((len(M_prog),1),dtype = np.float32)
    for i in range(batch_size):
        #compare
        p[3] =1
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        generate.append((p,p_2))
    ##########################
    args=np.zeros(shape= (table+10,1))
    p = np.zeros((len(M_prog),1),dtype = np.float32)
    for i in range(batch_size):
        #increment ptr
        p[4] =1
        args=np.zeros(shape= (table+10,1))
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        if args[int(args[table+1]+args[table+2])] !=0:
            p_2[4]=1
        generate.append((p,p_2))
        
        args=np.zeros(shape= (table+10,1))
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        if args[int(args[table+3]+args[table+4])] !=0:
            p_2[4]=1
        generate.append((p,p_2))
        
        args=np.zeros(shape= (table+10,1))
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        if args[table+7] ==2:
            p_2[3]=1
        generate.append((p,p_2))
    ##########################
    p = np.zeros((len(M_prog),1),dtype = np.float32)
    for i in range(batch_size):
        #match
        args=np.zeros(shape= (table+10,1))
        p[5] =1
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        if args[table+7] ==1 or args[table+7] ==2:
            p_2[4]=1
        generate.append((p,p_2))
    ##########################
    p = np.zeros((len(M_prog),1),dtype = np.float32)
    for i in range(batch_size):
        #card_match
        args=np.zeros(shape= (table+10,1))
        p[6] =1
        p_2 = np.zeros((len(M_prog),1),dtype = np.float32)
        if args[table+8] !=1:
            p_2[5]=1
        generate.append((p,p_2))
                
    random.shuffle(generate)
    training_cases=[]
    correct = []
    generate = generate[:batch_size*len(M_prog)]
    for g in generate:
        training_cases.append(g[0])
        correct.append(g[1])
    x = np.asarray(training_cases,dtype = np.float32)
    y = np.asarray(correct, dtype = np.float32)
    x_tr = tr.from_numpy(x.reshape((batch_size*len(M_prog),len(M_prog))))
    y_tr = tr.from_numpy(y.reshape(batch_size*len(M_prog),len(M_prog)))
    return x_tr,y_tr,e
    

#Generates batch_size full table
def generate_test_data(table_shape,card_shape,batch_size):
    table=[]
    '''
    This creates a batch_size set of the full table with random position assignments
    and with after a random number of steps have occurred (x cards have been matched)
    '''
    y_shape = table_shape[0]*table_shape[1]
    x_shape = y_shape*card_shape[0] *card_shape[1]
    #print(x_shape,y_shape)
    #create_matching_table
    X = np.asarray([],dtype =np.float32)
    Y = np.asarray([],dtype =np.float32)
    for i in range(batch_size):
        x,y,c =create_matching_table(table_shape,card_shape)
        table.append(c)
        X=np.append(th.vectorize_card(x),X)
        Y=np.append(th.vectorize(y),Y)
        X = X.reshape((x_shape,i+1))
        Y = Y.reshape((y_shape,i+1))
    return tr.from_numpy(X),tr.from_numpy(Y),table
    
if __name__ == "__main__":
    m,n,a,b = 4,4,2,2
    batch_size = 100
    X_,Y_,e =generate_training_data((m,n),(a,b),batch_size)
    print('Table Size: '+str(m)+'x'+str(n),'\nCard shape: ('+str(a)+','+str(b)+')')
    print('Training Data:',np.shape(X_),np.shape(Y_))
    X_,Y_ =generate_test_data((m,n),(a,b),batch_size)
    print('Table Size: '+str(m)+'x'+str(n),'\nCard shape: ('+str(a)+','+str(b)+')')
    print('Test Data:',np.shape(X_),np.shape(Y_))

