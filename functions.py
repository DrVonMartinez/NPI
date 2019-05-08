# -*- coding: utf-8 -*
"""
@author: bknightw
"""
#functions
#args[0]  = Q Solution Table Q0 = 0s
#args[1]  = ptr1 default = 0
#args[2]  = ptr1 default = 0
#args[3]  = ptr2 default 0
#args[4]  = ptr2 default 0
#args[5]  = m-1 CONSTANT Y-axis
#args[6]  = n-1 CONSTANT X-axis
#args[7]  = ptr #1,2 
#args[8]  = conditional (int)
#args[9]  = direction 0=up,1=right,2=down,3=left
#args[10] = current count
#E current table

def card_matching(args,E,Card_Table):
    table = (args[-5]+1)*(args[-6]+1)-1
    #print(table,type(table))
    '''
    card matching:
        ptr1 = position0
        while ptr1 != table end:
            ptr1 = match ptr1
        ###For Solvability###
        !increment ptr1
    '''
    print('card_matching')
    args[table+1] = 0 #Top Left Corner
    args[table+2] = 0
    args[table+10] = 0
    args[table+8] = 0
    while args[-2]==0:
        args,E = match(args,E,Card_Table)
    return args,E


def match(args,E,Card_Table):
    table = (args[-5]+1)*(args[-6]+1)-1
    '''            
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
    '''
    print('match')
    args[-3] = 1 #ptr1
    args[table+8] = 0 #Nothing Removed Yet
    increment(args,E,Card_Table)
    args[table+3] = args[table+1]
    args[table+4] = args[table+2]
    #print(args[2])
    while args[-2] ==0:    
        args[-3] = 2 #ptr2
        args,E = increment(args,E,Card_Table)
        args,E = compare (args,E,Card_Table)
    return args,E 

def increment(args,E,Card_Table):
    table = (args[-5]+1)*(args[-6]+1)-1
    '''
    increment ptr
        while ptr is empty and not at end of table
            move position ptr right
            ###For Solvability###
            if end of table == empty
                return False
            return True
            #####################
    '''
    print('increment')
    val1=args[table+2*int(args[-3])]
    val2=args[table+2*int(args[-3])]
    val3 = args[val1+val2]
    #print(val1,val2,val3)
    while  args[val1+val2]!=0:#These cards have been assigned so the table here is empty
        args,E = move_col_right(args,E,Card_Table)
        val1=args[table+2*int(args[-3])]
        val2=args[table+2*int(args[-3])]
        #print(args)
    return args,E

def compare (args,E,Card_Table):
    table = int((args[-5]+1)*(args[-6]+1)-1)
    cardSize = int(E.size/table)
    '''
    compare (ptr1 card,ptr2 card)
        flip cards up (ptr1, ptr2)
        if ptr1 card == ptr2 card
            remove cards (ptr1 card)
            return True
        else
            flip cards down (ptr1,ptr2)
            return False
    '''
    print('compare')
    print('flip')
    #print(args[table+1])
    card1 = int(args[table+1]+args[table+2])
    card2 = int(args[table+3] + args[table+4])
    ptr1 =Card_Table[card1]
    ptr1.turn_over()
    ptr2 =Card_Table[card2]
    ptr2.turn_over()
    if ptr1.equals(ptr2):
        print('remove')
        ptr1.remove_card()
        ptr2.remove_card()
        E[card1:card1+cardSize]=0
        E[card2:card2+cardSize] =0
        args[card1] = args[-1]
        args[card2] = args[-1]
        args[-1]+=1
        args[-3] = 1 #card removed
    else:
        print('flip')
        ptr1.turn_over()
        ptr2.turn_over()     
    return args,E

def move_row_down(args,E,Card_Table):
    table = (args[-5]+1)*(args[-6]+1)-1
    '''
    move_row_down ptr
        move ptr down
        while not at row start
            move ptr left
    '''
    print('move ptr'+args[-3]+' row down')
    #0=up,1=right,2=down,3=left
    #args[8] = args[2*args[7]-1] == args[5]
    #if args[8] ==0: #Then We are not at the bottom row
    args[table+9] = 2 #down
    move_ptr(args,E,Card_Table)
        #args[8] = args[2*args[7]] == args[6]
        #if args[8] ==0: #Then we can go further left
            #args[9] = 3 #left
    #while args[8] ==0:
         #args,E = move_ptr(args,E)
    #args[8] = 0
    return args,E    

def move_col_right(args,E,Card_Table):
    table = (args[-5]+1)*(args[-6]+1)-1
    '''
    move_position ptr right
        if not at end of row
            move ptr right
        else
            move_row_down ptr
    '''
    print('move ptr'+str(args[-4]) +' col right')
    #0=up,1=right,2=down,3=left
    if args[table+2*int(args[-3])] == args[table+6]: #End of Row
        args,E = move_row_down(args,E,Card_Table)
    else: #Not End of Row
        args[table+9] = 1
        args,E = move_ptr(args,E,Card_Table)
    return args,E

def move_ptr(args,E,Card_Table):
    table = (args[-5]+1)*(args[-6]+1)-1
    direction = args[table+9]
    if args[table+9]%2 ==0:
        modified = args[table+2*int(args[-3])-1]
    else:
        modified = args[table+2*int(args[-3])]
    if direction==0:
        modified = modified-1
        args[table+8]= int(modified == 0)
        word = 'up'
    elif direction==1:
        modified = modified+1
        args[table+8]= int(modified == args[table+5])
        word = 'right'
    elif direction==2:
        modified = modified+1
        args[table+2*int(args[-3])-1]=0
        args[table+8]= int(modified == args[table+6])
        word = 'down'
    else:
        modified = modified-1
        args[table+8]= int(modified == 0)
        word = 'left'
    if args[table+9]%2 ==0:
        modified = args[table+2*int(args[-3])-1]
    else:
        modified = args[table+2*int(args[-3])]
    print('move ptr'+str(args[-3])+' '+word)
    return args,E 