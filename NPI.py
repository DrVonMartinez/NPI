# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:14:18 2019

@author: Benjamin
"""

import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as op
from torch.nn.functional import relu,softmax
import scipy.stats as stats
import matplotlib.pyplot as plt
import card


class NPI(nn.Module):
    def __init__(self,alpha, b,s, table_shape, card_shape,M_prog,M_key):
        super(NPI, self).__init__()
        self.vector_shape = table_shape[0] *table_shape[1]*card_shape[0] *card_shape[1]
        self.alpha = alpha
        self.M_prog = M_prog
        self.M_key = M_key#            e                args[0]         args[1:10]
        self.linear_1 = nn.Linear(self.vector_shape+table_shape[0]*table_shape[1]+10,b)
        self.linear_2 = nn.Linear(b,s)
        self.lstm_1 = nn.LSTM(len(M_prog)+s, 256,batch_first=True)
        #print(len(M_prog)+s)
        self.lstm_2 = nn.LSTM(256, 256)
        self.arg = nn.Linear(256,10+table_shape[0]*table_shape[1])
        self.prog = nn.Linear(256,7)
        self.r = nn.Linear(256,1)
        self.Q = tr.zeros((table_shape[0]*table_shape[1],1))
        self.table_shape =table_shape
        self.table = None
        
    def forward_lstm(self,e,p,args):
        #print(np.shape(e),np.shape(p),np.shape(args))
        #print(p)
        x = np.append(e,args.detach().numpy())
        x=tr.from_numpy(x.reshape(1,x.size))
        linear1 = relu(self.linear_1(x))
        linear2 = relu(self.linear_2(linear1))
        linear2 = linear2.reshape((linear2.size()[1],1))
        #print(linear2.size())
        #temp = linear2.detach().numpy()
        p = p.reshape((p.size()[0],1))
        #print(p.size())
        #p = p.detach().numpy()
        #p.dtype = np.float32
        concat = tr.cat((linear2,p))
        #concat.dtype = np.float32
        concat = concat.reshape((1,1,concat.size()[0]))
        #y = tr.from_numpy(concat)
        #print(y)
        lstm_step_1,_ =self.lstm_1(concat)
        lstm1 = tr.sigmoid(lstm_step_1[-1])
        lstm1 = lstm1.reshape((1,1,lstm1.size()[1]))
        lstm_step_2, _ =self.lstm_2(lstm1)
        lstm2 = tr.sigmoid(lstm_step_2[-1])
        return lstm2
        
    def f_args(self,e,p,args):
        lstm =self.forward_lstm(e,p,args)
        return softmax(self.arg(lstm),dim=1)
    
    def f_prog(self,e,p,args):
        lstm=self.forward_lstm(e,p,args)
        return softmax(self.prog(lstm),dim=1)
    
    def f_end(self,e,p,args):
        lstm =self.forward_lstm(e,p,args)
        return tr.tanh(self.r(lstm))
    
    def f_env(self,e,p,args):
        e = e.detach().numpy()
        args= args.detach().numpy()
        e=e.reshape((e.size,1))
        args= args.reshape((args.size,1))
        args[-6]=self.table_shape[0]-1
        args[-5]=self.table_shape[1]-1
        args2 =np.zeros(args.shape,dtype=np.int32)
        for l in range(args2.size):
            args2[l] = int(args[l])
        #print(args2)
        args2,e = p(args2,e,self.table)
        e2 = np.zeros(e.shape,dtype=np.float32)
        args3 =np.zeros(args.shape,dtype=np.float32)
        for l in range(args3.size):
            args3[l] = args2[l]
        for k in range(e2.size):
            e2[k] = e[k]
        #print(args3)
        e2 = e2.reshape((1,e.size))
        args3= args3.reshape((1,args3.size))
        e= tr.from_numpy(e2)
        args=tr.from_numpy(args3)
        return args,e
    
    def train(self,x, y, num_iters, e):
        learning_rate=0.01
        criterion = nn.MSELoss()
        optimizer = op.SGD(self.parameters(),lr=learning_rate)
        elementOne = np.zeros(self.table_shape,dtype = np.float32)
        #print(self.table_shape[0]-1,self.table_shape[1]-1)
        elementTwo=np.asarray([0,0,0,0,self.table_shape[0]-1,self.table_shape[1]-1,0,0,0,0],dtype =np.float32)
        a = np.append(elementOne,elementTwo)
        args=tr.from_numpy(a)
        for i in range(num_iters):
            optimizer.zero_grad()
            out1 =self.f_args(e,x[i],args)
            out2 =self.f_prog(e,x[i],args)
            out3 =self.f_end(e,x[i],args)
            out1 = out1.reshape(out1.size()[1],1)
            out2 = out2.reshape(out2.size()[1],1)
            out3 = out3.reshape(out3.size()[1],1)
            #print(out1.dim())
            #print(out1,out2,out3)
            #print(out1.size(),out2.size(),out3.size())
            temp =tr.cat((out1,out2,out3))
            #print(temp.size())
            #print(temp.dim())
            train_loss = criterion(softmax(temp,dim = 1), y[i])
            train_loss.backward(retain_graph=True)
            optimizer.step()
        return train_loss.item()
                
    def test(self,x, y, tables, num_iters, verbose=False):
        learning_rate=0.01
        criterion = nn.MSELoss()
        optimizer = op.SGD(self.parameters(),lr=learning_rate)
        for i in range(num_iters):   
            optimizer.zero_grad()
            elementOne = np.zeros(self.table_shape,dtype = np.float32)
            elementTwo=np.asarray([0,0,0,0,0,0,0,0,0,0],dtype =np.float32)
            a = np.append(elementOne,elementTwo)
            #print(np.shape(a))
            args=tr.from_numpy(a)
            #print(args,x[i])
            self.table=tables[i]
            test_loss =criterion(self.run(x[:,i],6,args),y)
            optimizer.step()
        return test_loss.item()
            
            
    def run(self,e,i,args):
        r=0
        p = self.M_prog[i]
        p_i = tr.zeros((len(self.M_prog),1))
        p_i[i] = 1
        while(r<self.alpha):
            #print(args[0])
            e = e.detach().numpy()
            args= args.detach().numpy()
            e=e.reshape((e.size,1))
            args= args.reshape((args.size,1))
            args[-6]=self.table_shape[0]-1
            args[-5]=self.table_shape[1]-1
            args2 =np.zeros(args.shape,dtype=np.int32)
            for l in range(args2.size):
                args2[l] = int(args[l])
            #print(args2)
            args2,e = p(args2,e,self.table)
            e2 = np.zeros(e.shape,dtype=np.float32)
            args3 =np.zeros(args.shape,dtype=np.float32)
            for l in range(args3.size):
                args3[l] = args2[l]
            for k in range(e2.size):
                e2[k] = e[k]
            #print(args3)
            e2 = e2.reshape((1,e.size))
            args3= args3.reshape((1,args3.size))
            e= tr.from_numpy(e2)
            args=tr.from_numpy(args3)
            #Feed-Forward
            #h = self.forward(e,p_i,h,args)
            #temp = self.forward_lstm(e,p_i,h,args)
            args = self.f_args(e,p_i,args) #Size 10
            k = self.f_prog(e,p_i,args) #Size 7
            r = self.f_end(e,p_i,args) #Size 1
            #Decide Next to Run
            k= k.detach().numpy()
            k=k.reshape((k.size,1))
            i2 = np.argmax(np.matmul(self.M_key[:],k))
            #print('i2',i2)
            if i2 == 0:#ACT
                e, args = self.f_env(e,p,args)
            else:
                self.run(e,i2,args)
        return
    
    def equals(self,q,actual):
        for n in self.table_shape[0]:
            for m in self.table_shape[1]:
                if q[n][m] != actual[n][m]:
                    return False
        return True


def card_matching(m,n,a,b,alpha,beta,sigma,num_test,train_iters):
    print('('+str(m)+','+str(n)+') ('+str(a)+'x'+str(b)+')')
    M_prog, M_key = card.Load_M()
    net = NPI(alpha,beta,sigma,(m,n),(a,b),M_prog,M_key)
    train_x,train_y,e = card.generate_training_data((m,n),(a,b),train_iters)
    test_x,test_y,tables =card.generate_test_data((m,n),(a,b),num_test)
    training_losses = net.train(train_x, train_y, len(M_prog)*train_iters,e)
    test_losses = net.test(test_x,test_y,tables,num_test)
    slope, intercept, _, _, _ = stats.linregress(range(train_iters), training_losses)
    plt.plot(training_losses)
    plt.show()
    plt.close('all')
    slope, intercept, _, _, _ = stats.linregress(range(num_test), test_losses)
    plt.plot(test_losses)
    plt.plot(np.arange(num_test)*slope + intercept)
    plt.show()
    

if __name__ == "__main__":
    m=5
    n=4
    a = 2
    b = 2
    alpha =0.5
    beta=50
    sigma =25
    num_test = 20
    train_iters=10
    card_matching(m,n,a,b,alpha,beta,sigma,num_test,train_iters)
