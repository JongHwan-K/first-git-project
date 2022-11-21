#!/usr/bin/env python
# coding: utf-8

# In[58]:


#1. LR(Linear Regression)
#선형 회귀의 문제 : 아래 x와 y를 잘 fit하는 직선을 구하라

import numpy as np
import matplotlib.pyplot as plt

#주어진 데이터 포인트
x = [150, 160, 170, 175, 185]
y = [55, 70, 64, 80, 75]

#무작위로 정한 a와 b가 만들어내는 직선
a = 0.4
b = -35
x_plot = np.linspace(140, 190, 100)
y_plot = a*x_plot + b

plt.plot(x, y, 'o')
plt.plot(x_plot, y_plot)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.grid()
plt.title("Distribution of Data")


# In[59]:


#a와 b의 후보들 사이에서 최적의 a, b를 구하기

a_candidates = np.linspace(0.3, 0.7, 100)
b_candidates = np.linspace(-50, -10, 100)

A, B = np.meshgrid(a_candidates, b_candidates)
L = np.zeros_like(A)

for xi, yi in zip(x, y):
    L += (yi - (A*xi+B))**2


# In[60]:


a_opt = A[L == np.min(L)][0]
b_opt = B[L == np.min(L)][0]
print("Minimal value of L : ", np.min(L))
print("Optimal value of a : ", a_opt)
print("Optimal value of b : ", b_opt)


# In[61]:


new_y_plot = a_opt*x_plot+b_opt

plt.plot(x, y, 'o')
plt.plot(x_plot, new_y_plot)


# In[62]:


#2. GD(Gradient Descent)

#선형 회귀의 문제 : 아래 x와 y를 잘 fit하는 직선을 구하라

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import torch
import random

#주어진 데이터 포인트
x = [150, 160, 170, 175, 185]
y = [55, 70, 64, 80, 75]

#무작위로 정한 a와 b가 만들어내는 직선
a = torch.tensor(0.45, requires_grad = True)
b = torch.tensor(-35.0, requires_grad = True)
x_plot = np.linspace(140, 190, 100)
y_plot = (a.detach())*x_plot + b.detach()

plt.plot(x, y, 'o')
plt.plot(x_plot, y_plot)
plt.xlabel("Height")
plt.ylabel("Weight")


# In[63]:


#기본 파라미터 설정
LR = 1e-6
EPOCH = 10
a_history = [a.detach().item()]
b_history = [b.detach().item()]
l_history = []


#학습
for _ in range(EPOCH):
    a.requires_grad = True
    b.requires_grad = True
    
    Loss = 0
    for xi, yi in zip(x, y):
        Loss += (yi - (a*xi+b))**2 #Loss 계산 하기
    l_history.append(Loss.detach().item())

    Loss.backward() #Loss를 a와 b 각각에 대해 미분하는 과정

    a = a.detach() - LR*a.grad #gradient descent
    b = b.detach() - LR*b.grad #gradient descent
    
    a_history.append(a.item()) #a의 변화 양상을 보기 위해 a_history에 a 저장
    b_history.append(b.item()) #b의 변화 양상을 보기 위해 b_history에 b 저장
    
    plt.figure()
    plt.plot(x, y, 'o')
    plt.xlabel("Height")
    plt.ylabel("Weight")
    new_y_plot = a*x_plot + b
    plt.plot(x_plot, new_y_plot, 'r')
l_history.append(Loss.detach().item())


# In[64]:


#a,b 값에 따른 Loss의 Contour 관찰 1

a_candidates = np.linspace(0.3, 0.7, 100)
b_candidates = np.linspace(-50, -10, 100)

A, B = np.meshgrid(a_candidates, b_candidates)
L = np.zeros_like(A)

for xi, yi in zip(x, y):
    L += (yi - (A*xi + B))**2

plt.contour(a_candidates, b_candidates, L, 20)
plt.xlabel("a")
plt.ylabel("b")
plt.grid()
plt.plot(a_history, b_history, 'r*--', markersize = 10)


# In[65]:


#a,b 값에 따른 Loss의 Contour 관찰 2

plt.figure(figsize = [10, 9])
ax = plt.axes(projection = '3d')
ax.view_init(elev = 90, azim = -45)
ax.plot_surface(A, B, L)
ax.plot(a_history, b_history, l_history, 'r*--', markersize = 10)

plt.figure(figsize = [10, 9])
ax = plt.axes(projection = '3d')
ax.plot_wireframe(A, B, L)
ax.plot(a_history, b_history, l_history, 'r*--', markersize = 10)


# In[66]:


#a, b 값에 따른 Loss의 contour 관찰 3

fig = go.Figure(data = [go.Surface(x=a_candidates, y=b_candidates, z=L, colorscale = 'viridis',  opacity = 0.5)])
fig.update_traces(contours_z = dict(show=True, usecolormap = True, highlightcolor = 'limegreen', project_z =True ))
fig.add_trace(go.Scatter3d(x=a_history, y=b_history, z=l_history, marker=dict(size=3, symbol = "circle", color='rgb(217, 217, 217)',
colorscale='Jet',line=dict(width=0.0))))
fig.update_layout(title = 'The surface of the loss', width = 700, height = 600,
                 scene = dict(zaxis = dict(nticks = 20, range = [0,20000])) )


# In[67]:


#3. SGD(Stochastic Gradient Descent)

#선형 회귀의 문제 : 아래 x와 y를 잘 fit하는 직선을 구하라

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import random
import torch

#주어진 데이터 포인트
x = [150, 160, 170, 175, 185]
y = [55, 70, 64, 80, 75]

#무작위로 만든 직선
a = torch.tensor(0.45, requires_grad = True)
b = torch.tensor(-35.0, requires_grad = True)

x_plot = np.linspace(140, 190, 100)
y_plot = (a.detach().item())*x_plot + (b.detach().item())

plt.plot(x, y, 'o')
plt.plot(x_plot, y_plot)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.grid()


# In[68]:


#기본 파라미터 설정

EPOCH = 10
LR = 1e-6
a_history = [a.detach().item()]
b_history = [b.detach().item()]
l_history = []

for _ in range(EPOCH):
    shuffle_idx = list(range(len(x)))
    random.shuffle(shuffle_idx)
    
    for i in shuffle_idx:
        a.requires_grad = True
        b.requires_grad = True

        loss = (y[i] - (a*x[i]+b))**2

        loss.backward()

        a = a.detach() - LR* a.grad
        b = b.detach() - LR* b.grad

        a_history.append(a.item())
        b_history.append(b.item())
        l_history.append(loss.detach().item())
    
    plt.figure()
    plt.plot(x, y, 'o')
    new_y_plot = (a.item())*x_plot + (b.item())
    plt.plot(x_plot, new_y_plot, 'r')
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.grid()
l_history.append(loss.detach().item())
    


# In[69]:


#a, b 값에 따른 loss의 contour 관찰 1

plt.figure()
a = np.linspace(0.3, 0.7, 100)
b = np.linspace(-50, -10, 100)
A, B = np.meshgrid(a, b)
L = np.zeros_like(A)
for xi, yi in zip(x, y):
    L += (yi - (A*xi+B))**2
plt.contour(A, B, L, 20)
plt.xlabel("a")
plt.ylabel("b")
plt.grid()
plt.plot(a_history, b_history, 'r*--', markersize =10)


# In[72]:


#a, b 값에 따른 loss의 contour 관찰 2

plt.figure(figsize = [10,9])
ax = plt.axes(projection = '3d')
ax.plot_surface(A, B, L)


plt.figure(figsize = [10,9])
ax = plt.axes(projection = '3d')
ax.plot_wireframe(A,B,L)
ax.plot(a_history, b_history, l_history, 'r*--')

plt.figure(figsize = [10,9])
ax = plt.axes(projection = '3d')
ax.plot_wireframe(A,B,L)
ax.plot(a_history, b_history, l_history, 'r*--')
ax.view_init(elev = 0, azim = -90)

plt.figure(figsize = [10,9])
ax = plt.axes(projection = '3d')
ax.plot_wireframe(A,B,L)
ax.plot(a_history, b_history, l_history, 'r*--')
ax.view_init(elev = 90, azim = 0)


# In[71]:


#a, b 값에 따른 loss의 contour 관찰 3

fig = go.Figure(data = [go.Surface(x=a_candidates, y=b_candidates, z=L, colorscale = 'viridis',  opacity = 0.5)])
fig.update_traces(contours_z = dict(show=True, usecolormap = True, highlightcolor = 'limegreen', project_z =True ))
fig.add_trace(go.Scatter3d(x=a_history, y=b_history, z=l_history, marker=dict(size=3, symbol = "circle", color='rgb(217, 217, 217)',
colorscale='Jet',line=dict(width=0.0))))
fig.update_layout(title = 'The surface of the loss', width = 700, height = 600,
                 scene = dict(zaxis = dict(nticks = 20, range = [0,20000])) )

