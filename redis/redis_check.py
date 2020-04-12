#!/usr/bin/env python
# coding: utf-8

# In[1]:


#from time import time
import numpy as np
import matplotlib.pyplot as plt


# # Establish redis connection

# In[27]:


import redis

if __name__ == "__main__":
    r = redis.Redis(host="192.168.99.100",port=6379)
    
    r.set("check","connection")
    print("Check ",r.get("check"))


# In[28]:


r = redis.Redis(host="192.168.99.100",port=6379)


# # Test job size vs job time

# In[38]:


# clear keys
for i in range(N):
    r.delete("{0:015b}".format(i))


# In[39]:


N = 10000

memory_set = {}
memory_get = {}

time_table_set = {}
time_table_get = {}

time_complete_set = {}
time_complete_get = {}



# memory issues around size 256 ... 
# 9 
array_sizes = 2**np.arange(9)
for size in array_sizes:
    
    print("size {}".format(size**2 * 8))
    t_set = []
    t_get = []
    t_set_complete = []
    t_get_complete =[]
        
    key = "{0:015b}".format(1)
    for i in range(N):

        # set
        x = np.random.uniform(0,1,size=(size,size))
        
        t_start = r.time()
        r.set(key,x.tobytes())
        t_end = r.time()
        
        job_time = (t_end[0] + t_end[1]*1e-6)-(t_start[0] + t_start[1]*1e-6)
        t_set.append(job_time)
        t_set_complete.append(t_end[0] + t_end[1]*1e-6)

        # get        
        t_start = r.time()
        r.get(key)
        t_end =  r.time()
        
        job_time = (t_end[0] + t_end[1]*1e-6)-(t_start[0] + t_start[1]*1e-6)
        t_get.append(job_time)
        t_get_complete.append(t_end[0] + t_end[1]*1e-6)
        
        # delete key
        r.delete(key)

    
    memory_set[str(size**2 * 8)] = np.mean(t_set)
    time_table_set[str(size**2 * 8)] = t_set
    
    memory_get[str(size**2 * 8)] = np.mean(t_get)
    time_table_get[str(size**2 * 8)] = t_get
    
    time_complete_set[str(size**2 * 8)] = time_complete_set
    time_complete_get[str(size**2 * 8)] = time_complete_get


# In[50]:


plt.plot(np.array(list(memory_set.keys())),np.array(list(memory_set.values())))
plt.plot(np.array(list(memory_get.keys())),np.array(list(memory_get.values())))
plt.xlabel("Memory (bytes)",fontsize=15)
plt.ylabel("Average Time (s)",fontsize=15)
plt.xticks(rotation=-90)
plt.legend(["Set","Get"])


# In[52]:


plt.plot(np.array(list(memory_get.keys())),np.array(list(memory_get.values()))-np.array(list(memory_set.values())))
plt.xlabel("Memory (bytes)",fontsize=15)
plt.xticks(rotation=-90)
plt.ylabel("Average Time (s)",fontsize=15)
plt.title("Get - Set as memory increased",fontsize=20)
plt.show()


# In[61]:


plt.figure(figsize=(10,5))


for size in array_sizes:
    key = size**2 * 8
    
    t = time_table_set[str(key)]

    num_bins = int(np.sqrt(len(t)))#int(np.floor(5/3 * (len(t)**(1/3))))
    counts, bin_edges = np.histogram (t, bins=num_bins, normed=True)
    cdf = np.cumsum (counts)
    #plt.plot (np.hstack((np.zeros((1,)),bin_edges[1:])), np.hstack((np.zeros(1,),cdf/cdf[-1])))
    plt.plot(bin_edges[1:], cdf/cdf[-1])
    plt.xlabel("t (s)",fontsize=15); plt.ylabel("F(t)",fontsize=15);
    
    

plt.xlim([0.0002,0.01])
plt.legend(array_sizes**2 * 8)
plt.title("Set time CDF",fontsize=20)
plt.show() 


# In[67]:


plt.figure(figsize=(10,5))


for size in array_sizes:
    key = size**2 * 8
    
    t = time_table_get[str(key)]

    num_bins = int(np.sqrt(len(t)))#int(np.floor(5/3 * (len(t)**(1/3))))
    counts, bin_edges = np.histogram (t, bins=num_bins, normed=True)
    cdf = np.cumsum (counts)
    #plt.plot (np.hstack((np.zeros((1,)),bin_edges[1:])), np.hstack((np.zeros(1,),cdf/cdf[-1])))
    plt.plot(bin_edges[1:], cdf/cdf[-1])
    plt.xlabel("t (s)",fontsize=15); plt.ylabel("F(t)",fontsize=15);
    
    

plt.xlim([0,0.01])
plt.legend(array_sizes**2 * 8)
plt.title("Get time CDF",fontsize=20)
plt.show()


# In[75]:


# clear keys
for i in range(N):
    r.delete("{0:015b}".format(i))


# #  \#job vs total time

# In[72]:


N = 10000

memory_set = {}
memory_get = {}

time_table_set = {}
time_table_get = {}

time_complete_set = {}
time_complete_get = {}

pipeline = r.pipeline(transaction=True)


job_size = np.array([1,10,100,1000,10000,100000])

for size in job_size:
    
    print("N jobs {}".format(size))
    t_set = []
    t_get = []
    t_set_complete = []
    t_get_complete =[]
    
    # store values
    
    for _ in range(N):
        for i in range(size):
            key = "{0:015b}".format(i)
            x = np.random.uniform(0,1,size=(8,8))
            pipeline.set(key,x.tobytes())

        t_start = r.time()
        pipeline.execute()
        t_end = r.time()
        
        job_time = (t_end[0] + t_end[1]*1e-6)-(t_start[0] + t_start[1]*1e-6)
        t_set.append(job_time)
        t_set_complete.append(t_end[0] + t_end[1]*1e-6)
        
    # get values
    for _ in range(N):
        for i in range(size):
            key = "{0:015b}".format(i)
            pipeline.get(key)


        t_start = r.time()
        pipeline.execute()
        t_end = r.time()
        
        job_time = (t_end[0] + t_end[1]*1e-6)-(t_start[0] + t_start[1]*1e-6)
        t_get.append(job_time)
        t_get_complete.append(t_end[0] + t_end[1]*1e-6)

    # clear keys
    for i in range(N):
        r.delete("{0:015b}".format(i))
    
    memory_set[str(size)] = np.mean(t_set)
    time_table_set[str(size)] = t_set
    
    memory_get[str(size)] = np.mean(t_get)
    time_table_get[str(size)] = t_get
    
    


# In[ ]:


N = 10000

memory_set = {}
memory_get = {}

time_table_set = {}
time_table_get = {}

time_complete_set = {}
time_complete_get = {}

pipeline = r.pipeline(transaction=True)


job_size = np.array([1,10,100,1000,10000,100000])

for size in job_size:
    
    print("N jobs {}".format(size))
    t_set = []
    t_get = []
    t_set_complete = []
    t_get_complete =[]
    
    # store values
    
    for _ in range(N):
        for i in range(size):
            key = "{0:015b}".format(i)
            x = np.random.uniform(0,1,size=(8,8))
            pipeline.set(key,x.tobytes())

        t_start = r.time()
        pipeline.execute()
        t_end = r.time()
        
        job_time = (t_end[0] + t_end[1]*1e-6)-(t_start[0] + t_start[1]*1e-6)
        t_set.append(job_time)
        t_set_complete.append(t_end[0] + t_end[1]*1e-6)
        
        # get values
        for i in range(size):
            key = "{0:015b}".format(i)
            pipeline.get(key)

        t_start = r.time()
        pipeline.execute()
        t_end = r.time()
        
        job_time = (t_end[0] + t_end[1]*1e-6)-(t_start[0] + t_start[1]*1e-6)
        t_get.append(job_time)
        t_get_complete.append(t_end[0] + t_end[1]*1e-6)

        # clear keys
        for i in range(size):
            r.delete("{0:015b}".format(i))
    
    memory_set[str(size)] = np.mean(t_set)
    time_table_set[str(size)] = t_set
    
    memory_get[str(size)] = np.mean(t_get)
    time_table_get[str(size)] = t_get


# In[42]:


plt.plot(np.array(list(memory_set.keys())),np.array(list(memory_set.values())))
plt.plot(np.array(list(memory_get.keys())),np.array(list(memory_get.values())))
plt.xlabel("Number of Jobs",fontsize=15)
plt.ylabel("Average Time (s)",fontsize=15)
plt.title("Number of jobs all at once",fontsize=20)
plt.legend(["Set","Get"])
plt.show()


# In[44]:


plt.plot(np.array(list(memory_get.keys())),np.array(list(memory_get.values()))-np.array(list(memory_set.values())))
plt.xlabel("Number Of Jobs",fontsize=15)
plt.ylabel("Average Time",fontsize=15)
plt.title("Get - Set as Number Of Jobs increased",fontsize=20)
plt.show()


# In[66]:


plt.figure(figsize=(10,5))


for size in job_size:
    key = size
    
    t = time_table_set[str(key)]

    num_bins = int(np.sqrt(len(t))) #int(np.floor(5/3 * (len(t)**(1/3))))
    counts, bin_edges = np.histogram (t, bins=num_bins, normed=True)
    cdf = np.cumsum (counts)
    #plt.plot (np.hstack((np.zeros((1,)),bin_edges[1:])), np.hstack((np.zeros(1,),cdf/cdf[-1])))
    plt.plot (bin_edges[1:],cdf/cdf[-1])
    plt.xlabel("t (s)",fontsize=15); plt.ylabel("F(t)",fontsize=15);
    plt.title("Set number of jobs all at once CDF",fontsize=20)
    
    

plt.xlim([0,2])
plt.legend(job_size)
plt.show()


# In[69]:


plt.figure(figsize=(10,5))


for size in job_size:
    key = size
    
    t = time_table_get[str(key)]

    num_bins = int(np.sqrt(len(t)))#int(np.floor(5/3 * (len(t)**(1/3))))
    counts, bin_edges = np.histogram (t, bins=num_bins, normed=True)
    cdf = np.cumsum (counts)
    #plt.plot (np.hstack((np.zeros((1,)),bin_edges[1:])), np.hstack((np.zeros(1,),cdf/cdf[-1])))
    plt.plot (bin_edges[1:], cdf/cdf[-1])
    plt.title("Get Number of jobs all at once CDF",fontsize=20)
    plt.xlabel("t (s)",fontsize=15); plt.ylabel("F(t)",fontsize=15);
    
    

plt.xlim([0,0.003])
plt.legend(job_size)
plt.show()


# In[70]:


# clear keys
for i in range(N):
    r.delete("{0:015b}".format(i))

