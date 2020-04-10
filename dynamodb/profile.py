import boto3
import numpy as np
import time
import matplotlib.pyplot as plt

# docker run -p 8000:8000 -v $(pwd)/local/dynamodb:/data/ amazon/dynamodb-local -jar DynamoDBLocal.jar -sharedDb -dbPath /data
def main():
    dynamodb=boto3.client(
        'dynamodb',
        endpoint_url='http://localhost:8000',
        aws_access_key_id='foo',
        aws_secret_access_key='bar',
        verify=False
    )

    try:
        dynamodb.delete_table(TableName='values')
    except dynamodb.exceptions.ResourceNotFoundException:
        print("Table does not exist yet, creating...")

    resource = boto3.resource(
        'dynamodb',
        endpoint_url='http://localhost:8000',
        aws_access_key_id='foo',
        aws_secret_access_key='bar',
        verify=False
        )
    t = resource.Table('values')

    dynamodb.create_table(
        TableName='values',
        KeySchema=[
            {
                'AttributeName': 'key',
                'KeyType': 'HASH'
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'key',
                'AttributeType': 'S'
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )

    # # Test job size vs job time

    N = 1000

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
        
        # store values
        
        
        for i in range(N):

            #t_start = r.time()
            t_start = time.time_ns()
            
            key = "{0:015b}".format(i)
            x = np.random.uniform(0,1,size=(size,size))
            #r.set(key,x.tobytes())
            t.put_item(Item={'key': key})

            #t_end = r.time()
            t_end = time.time_ns()
            
            #job_time = (t_end[0] + t_end[1]*1e-6)-(t_start[0] + t_start[1]*1e-6)
            job_time = t_end - t_start
            
            t_set.append(job_time)
            #t_set_complete.append(t_end[0] + t_end[1]*1e-6)
            t_set_complete.append(t_end)

            
            
        # get values
        for i in range(N):

            #t_start = r.time()
            t_start = time.time_ns()
            
            key = "{0:015b}".format(i)
            #r.get(key)
            t.get_item(Key={'key': key})
            
            #t_end =  r.time()
            t_end = time.time_ns()
            
            #job_time = (t_end[0] + t_end[1]*1e-6)-(t_start[0] + t_start[1]*1e-6)
            job_time = t_end - t_start
            
            t_get.append(job_time)
            #t_get_complete.append(t_end[0] + t_end[1]*1e-6)
            t_set_complete.append(t_end)
        #breakpoint()
            
        # clear keys
        for i in range(N):
            #r.delete("{0:015b}".format(i))
            t.delete_item(Key={'key': "{0:015b}".format(i)})
        
        memory_set[str(size**2 * 8)] = np.mean(t_set)
        time_table_set[str(size**2 * 8)] = t_set
        
        memory_get[str(size**2 * 8)] = np.mean(t_get)
        time_table_get[str(size**2 * 8)] = t_get
        
        time_complete_set[str(size**2 * 8)] = time_complete_set
        time_complete_get[str(size**2 * 8)] = time_complete_get



    plt.plot(np.array(list(memory_set.keys())),np.array(list(memory_set.values())))
    plt.plot(np.array(list(memory_get.keys())),np.array(list(memory_get.values())))
    plt.xlabel("Memory (bytes)")
    plt.ylabel("Average Time")
    plt.legend(["Set","Get"])


    plt.plot(np.array(list(memory_get.keys())),np.array(list(memory_get.values()))-np.array(list(memory_set.values())))
    plt.xlabel("Memory (bytes)")
    plt.ylabel("Average Time")
    plt.title("Set - Get as memory increased")
    plt.show()
    plt.figure(figsize=(10,5))


    for size in array_sizes:
        key = size**2 * 8
        
        t = time_table_set[str(key)]

        num_bins = int(np.sqrt(len(t)))#int(np.floor(5/3 * (len(t)**(1/3))))
        counts, bin_edges = np.histogram (t, bins=num_bins, normed=True)
        cdf = np.cumsum (counts)
        plt.plot (np.hstack((np.zeros((1,)),bin_edges[1:])), np.hstack((np.zeros(1,),cdf/cdf[-1])))
        plt.xlabel("t"); plt.ylabel("f(t)");
        
        

    plt.xlim([0,0.005])
    plt.legend(array_sizes**2 * 8)
    plt.show()
    plt.figure(figsize=(10,5))


    for size in array_sizes:
        key = size**2 * 8
        
        ti = time_table_get[str(key)]

        num_bins = int(np.sqrt(len(ti)))#int(np.floor(5/3 * (len(t)**(1/3))))
        counts, bin_edges = np.histogram (ti, bins=num_bins, normed=True)
        cdf = np.cumsum (counts)
        plt.plot (np.hstack((np.zeros((1,)),bin_edges[1:])), np.hstack((np.zeros(1,),cdf/cdf[-1])))
        plt.xlabel("t"); plt.ylabel("f(t)");
        
        

    plt.xlim([0,0.0015])
    plt.legend(array_sizes**2 * 8)
    plt.show()



    # clear keys
    for i in range(N):
        #r.delete("{0:015b}".format(i))
        t.delete_item(Key={'key': "{0:015b}".format(i)})

"""
    ## job vs total time


    N = 100

    memory_set = {}
    memory_get = {}

    time_table_set = {}
    time_table_get = {}

    time_complete_set = {}
    time_complete_get = {}

    #pipeline = r.pipeline(transaction=True)
    batcher = boto3.client('batch')


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
        

    plt.plot(np.array(list(memory_set.keys())),np.array(list(memory_set.values())))
    plt.plot(np.array(list(memory_get.keys())),np.array(list(memory_get.values())))
    plt.xlabel("Memory (bytes)")
    plt.ylabel("Average Time")
    plt.legend(["Set","Get"])
    plt.show()


    plt.plot(np.array(list(memory_get.keys())),np.array(list(memory_get.values()))-np.array(list(memory_set.values())))
    plt.xlabel("Memory (bytes)")
    plt.ylabel("Average Time")
    plt.title("Set - Get as memory increased")
    plt.show()


    plt.figure(figsize=(10,5))


    for size in job_size:
        key = size
        
        t = time_table_set[str(key)]

        num_bins = int(np.sqrt(len(t))) #int(np.floor(5/3 * (len(t)**(1/3))))
        counts, bin_edges = np.histogram (t, bins=num_bins, normed=True)
        cdf = np.cumsum (counts)
        plt.plot (np.hstack((np.zeros((1,)),bin_edges[1:])), np.hstack((np.zeros(1,),cdf/cdf[-1])))
        plt.xlabel("t"); plt.ylabel("f(t)");
        
        

    plt.xlim([0,5])
    plt.legend(job_size)
    plt.show()

    plt.figure(figsize=(10,5))


    for size in job_size:
        key = size
        
        t = time_table_get[str(key)]

        num_bins = int(np.sqrt(len(t)))#int(np.floor(5/3 * (len(t)**(1/3))))
        counts, bin_edges = np.histogram (t, bins=num_bins, normed=True)
        cdf = np.cumsum (counts)
        plt.plot (np.hstack((np.zeros((1,)),bin_edges[1:])), np.hstack((np.zeros(1,),cdf/cdf[-1])))
        plt.xlabel("t"); plt.ylabel("f(t)");


    plt.xlim()
    plt.legend(job_size)
    plt.show()

    # clear keys
    for i in range(N):
        r.delete("{0:015b}".format(i))
"""




if __name__ == "__main__":
    main()