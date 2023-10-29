from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import random
import numpy as np

# After how many items should we stop?
THRESHOLD = 10000000


#------------------------------------------------------------------------------
# Functions definition

# Definition of the hash function that maps each element to a column of the count sketch 
# For every i from 0 to D, the a and b parameters should change, so it's like having D instances of the hash function)
def h(u, hParams):
    a = hParams[0]
    b = hParams[1]
    return ((a * u + b) % p) % W

# Definition of the hash function that maps each element to -1 or 1
# For every i from 0 to D, the a and b parameters should change, so it's like having D instances of the hash function)
def g(u, gParams):
    a = gParams[0]
    b = gParams[1]
    C = 2
    result = ((a * u + b) % p) % C
    if result == 0:
        result = -1
    return result
           
# Definition of the function to update the count sketch for every element that arrives
def computeHashFunctions(element):
            
    # A list of key-value pairs ((i, hi), gi) is returned. This will allow the countsketch to be updated easily
    # Since each of the hi and gi functions are represented by their (a, b) parameter tuple,
    # each of the hi and gi functions can be computed as a list comprehension using the hParameters and the gParameters alists
    listPairs = [((i, h(element, hParameters[i])),
                  g(element, gParameters[i])) for i in range(D)]
    return listPairs
        
      
# Definition of the main processing function (the one that is used by each batch in `foreachRDD`)
def process_batch(time, batch):
    global streamLength, histogram, countSketch
    
    # We are working on the batch at time `time`.
    batch_size = batch.count()
    streamLength[0] += batch_size
    
    #--------------------------------------------------------------------------
    # Exact computations
    
    # Extraction of the item counts from the batch
    batchCounts = (batch.filter(lambda e: int(e) >= left and int(e) <= right).map(lambda e: (int(e), 1))
                   .reduceByKey(lambda a, b: a+b).collectAsMap())

    # Update of the histogram for the exact computations
    for key, count in batchCounts.items():
        if key not in histogram:
            histogram[key] = count
        else:
            histogram[key] += count
    
    #--------------------------------------------------------------------------
    # Approximate computations - Update of the countSketch for each element
    
    batchCountSketch = (batch.filter(lambda e: int(e) >= left and int(e) <= right).flatMap(lambda e: computeHashFunctions(int(e)))
                       .reduceByKey(lambda a, b: a+b).collectAsMap())
    
    for (i, j), value in batchCountSketch.items():
        countSketch[i, j] += value
    
    #--------------------------------------------------------------------------
    # Assesment of the stopping conditions
    # If we wanted, here we could run some additional code on the global histogram
    if batch_size > 0:
        print("Batch size at time [{0}] is: {1}".format(time, batch_size))

    if streamLength[0] >= THRESHOLD:
        stopping_condition.set()
        

#------------------------------------------------------------------------------
# Execution of the program
if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    # PARAMETERS READING
    D = int(sys.argv[1])
    W = int(sys.argv[2])
    left = int(sys.argv[3])
    right = int(sys.argv[4])
    K = int(sys.argv[5])
    portExp = int(sys.argv[6])
    print("Receiving data from port =", portExp)

    #--------------------------------------------------------------------------
    # CONFIGURATION OF THE PROGRAM (left as in the template)
    
    # IMPORTANT: when running locally, it is *fundamental* that the
    # `master` setting is "local[*]" or "local[n]" with n > 1, otherwise
    # there will be no processor running the streaming computation and your
    # code will crash with an out of memory (because the input keeps accumulating).
    conf = SparkConf().setMaster("local[*]").setAppName("DistinctExample")
    # If you get an OutOfMemory error in the heap consider to increase the
    # executor and drivers heap space with the following lines:
    conf = conf.set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    
    
    # Here, with the duration you can control how large to make your batches.
    # Beware that the data generator we are using is very fast, so the suggestion
    # is to use batches of less than a second, otherwise you might exhaust the memory.
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 1)  # Batch duration of 1 second
    ssc.sparkContext.setLogLevel("ERROR")
    
    # TECHNICAL DETAIL:
    # The streaming spark context and our code and the tasks that are spawned all
    # work concurrently. To ensure a clean shut down we use this semaphore.
    # The main thread will first acquire the only permit available and then try
    # to acquire another one right after spinning up the streaming computation.
    # The second tentative at acquiring the semaphore will make the main thread
    # wait on the call. Then, in the `foreachRDD` call, when the stopping condition
    # is met we release the semaphore, basically giving "green light" to the main
    # thread to shut down the computation.
    # We cannot call `ssc.stop()` directly in `foreachRDD` because it might lead
    # to deadlocks.
    stopping_condition = threading.Event()
    
    
    # --------------------------------------------------------------------------
    # DEFINING THE REQUIRED DATA STRUCTURES REQUIRED TO MAINTAIN THE STATE OF THE STREAM

    streamLength = [0] # Stream length (an array to be passed by reference)
    histogram = {} # Hash Table for the distinct elements
    countSketch = np.zeros(shape = (D,W), dtype = int)
    
    # Definition of the parameters required for the D hash functions. 
    # They all share p, but a and b change for every hash function
    # The parameters corresponding to each of the hi and gi functions will be 
    # stored in a tuple (a,b), using the the definitions of a and b from the previous homeworks  
    p = 8191
    hParameters = [(random.randint(1, p - 1),random.randint(0, p - 1)) for i in range(D)]
    gParameters = [(random.randint(1, p - 1),random.randint(0, p - 1)) for i in range(D)]
    
    
    # CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    
    # For each batch, do the following.
    # BEWARE: the `foreachRDD` method has "at least once semantics", meaning
    # that the same data might be processed multiple times in case of failure.
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))
    
    # MANAGING STREAMING SPARK CONTEXT
    print("Starting streaming engine")
    ssc.start()
    print("Waiting for shutdown condition")
    stopping_condition.wait()
    print("Stopping the streaming engine")
    # NOTE: You will see some data being processed even after the
    # shutdown command has been issued: This is because we are asking
    # to stop "gracefully", meaning that any outstanding work
    # will be done.
    ssc.stop(False, True)
    print("Streaming engine stopped")

    #--------------------------------------------------------------------------
    # COMPUTATION AND PRINTING OF THE FINAL STATISTICS
    
    # Computation of the exact statistics
    frequencySum = sum(histogram.values())
    trueF2 = sum(value ** 2 for value in histogram.values())/(frequencySum**2)
    
    # Computation of the approximate statistics from the count sketch
    
    ## Approximate F2
    rowSumSketch = np.sum(np.square(countSketch, dtype=np.float64), axis=1)
    approxF2 = np.median(rowSumSketch)/(frequencySum**2)
    
    ## Approximate histogram
    def aprox_histogram():
        approxHistogram = {}
        for u in histogram:
            listFuj = []
            for i in range(D):
                fuj =  g(u, gParameters[i]) * countSketch[i,h(u, hParameters[i])]
                listFuj.append(fuj)
            fu = np.median(listFuj)
            approxHistogram[u] = fu
        return approxHistogram
    approxHistogram = aprox_histogram()
    
    ## Approximate relative error
    histogram = dict(sorted(histogram.items(), key=lambda entry: entry[1], reverse=True))
    KthFreq = list(histogram.values())[K-1]
    topK = {u: frequency for u, frequency in histogram.items() if frequency >= KthFreq}
    def avg_relative_error():
        listErrors = []
        for u, trueFrequency in topK.items():
            aproxFrequency = approxHistogram[u]
            relativeError = abs(trueFrequency-aproxFrequency)/trueFrequency
            listErrors.append(relativeError)
        return np.mean(listErrors)
    avgRelativeError = avg_relative_error()
    
    
    # Definition of the output string (the output is different if K <= 20)
    if K <= 20:
        # String for the top K element frequencies if K <= 20
        stringFrequencies = ''
        for u, value in topK.items():
            stringFrequencies += f'\n        Item {u} Freq = {value}  Est. Freq = {approxHistogram[u]:.0f}'
        
        # Final output string
        outputString = '''
        D = {D}  W = {W}  [left,right] = [{left},{right}]  K = {K}  Port = {port}
        Total number of items = {stream_length}
        Total number of items in [{left},{right}] = {frequency_sum}
        Number of distinct items in [{left},{right}] = {distinct_items}\
        {item_frequencies}
        Average err for top {K} = {avg_error:.4f}
        F2 = {exact_F2:.4f}    F2 estimate = {approx_F2:.4f}'''.format(D = D,
                                                               W = W,
                                                               left = left,
                                                               right = right,
                                                               K = K,
                                                               port = portExp,
                                                               stream_length = streamLength[0],
                                                               frequency_sum = frequencySum,
                                                               distinct_items = len(histogram),
                                                               avg_error = avgRelativeError,
                                                               exact_F2 = trueF2,
                                                               approx_F2 = approxF2,
                                                               item_frequencies = stringFrequencies)
    elif K > 20:
        outputString = '''
        D = {D}  W = {W}  [left,right] = [{left},{right}]  K = {K}  Port = {port}
        Total number of items = {stream_length}
        Total number of items in [{left},{right}] = {frequency_sum}
        Number of distinct items in [{left},{right}] = {distinct_items}
        Average err for top {K} = {avg_error:.4f}
        F2 = {exact_F2:.4f}    F2 estimate = {approx_F2:.4f}'''.format(D = D,
                                                               W = W,
                                                               left = left,
                                                               right = right,
                                                               K = K,
                                                               port = portExp,
                                                               stream_length = streamLength[0],
                                                               frequency_sum = frequencySum,
                                                               distinct_items = len(histogram),
                                                               avg_error = avgRelativeError,
                                                               exact_F2 = trueF2,
                                                               approx_F2 = approxF2)
    
    
    # Printing of the output string
    print(outputString)