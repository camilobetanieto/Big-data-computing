from pyspark import SparkContext
import random
from collections import defaultdict
import numpy as np
import time

# Given function to count triangles
def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count



# Function to assign the color of each vertex
def AssignEdgeColor(edge, dict_vertex, hash_function):
    edgeColor = None

    for vertex in edge:
        if vertex not in dict_vertex.keys():
            color = hash_function(vertex)
            dict_vertex[vertex]=color

    colorU = dict_vertex[edge[0]]
    colorV = dict_vertex[edge[1]]
    if colorU == colorV:
        edgeColor = colorU
    return (edgeColor,edge)


# Definition of the first algorithm
def MR_ApproxTCwithNodeColors(edges, C):
    """
    this takes an RDD of edges and number of colors and returns an estimate of the number of triangles with the hash function.
    """
    
    
    # Define the hash function for each invocation of MR_ApproxTCwithNodeColors
    # Initializing variables
    p = 8191
    a = random.randint(1, p - 1)
    b = random.randint(0, p - 1)

    def h(u):
        return ((a * u + b) % p) % C

    # R1
    # Map phase: create the subsets of edges
    dictVertex = {}
    newPairs = edges.map(lambda e: AssignEdgeColor(edge=e,dict_vertex=dictVertex,hash_function=h)).filter(lambda e: e[0] is not None) 

    # Shuffle + grouping + Reduce phase (R1): count the number of triangles formed by edges in the same subset
    countsR1 = (newPairs.groupByKey()                                    # Shuffle + grouping
                .mapValues(lambda es: CountTriangles(es)))               # Reduce phase (R1):
                
    # R2
    # Map phase + Reduce phase (R2): summing the partial counts from each partition
    countsR2 = (C ** 2) * countsR1.values().sum()
    
    return countsR2

# Definition of the second algorithm
def MR_ApproxTCwithSparkPartitions(edges, C):
    """
    This takes and RDD of edges and colors and returns the number of triangles using the Spark partitions.
    """
    # Map R1 + Reduce R1: The partition was made when the RDD was defined and then the triangle count is made within each partition
    countsR1 = edges.mapPartitions(lambda subset: [CountTriangles(subset)]) 
     
    # Reduce R2: summing the partial counts from each partition
    countsR2 = (C**2) * countsR1.reduce(lambda x,y: x+y)

    return countsR2


# Main function
if __name__ == "__main__":
    import sys
    C = int(sys.argv[1]) # Number of colors
    R = int(sys.argv[2]) # Number of iterations
    path = sys.argv[3] # Path to the input file
    sc = SparkContext(appName="Triangle Counting")

    # Read the input graph into an RDD of strings and transform it into an RDD of edges
    rawData = sc.textFile(path).repartition(C)
    edges = rawData.map(lambda line: tuple(map(int, line.split(',')))).cache()

    # Run MR_ApproxTCwithNodeColors R times to get R independent estimates of the number of triangles
    estimatesNodeColors = []
    timesNC = []
    for i in range(R):
        start_time = time.time()
        estimateColors = MR_ApproxTCwithNodeColors(edges, C)
        end_time = time.time()
        elapsed_time_NC = 1000*(end_time - start_time)
        estimatesNodeColors.append(estimateColors)
        timesNC.append(elapsed_time_NC)

    # Run MR_ApproxTCwithSparkPartitions to get another estimate of the number of triangles
    start_time = time.time()
    estimateSparkPartitions = MR_ApproxTCwithSparkPartitions(edges, C)
    end_time = time.time()
    elapsed_time_SP = 1000*(end_time - start_time)
    
    
    # Printing the information summary
    outputString = '''\
    Data set = {path}
    Number of Edges = {num_edges}
    Number of Colors: {C}
    Number of Repetitions: {R}
    Approximation through node coloring
    - Number of triangles (median over {R} runs) = {medianNC}
    - Running time (average over {R} runs) = {timeNC:.2f} ms
    Approximation through Spark partitions
    - Number of triangles = {estimateSP}
    - Running time = {timeSP:.2f} ms'''.format(path = path,
                                               num_edges = edges.count(),
                                               C = C,
                                               R = R,
                                               medianNC = np.median(estimatesNodeColors),
                                               timeNC = np.mean(timesNC),
                                               estimateSP = estimateSparkPartitions,
                                               timeSP = elapsed_time_SP)
    print(outputString)
    
    