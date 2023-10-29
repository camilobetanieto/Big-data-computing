from pyspark import SparkContext, SparkConf
import random
from collections import defaultdict
import numpy as np
import time
import sys

#------------------------------------------------------------------------------
# Auxiliary functions

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

# New modified given function to count triangles
def countTriangles2(colors_tuple, edges, rand_a, rand_b, p, num_colors):
    #We assume colors_tuple to be already sorted by increasing colors. Just transform in a list for simplicity
    colors = list(colors_tuple)  
    #Create a dictionary for adjacency list
    neighbors = defaultdict(set)
    #Creare a dictionary for storing node colors
    node_colors = dict()
    for edge in edges:

        u, v = edge
        node_colors[u]= ((rand_a*u+rand_b)%p)%num_colors
        node_colors[v]= ((rand_a*v+rand_b)%p)%num_colors
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph
    for v in neighbors:
        # Iterate over each pair of neighbors of v
        for u in neighbors[v]:
            if u > v:
                for w in neighbors[u]:
                    # If w is also a neighbor of v, then we have a triangle
                    if w > u and w in neighbors[v]:
                        # Sort colors by increasing values
                        triangle_colors = sorted((node_colors[u], node_colors[v], node_colors[w]))
                        # If triangle has the right colors, count it.
                        if colors==triangle_colors:
                            triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count



# Function to assign the color of each vertex
def assignEdgeColor(edge, dict_vertex, hash_function):
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

# Function to create the keys as a tuple of sorted colors (hC(u),hC(v),i)
def assignEdgeTuple(edge, dict_vertex, hash_function, num_colors):
    for vertex in edge:
        if vertex not in dict_vertex.keys():
            color = hash_function(vertex)
            dict_vertex[vertex]=color
            
    colorU = dict_vertex[edge[0]]
    colorV = dict_vertex[edge[1]]
    return [(tuple(sorted((colorU,colorV,i))), edge) for i in range(num_colors)]

#------------------------------------------------------------------------------
# Definition of the algorithms

# First algorithm
def MR_ApproxTCwithNodeColors(edges, C):
    """
    This takes an RDD of edges and number of colors and returns an estimate of the number of triangles with the hash function.
    """

    # Definition of the hash function for each invocation of MR_ApproxTCwithNodeColors
    # Variable inizialization
    p = 8191
    a = random.randint(1, p - 1)
    b = random.randint(0, p - 1)

    def h(u):
        return ((a * u + b) % p) % C

    # R1
    # Map phase: create the subsets of edges
    dictVertex = {}
    newPairs = edges.map(lambda e: assignEdgeColor(edge=e,dict_vertex=dictVertex,hash_function=h)).filter(lambda e: e[0] is not None) 

    # Shuffle + grouping + Reduce phase (R1): count the number of triangles formed by edges in the same subset
    countsR1 = (newPairs.groupByKey()                                    # Shuffle + grouping
                .mapValues(lambda es: CountTriangles(es)))               # Reduce phase (R1)
                
    # R2
    # Map phase + Reduce phase (R2): summing the partial counts from each partition
    countsR2 = (C ** 2) * countsR1.values().sum()
    return countsR2

# Second algorithm
def MR_ExactTC(edges, C):
    """
    This takes and RDD of edges and colors and returns the exact triangle count.
    """
    
    # Definition of the hash function for each invocation of MR_ApproxTCwithNodeColors
    # Variable inizialization
    p = 8191
    a = random.randint(1, p - 1)
    b = random.randint(0, p - 1)

    def h(u):
        return ((a * u + b) % p) % C
    
    # R1
    # Map phase: create the subsets of edges
    dictVertex = {}
    newPairs = edges.flatMap(lambda e: assignEdgeTuple(edge=e,dict_vertex=dictVertex,hash_function=h, num_colors=C))
    
    # Shuffle + grouping + Reduce phase (R1): count the number of triangles formed by edges within the same group (the ones with tha same key)
    
    countsR1 = (newPairs.groupByKey()                                                    # Shuffle + grouping
                .map(lambda eg: countTriangles2(colors_tuple=eg[0], edges=eg[1],         # Reduce phase (R1)
                                                rand_a=a, rand_b=b, p=p, num_colors=C)))
    
    # R2
    # Map phase + Reduce phase (R2): summing the partial counts from each group
    countsR2 = countsR1.reduce(lambda x,y: x+y)
    return countsR2
    

#------------------------------------------------------------------------------
# Main function
if __name__ == "__main__":

    # Spark setup
    conf = SparkConf().setAppName('WordCountExample')
    conf.set('spark.locality.wait', '0s')
    sc = SparkContext(conf=conf)
    
    C = int(sys.argv[1]) # Number of colors
    R = int(sys.argv[2]) # Number of iterations
    F = int(sys.argv[3]) # Binary flag
    path = sys.argv[4] # Path to the input file

    # Read the input graph into an RDD of strings and transform it into an RDD of edges
    rawData = sc.textFile(path).repartition(32)
    edges = rawData.map(lambda line: tuple(map(int, line.split(',')))).cache()
    
    
    #--------------------------------------------------------------------------
    # Running the algorithms according to the binary flag F given as input
    if F==0:
        # Run MR_ApproxTCwithNodeColors R times to get R independent estimates of the number of triangles
        estimatesNodeColors = []
        timesNC = []
        for i in range(R):
            startTime = time.time()
            estimateColors = MR_ApproxTCwithNodeColors(edges, C)
            endTime = time.time()
            elapsed_time_NC = 1000*(endTime - startTime)
            estimatesNodeColors.append(estimateColors)
            timesNC.append(elapsed_time_NC)
            
        outputString = '''\
            Dataset = {path}
            Number of Edges = {num_edges}
            Number of Colors: {C}
            Number of Repetitions: {R}
            Approximation through node coloring
            - Number of triangles (median over {R} runs) = {median_NC}
            - Running time (average over {R} runs) = {time_NC:.2f} ms'''.format(path = path,
                                                                               num_edges = edges.count(),
                                                                               C = C,
                                                                               R = R,
                                                                               median_NC = np.median(estimatesNodeColors),
                                                                               time_NC = np.mean(timesNC))
    elif F==1:
        # Run MR_ExactTC R times to get the average time and the exat triangle count
        timesExact = []
        for i in range(R):
            startTime = time.time()
            exactCount = MR_ExactTC(edges, C)
            endTime = time.time()
            elapsedTimeExact = 1000*(endTime - startTime)
            timesExact.append(elapsedTimeExact)
            
        outputString = '''\
            Dataset = {path}
            Number of Edges = {num_edges}
            Number of Colors: {C}
            Number of Repetitions: {R}
            Exact algorithm with node coloring
            - Number of triangles = {exact_count}
            - Running time (average over {R} runs) = {times_exact:.2f} ms'''.format(path = path,
                                                                               num_edges = edges.count(),
                                                                               C = C,
                                                                               R = R,
                                                                               exact_count = exactCount,
                                                                               times_exact = np.mean(timesExact))
    else:
        outputString = 'Wrong F input'
    
    # Printing the output String
    print(outputString)
    
    