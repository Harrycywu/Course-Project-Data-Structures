# Course-Project-Data-Structures
Course Project - Data Structures (Graph Implementation)

Course: CS 261 - Data Structures

Term: Spring 2021

**Course Description:**

Abstract data types, dynamic arrays, linked lists, trees and graphs, binary search trees, hash tables, storage management, complexity analysis of data structures.

My Grade: A (96.96%)

# Project Name: Graph Implementation
**Project Description**

This assignment is comprised of 2 parts. In the first part, you will complete the implementation of a undirected graph ADT (ud_graph.py) where the vertices and edges should be stored as an adjacency list. In the second part, you will implement a directed graph ADT (d_graph.py) where the vertices and edges should be stored using an adjacency matrix.

**Part 1 - Undirected Graph (via Adjacency List)**

Implement the UndirectedGraph class by completing provided skeleton code in the file ud_graph.py. UndirectedGraph class is designed to support the following type of graph: undirected, unweighted, no duplicate edges, no loops. Cycles are allowed.

Undirected graphs should be stored as a Python dictionary of lists where keys are vertex names (strings) and associated values are Python lists with names (in any order) of vertices connected to the 'key' vertex.

RESTRICTIONS: For this assignment, you ARE allowed to use any built-in data structures from Python standard library.

Methods:
* add_vertex(self, v: str) -> None:
  * This method adds a new vertex to the graph. Vertex names can be any string. If vertex with the same name is already present in the graph, the method does nothing (no exception needs to be raised).
* add_edge(self, u: str, v: str) -> None:
  * This method adds a new edge to the graph, connecting two vertices with provided names. If either (or both) vertex names do not exist in the graph, this method will first create them and then create an edge between them. If an edge already exists in the graph, or if u and v refer to the same vertex, the method does nothing (no exception needs to be raised).
* remove_edge(self, u: str, v: str) -> None:
  * This method removes an edge between two vertices with provided names. If either (or both) vertex names do not exist in the graph, or if there is no edge between them, the method does nothing (no exception needs to be raised).
* remove_vertex(self, v: str) -> None:
  * This method removes a vertex with a given name and all edges incident to it from the graph. If the given vertex does not exist, the method does nothing (no exception needs to be raised).
* get_vertices(self) -> []:
  * This method returns a list of vertices of the graph. Order of the vertices in the list does not
matter .
* get_edges(self) -> []:
  * This method returns a list of edges in the graph. Each edge is returned as a tuple of two incident vertex names. Order of the edges in the list or order of the vertices incident to each edge does not matter.
* is_valid_path(self, path: []) -> bool:
  * This method takes a list of vertex names and returns True if the sequence of vertices represents a valid path in the graph (so one can travel from the first vertex in the list to the last vertex in the list, at each step traversing over an edge in the graph). Empty path is considered valid.
* dfs(self, v_start: str, v_end=None) -> []:
  * This method performs a depth-first search (DFS) in the graph and returns a list of vertices visited during the search, in the order they were visited. It takes one required parameter, name of the vertex from which the search will start, and one optional parameter - name of the ‘end’ vertex that will stop the search once that vertex is reached.
  * If the starting vertex is not in the graph, the method should return an empty list (no exception needs to be raised). If the name of the ‘end’ vertex is provided but is not in the graph, the search should be done as if there was no end vertex.
  * When several options are available for picking the next vertex to continue the search, your implementation should pick the vertices in ascending lexicographical order (so, for example, vertex ‘APPLE’ is explored before vertex ‘BANANA’).
* bfs(self, v_start: str, v_end=None) -> []:
  * This method works the same as DFS above, except it implements a breadth-first search.
* count_connected_components(self) -> int:
  * This method returns the number of connected components in the graph.
* has_cycle(self) -> bool:
  * This method returns True if there is at least one cycle in the graph. If the graph is acyclic,
the method returns False.

**Part 2 - Directed Graph (via Adjacency Matrix)**

Implement the DirectedGraph class by completing provided skeleton code in the file d_graph.py. DirectedGraph class is designed to support the following type of graph: directed, weighted (positive edge weights only), no duplicate edges, no loops. Cycles are allowed.

Directed graphs should be stored as a two dimensional matrix, which is a list of lists in Python. Element on the i-th row and j-th column in the matrix is the weight of the edge going from the vertex with index i to the vertex with index j. If there is no edge between those vertices, the value is zero.

RESTRICTIONS: For this assignment, you ARE allowed to use any built-in data structures from Python standard library.

Methods:
* add_vertex(self) -> int:
  * This method adds a new vertex to the graph. Vertex name does not need to be provided, instead vertex will be assigned a reference index (integer). First vertex created in the graph will be assigned index 0, subsequent vertices will have indexes 1, 2, 3 etc. This method returns a single integer - the number of vertices in the graph after the addition.
* add_edge(self, src: int, dst: int, weight=1) -> None:
  * This method adds a new edge to the graph, connecting two vertices with provided indices. If either (or both) vertex indices do not exist in the graph, or if the weight is not a positive integer, or if src and dst refer to the same vertex, the method does nothing. If an edge already exists in the graph, the method will update its weight.
* remove_edge(self, u: int, v: int) -> None:
  * This method removes an edge between two vertices with provided indices. If either (or both) vertex indices do not exist in the graph, or if there is no edge between them, the method does nothing (no exception needs to be raised).
* get_vertices(self) -> []:
  * This method returns a list of vertices of the graph. Order of the vertices in the list does not
matter .
* get_edges(self) -> []:
  * This method returns a list of edges in the graph. Each edge is returned as a tuple of two incident vertex indices and weight. First element in the tuple refers to the source vertex. Second element in the tuple refers to the destination vertex. Third element in the tuple is the weight of the edge. Order of the edges in the list does not matter.
* is_valid_path(self, path: []) -> bool:
  * This method takes a list of vertex indices and returns True if the sequence of vertices represents a valid path in the graph (so one can travel from the first vertex in the list to the last vertex in the list, at each step traversing over an edge in the graph). Empty path is considered valid.
* dfs(self, v_start: int, v_end=None) -> []:
  * This method performs a depth-first search (DFS) in the graph and returns a list of vertices visited during the search, in the order they were visited. It takes one required parameter, index of the vertex from which the search will start, and one optional parameter - index of the ‘end’ vertex that will stop the search once that vertex is reached.
  * If the starting vertex is not in the graph, the method should return an empty list (no exception needs to be raised). If the ‘end’ vertex is provided but is not in the graph, the search should be done as if there was no end vertex.
  * When several options are available for picking the next vertex to continue the search, your implementation should pick the vertices by vertex index in ascending order (so, for example, vertex 5 is explored before vertex 6).
* bfs(self, v_start: int, v_end=None) -> []:
  * This method works the same as DFS above, except it implements a breadth-first search.
* has_cycle(self) -> bool:
  * This method returns True if there is at least one cycle in the graph. If the graph is acyclic,
the method returns False.
* dijkstra(self, src: int) -> []:
  * This method implements the Dijkstra algorithm to compute the length of the shortest path from a given vertex to all other vertices in the graph. It returns a list with one value per each vertex in the graph, where value at index 0 is the length of the shortest path from vertex SRC to vertex 0, value at index 1 is the length of the shortest path from vertex SRC to vertex 1 etc. If a certain vertex is not reachable from SRC, returned value should be INFINITY (in Python, use float(‘inf’)).
