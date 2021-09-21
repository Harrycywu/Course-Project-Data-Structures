# Course: CS261 - Data Structures
# Author: Cheng Ying Wu
# Assignment: 6
# Description: Implement the DirectedGraph class by completing provided skeleton code in the file d_graph.py.

# Import deque
from collections import deque


class DirectedGraph:
    """
    Class to implement directed weighted graph
    - duplicate edges not allowed
    - loops not allowed
    - only positive edge weights
    - vertex names are integers
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency matrix
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.v_count = 0
        self.adj_matrix = []

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            v_count = 0
            for u, v, _ in start_edges:
                v_count = max(v_count, u, v)
            for _ in range(v_count + 1):
                self.add_vertex()
            for u, v, weight in start_edges:
                self.add_edge(u, v, weight)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        if self.v_count == 0:
            return 'EMPTY GRAPH\n'
        out = '   |'
        out += ' '.join(['{:2}'.format(i) for i in range(self.v_count)]) + '\n'
        out += '-' * (self.v_count * 3 + 3) + '\n'
        for i in range(self.v_count):
            row = self.adj_matrix[i]
            out += '{:2} |'.format(i)
            out += ' '.join(['{:2}'.format(w) for w in row]) + '\n'
        out = f"GRAPH ({self.v_count} vertices):\n{out}"
        return out

    # ------------------------------------------------------------------ #

    def add_vertex(self) -> int:
        """
        Adds a new vertex to the graph.
        Returns a single integer - the number of vertices in the graph after the addition.
        """
        # Iterates through the vertex in the graph and add one element for each vertex
        for vertex in self.adj_matrix:
            vertex.append(0)

        # Increases the number of vertices accordingly
        self.v_count += 1
        # Adds a new vertex with according number of zeros to the matrix
        self.adj_matrix.append([0] * self.v_count)
        # Returns the number of vertices in the graph after the addition
        return self.v_count

    def add_edge(self, src: int, dst: int, weight=1) -> None:
        """
        Adds a new edge to the graph, connecting two vertices with provided indices.
        If either vertex indices do not exist in the graph, or if the weight is not a positive integer, or if src and
        dst refer to the same vertex, the method does nothing.
        If an edge already exists in the graph, the method will update its weight.
        """
        # If src and dst refer to the same vertex, the method does nothing
        if src == dst:
            return

        # Gets the indices in the graph
        graph_index = self.get_vertices()

        # Checks some conditions
        if src not in graph_index or dst not in graph_index:
            # If either vertex indices do not exist in the graph, the method does nothing
            return

        if weight <= 0:
            # If the weight is not a positive integer, the method does nothing
            return

        # Adds a new edge to the graph, connecting two vertices with provided indices
        self.adj_matrix[src][dst] = weight

    def remove_edge(self, src: int, dst: int) -> None:
        """
        Removes an edge between two vertices with provided indices.
        If either vertex indices do not exist in the graph, or if there is no edge between them,
        the method does nothing.
        """
        # Gets the indices in the graph
        graph_index = self.get_vertices()

        # Checks some conditions
        if src not in graph_index or dst not in graph_index:
            # If either vertex indices do not exist in the graph, the method does nothing
            return

        # Removes an edge between two vertices with provided indices
        self.adj_matrix[src][dst] = 0

    def get_vertices(self) -> []:
        """
        Returns a list of vertices of the graph (any order).
        """
        # Gets the indices in the graph
        graph_index = []
        for num in range(self.v_count):
            graph_index.append(num)

        # Returns a list of vertices of the graph
        return graph_index

    def get_edges(self) -> []:
        """
        Returns a list of edges in the graph (any order).
        Each edge is returned as a tuple of two incident vertex indices and weight.
        """
        # Initiates the list to be returned
        ret_lst = []

        # Gets the indices in the graph
        graph_index = self.get_vertices()

        # Iterates each vertex in the graph
        for vertex in graph_index:
            # Uses indices to access the destination vertex
            for index in range(len(self.adj_matrix[vertex])):
                edge_weight = self.adj_matrix[vertex][index]
                # Checks whether the edge between two vertices exists
                if edge_weight > 0:
                    ret_lst.append((vertex, index, edge_weight))

        # Returns a list of edges in the graph
        return ret_lst

    def is_valid_path(self, path: []) -> bool:
        """
        Takes a list of vertex indices and returns True if the sequence of vertices represents a valid path
        in the graph.
        Empty path is considered valid.
        """
        # Checks whether it is an empty list
        if not path:
            return True

        # Gets the indices in the graph
        graph_index = self.get_vertices()

        # Checks whether the first vertex is in the graph
        if path[0] not in graph_index:
            return False

        # Follows each vertex in the list (Accesses two consecutive elements)
        for index in range(len(path) - 1):
            # Checks whether there is an edge between them (Whether the weight between them is positive)
            if self.adj_matrix[path[index]][path[index + 1]] <= 0:
                # If one edge doesn't exist, then return False
                return False
        return True

    def dfs(self, v_start, v_end=None) -> []:
        """
        Returns a list of vertices visited during the DFS search.
        Vertices are picked in alphabetical order.
        """
        # Initiates the list to be returned
        ret_lst = []

        # Gets the indices in the graph
        graph_index = self.get_vertices()

        # If the starting vertex is not in the graph, the method should return an empty list
        if v_start not in graph_index:
            return ret_lst

        # Follows the algorithms provided in Module 10's Exploration: Working with Graphs
        # Initializes an empty set of visited vertices
        visited_set = set()

        # Initializes an empty stack (implemented by a list) and adds the starting vertex to the stack
        stack = [v_start]

        # If the stack is not empty, pop a vertex v
        while stack:
            # stack: Last in First out
            pop_vertex = stack.pop()
            # Adds it to the list to be returned
            if pop_vertex not in ret_lst:
                ret_lst.append(pop_vertex)

            # Checks whether it is the ending vertex
            if v_end is not None:
                # If it is the ending vertex, then stop!
                if pop_vertex == v_end:
                    return ret_lst

            # Checks whether it is in the set of visited vertices
            if pop_vertex not in visited_set:
                # If not, adds it to the set of visited vertices
                visited_set.add(pop_vertex)

                # Finds the direct successors of v
                suc_lst = []
                for index in range(len(self.adj_matrix[pop_vertex])):
                    if self.adj_matrix[pop_vertex][index] > 0:
                        suc_lst.append(index)

                # Since the vertices should be picked in ascending lexicographical order, sorts the list first
                sorted_lst = sorted(suc_lst, reverse=True)
                # Pushes each vertex that is the direct successor of v to the stack
                for vertex in sorted_lst:
                    stack.append(vertex)

        # Returns the list of vertices visited during the DFS search
        return ret_lst

    def bfs(self, v_start, v_end=None) -> []:
        """
        Returns a list of vertices visited during the BFS search.
        Vertices are picked in alphabetical order.
        """
        # Initiates the list to be returned
        ret_lst = []

        # Gets the indices in the graph
        graph_index = self.get_vertices()

        # If the starting vertex is not in the graph, the method should return an empty list
        if v_start not in graph_index:
            return ret_lst

        # Follows the algorithms provided in Module 10's Exploration: Working with Graphs
        # Initializes an empty set of visited vertices
        visited_set = set()

        # Initializes an empty queue (implemented by a deque) and adds the starting vertex to the queue
        queue = deque([v_start])

        # If the queue is not empty, dequeue a vertex v
        while queue:
            # queue: First in First out
            deq_vertex = queue.popleft()
            # Adds it to the list to be returned
            if deq_vertex not in ret_lst:
                ret_lst.append(deq_vertex)

            # Checks whether it is the ending vertex
            if v_end is not None:
                # If it is the ending vertex, then stop!
                if deq_vertex == v_end:
                    return ret_lst

            # Checks whether it is in the set of visited vertices
            if deq_vertex not in visited_set:
                # If not, adds it to the set of visited vertices
                visited_set.add(deq_vertex)

                # Finds the direct successors of v
                suc_lst = []
                for index in range(len(self.adj_matrix[deq_vertex])):
                    if self.adj_matrix[deq_vertex][index] > 0:
                        suc_lst.append(index)

                # Since the vertices should be picked in ascending lexicographical order, sorts the list first
                sorted_lst = sorted(suc_lst)
                # For each direct successor of v
                for vertex in sorted_lst:
                    # If the direct successor of v is not in the set of visited vertices, enqueues it into the queue
                    if vertex not in visited_set:
                        queue.append(vertex)

        # Returns the list of vertices visited during the BFS search
        return ret_lst

    def has_cycle(self):
        """
        Returns True if there is at least one cycle in the graph.
        If the graph is acyclic, the method returns False.
        """
        # Gets the indices in the graph
        graph_index = self.get_vertices()

        # Iterates through the vertices in the graph
        for vertex in graph_index:
            # Initializes the set of visited vertices
            vis_set = []

            # Initializes an empty stack (implemented by a list) and adds the starting vertex to the stack
            stack = [vertex]

            # If the stack is not empty, pop a vertex v
            while stack:
                # stack: Last in First out
                pop_vertex = stack.pop()

                # Checks whether it is in the set of visited vertices and equal to the vertex
                if pop_vertex in vis_set and pop_vertex == vis_set[0]:
                    # If yes, return True
                    return True
                elif pop_vertex in vis_set and pop_vertex != vis_set[0]:
                    continue
                else:
                    # If it is not in the set of visited vertices, adds it to the set of visited vertices
                    vis_set.append(pop_vertex)

                    # Finds the direct successors of v
                    suc_ver = []
                    for index in range(len(self.adj_matrix[pop_vertex])):
                        if self.adj_matrix[pop_vertex][index] > 0:
                            suc_ver.append(index)

                    # Pushes each vertex that is the direct successor of v to the stack
                    for suc_vertex in suc_ver:
                        stack.append(suc_vertex)

        # If the graph is acyclic, the method returns False
        return False

    def dijkstra(self, src: int) -> []:
        """
        Implements the Dijkstra algorithm to compute the length of the shortest path from a given vertex to all other
        vertices in the graph. Returns a list with one value per each vertex in the graph.
        If a certain vertex is not reachable from SRC, returned value should be INFINITY (in Python, use float(‘inf’)).
        """
        # Initiates the list to be returned
        ret_lst = []

        # Gets the indices in the graph
        graph_index = self.get_vertices()

        # Follows the algorithms provided in Module 10's Exploration: Working with Graphs
        # Initialize an empty dictionary representing visited vertices
        visited_vertex = dict()

        # Initialize an empty priority queue, and insert vs into it with distance (priority) 0
        queue = deque([(src, 0)])

        # While the priority queue is not empty
        while queue:
            # Removes the first element (a vertex) from the priority queue and assigns it to v
            vertex, distance = queue.popleft()
            # Let d be v’s distance (priority)

            # If v is not in the map of visited vertices
            if vertex not in visited_vertex:
                # Adds v to the visited map with distance/cost d
                visited_vertex[vertex] = distance

                # Finds the direct successors of v
                suc_lst = []
                for index in range(len(self.adj_matrix[vertex])):
                    if self.adj_matrix[vertex][index] > 0:
                        suc_lst.append(index)

                # For each direct successor vi of v
                for successor in suc_lst:
                    # Lets di equal the cost/distance associated with edge
                    cost = distance + self.adj_matrix[vertex][successor]
                    # Inserts vi to the priority queue with distance (priority) d + di
                    queue.append((successor, cost))

                # Maintains the property of the priority queue
                unsort_lst = list(queue)
                # Sorts the list by the distance
                sort_lst = sorted(unsort_lst, key=lambda x: x[1])
                queue = deque(sort_lst)

        # Iterates through the map of visited vertices
        for index in graph_index:
            # If a certain vertex is not reachable from SRC, returned value should be INFINITY
            if index not in visited_vertex:
                ret_lst.append(float("inf"))
            else:
                ret_lst.append(visited_vertex[index])

        # Returns a list with one value per each vertex in the graph
        return ret_lst


if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = DirectedGraph()
    print(g)
    for _ in range(5):
        g.add_vertex()
    print(g)

    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)
    print(g)

    print("\nPDF - method get_edges() example 1")
    print("----------------------------------")
    g = DirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    print(g.get_edges(), g.get_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    test_cases = [[0, 1, 4, 3], [1, 3, 2, 1], [0, 4], [4, 0], [], [2]]
    for path in test_cases:
        print(path, g.is_valid_path(path))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for start in range(5):
        print(f'{start} DFS:{g.dfs(start)} BFS:{g.bfs(start)}')

    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)

    edges_to_remove = [(3, 1), (4, 0), (3, 2)]
    for src, dst in edges_to_remove:
        g.remove_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')

    edges_to_add = [(4, 3), (2, 3), (1, 3), (4, 0)]
    for src, dst in edges_to_add:
        g.add_edge(src, dst)
        print(g.get_edges(), g.has_cycle(), sep='\n')
    print('\n', g)

    print("\nPDF - method has_cycle() example 2")
    print("----------------------------------")
    edges = [(2, 10, 18), (3, 1, 13), (3, 5, 19), (3, 9, 8), (4, 6, 5), (5, 11, 14), (5, 12, 8), (9, 10, 10),
             (10, 11, 10), (11, 7, 10), (12, 2, 3), (12, 5, 8)]
    g = DirectedGraph(edges)
    print(g)
    print(g.has_cycle())

    print("\nPDF - dijkstra() example 1")
    print("--------------------------")
    edges = [(0, 1, 10), (4, 0, 12), (1, 4, 15), (4, 3, 3),
             (3, 1, 5), (2, 1, 23), (3, 2, 7)]
    g = DirectedGraph(edges)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')
    g.remove_edge(4, 3)
    print('\n', g)
    for i in range(5):
        print(f'DIJKSTRA {i} {g.dijkstra(i)}')

    print("\nPDF - dijkstra() example 2")
    print("----------------------------------")
    edges = [(0, 6, 13), (2, 4, 15), (3, 1, 2), (4, 3, 1), (4, 8, 20), (6, 12, 7), (7, 4, 17), (7, 10, 19), (9, 3, 18),
             (9, 11, 5), (11, 4, 6), (11, 5, 14), (11, 7, 10)]
    g = DirectedGraph(edges)
    print(g)
    print(g.dijkstra(9))
