# Course: CS261 - Data Structures
# Author: Cheng Ying Wu
# Assignment: 6
# Description: Implement the Undirected Graph class by completing provided skeleton code in the file ud_graph.py.

# Import deque
from collections import deque


class UndirectedGraph:
    """
    Class to implement undirected graph
    - duplicate edges not allowed
    - loops not allowed
    - no edge weights
    - vertex names are strings
    """

    def __init__(self, start_edges=None):
        """
        Store graph info as adjacency list
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        self.adj_list = dict()

        # populate graph with initial vertices and edges (if provided)
        # before using, implement add_vertex() and add_edge() methods
        if start_edges is not None:
            for u, v in start_edges:
                self.add_edge(u, v)

    def __str__(self):
        """
        Return content of the graph in human-readable form
        DO NOT CHANGE THIS METHOD IN ANY WAY
        """
        out = [f'{v}: {self.adj_list[v]}' for v in self.adj_list]
        out = '\n  '.join(out)
        if len(out) < 70:
            out = out.replace('\n  ', ', ')
            return f'GRAPH: {{{out}}}'
        return f'GRAPH: {{\n  {out}}}'

    # ------------------------------------------------------------------ #

    def add_vertex(self, v: str) -> None:
        """
        Adds a new vertex to the graph.
        If the vertex with the same name is already present in the graph, the method does nothing.
        """
        # Checks whether the vertex with the same name is already present in the graph
        if v in self.adj_list:
            # If yes, the method does nothing
            return
        else:
            # If no, adds a new vertex to the graph with an empty list
            self.adj_list[v] = []

    def add_edge(self, u: str, v: str) -> None:
        """
        Adds a new edge to the graph, connecting two vertices with provided names.
        If either (or both) vertex names do not exist in the graph, this method will first create them and then
        create an edge between them.
        If an edge already exists in the graph, or if u and v refer to the same vertex, the method does nothing.
        """
        # Checks whether u and v refer to the same vertex
        if u == v:
            # If yes, the method does nothing
            return

        # Checks whether the vertices are in the graph
        if u not in self.adj_list:
            # If not, first creates the vertex
            self.add_vertex(u)

        if v not in self.adj_list:
            self.add_vertex(v)

        # Checks whether an edge already exists in the graph
        if u in self.adj_list[v] and v in self.adj_list[u]:
            # If yes, the method does nothing
            return
        else:
            # If not, creates an edge between them
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)

    def remove_edge(self, v: str, u: str) -> None:
        """
        Removes an edge between two vertices with provided names.
        If either vertex names do not exist in the graph or there is no edge between them, the method does nothing.
        """
        # Checks whether the vertex and the edge exists
        if u not in self.adj_list or v not in self.adj_list or not (u in self.adj_list[v] and v in self.adj_list[u]):
            # If not, the method does nothing
            return
        else:
            # If yes, removes an edge between them
            self.adj_list[u].remove(v)
            self.adj_list[v].remove(u)

    def remove_vertex(self, v: str) -> None:
        """
        Removes a vertex with a given name and all edges incident to it from the graph.
        If the given vertex does not exist, the method does nothing.
        """
        # Checks whether the vertex exists
        if v not in self.adj_list:
            # If not, the method does nothing
            return
        else:
            # If yes, removes that vertex and the edges associated with it
            del self.adj_list[v]
            # Removes the edges
            for each_edge in self.adj_list.values():
                # Finds the edge associated with it
                if v in each_edge:
                    each_edge.remove(v)

    def get_vertices(self) -> []:
        """
        Returns a list of vertices in the graph (any order).
        """
        # Initiates the list to be returned
        ret_lst = []

        # Iterates through the keys in the dictionary
        for vertex in self.adj_list:
            ret_lst.append(vertex)

        # Returns a list of vertices
        return ret_lst

    def get_edges(self) -> []:
        """
        Returns a list of edges in the graph (any order).
        Each edge is returned as a tuple of two incident vertex names.
        """
        # Initiates the list to be returned
        ret_lst = []

        # Iterates through the keys and values in the dictionary
        for vertex, each_edge in self.adj_list.items():
            # Gets the edges
            for other_vertex in each_edge:
                # Checks whether the edge is already added
                if (vertex, other_vertex) not in ret_lst and (other_vertex, vertex) not in ret_lst:
                    ret_lst.append((vertex, other_vertex))

        # Returns a list of edges
        return ret_lst

    def is_valid_path(self, path: []) -> bool:
        """
        Takes a list of vertex names and returns True if the sequence of vertices represents a valid path in the graph.
        Returns False, otherwise. Empty path is considered valid.
        """
        # Checks whether it is an empty list
        if not path:
            return True

        # Checks whether the first vertex is in the graph
        if path[0] not in self.adj_list:
            return False

        # Follows each vertex in the list (Accesses two consecutive elements)
        for index in range(len(path) - 1):
            # Checks whether there is an edge between them (Whether the second vertex is in the first vertex's list)
            if path[index + 1] not in self.adj_list[path[index]]:
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

        # If the starting vertex is not in the graph, the method should return an empty list
        if v_start not in self.adj_list:
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

                # Since the vertices should be picked in ascending lexicographical order, sorts the list first
                sorted_lst = sorted(self.adj_list[pop_vertex], reverse=True)
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

        # If the starting vertex is not in the graph, the method should return an empty list
        if v_start not in self.adj_list:
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

                # Since the vertices should be picked in ascending lexicographical order, sorts the list first
                sorted_lst = sorted(self.adj_list[deq_vertex])
                # For each direct successor of v
                for vertex in sorted_lst:
                    # If the direct successor of v is not in the set of visited vertices, enqueues it into the queue
                    if vertex not in visited_set:
                        queue.append(vertex)

        # Returns the list of vertices visited during the BFS search
        return ret_lst

    def count_connected_components(self):
        """
        Returns the number of connected components in the graph.
        """
        # Initializes the counter
        counter = 0

        # Initializes the not yet visited vertices set to the set containing all vertices in the graph
        not_visited = set(self.get_vertices())

        for vertex in self.adj_list:
            if vertex in not_visited:
                # Uses DFS to get the reachable vertices set
                reach_set = set(self.dfs(vertex))
                # Subtracts all visited vertices from the not yet visited vertices set
                not_visited = not_visited - reach_set
                # Increases the counter accordingly
                counter += 1

        # Returns the number of connected components
        return counter

    def has_cycle(self):
        """
        Returns True if there is at least one cycle in the graph.
        If the graph is acyclic, the method returns False.
        """
        # Iterates through the vertices in the graph
        for vertex in self.adj_list:
            # Gets the length of the list containing the vertex's direct successors
            suc_len = len(self.adj_list[vertex])
            # Iterates through the direct successors of each vertex
            if suc_len == 1:
                # Ignores the vertex with only one direct successor
                continue
            else:
                for index in range(suc_len - 1):
                    # Gets two direct successors
                    first_suc = self.adj_list[vertex][index]
                    second_suc = self.adj_list[vertex][index + 1]

                    # Checks whether they can reach out to each other without passing their parent
                    ver_lst = []
                    # First, gets the direct successors (without the vertex) of the first direct successors
                    for successor in self.adj_list[first_suc]:
                        if successor != vertex:
                            ver_lst.append(successor)

                    # Initializes the set of visited vertices
                    vis_set = set(first_suc)
                    while ver_lst:
                        pop_vertex = ver_lst.pop()
                        # Checks whether it is in the set of visited vertices
                        if pop_vertex not in vis_set:
                            # If not, adds it to the set of visited vertices
                            vis_set.add(pop_vertex)

                            # Pushes each vertex that is the direct successor of v to the stack
                            for a_vertex in self.adj_list[pop_vertex]:
                                ver_lst.append(a_vertex)

                    # Checks whether the second direct successor is in the visited set
                    if second_suc in vis_set:
                        return True

        # If the graph is acyclic, the method returns False
        return False


# BASIC TESTING
if __name__ == '__main__':

    print("\nPDF - method add_vertex() / add_edge example 1")
    print("----------------------------------------------")
    g = UndirectedGraph()
    print(g)

    for v in 'ABCDE':
        g.add_vertex(v)
    print(g)

    g.add_vertex('A')
    print(g)

    for u, v in ['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE', ('B', 'C')]:
        g.add_edge(u, v)
    print(g)

    print("\nPDF - method remove_edge() / remove_vertex example 1")
    print("----------------------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    g.remove_vertex('DOES NOT EXIST')
    g.remove_edge('A', 'B')
    g.remove_edge('X', 'B')  # No vertex "X"
    g.remove_edge('A', 'E')  # No such edge
    print(g)
    g.remove_vertex('D')
    print(g)

    print("\nPDF - method get_vertices() / get_edges() example 1")
    print("---------------------------------------------------")
    g = UndirectedGraph()
    print(g.get_edges(), g.get_vertices(), sep='\n')
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE'])
    print(g.get_edges(), g.get_vertices(), sep='\n')

    print("\nPDF - method is_valid_path() example 1")
    print("--------------------------------------")
    g = UndirectedGraph(['AB', 'AC', 'BC', 'BD', 'CD', 'CE', 'DE'])
    test_cases = ['ABC', 'ADE', 'ECABDCBE', 'ACDECB', '', 'D', 'Z']
    for path in test_cases:
        print(list(path), g.is_valid_path(list(path)))

    print("\nPDF - method dfs() and bfs() example 1")
    print("--------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    print("---")
    print("Initial Graph: ", g)
    print("---")
    test_cases = 'ABCDEGH'
    for case in test_cases:
        print(f'{case} DFS:{g.dfs(case)} BFS:{g.bfs(case)}')
    print('-----')
    for i in range(1, len(test_cases)):
        v1, v2 = test_cases[i], test_cases[-1 - i]
        print(f'{v1}-{v2} DFS:{g.dfs(v1, v2)} BFS:{g.bfs(v1, v2)}')

    print("\nPDF - method count_connected_components() example 1")
    print("---------------------------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    print("---")
    print("Initial Graph: ", g)
    print("---")
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print(g.count_connected_components(), end=' ')
    print()

    print("\nPDF - method has_cycle() example 1")
    print("----------------------------------")
    edges = ['AE', 'AC', 'BE', 'CE', 'CD', 'CB', 'BD', 'ED', 'BH', 'QG', 'FG']
    g = UndirectedGraph(edges)
    print("---")
    print("Initial Graph: ", g)
    print("---")
    test_cases = (
        'add QH', 'remove FG', 'remove GQ', 'remove HQ',
        'remove AE', 'remove CA', 'remove EB', 'remove CE', 'remove DE',
        'remove BC', 'add EA', 'add EF', 'add GQ', 'add AC', 'add DQ',
        'add EG', 'add QH', 'remove CD', 'remove BD', 'remove QG',
        'add FG', 'remove GE')
    for case in test_cases:
        command, edge = case.split()
        u, v = edge
        g.add_edge(u, v) if command == 'add' else g.remove_edge(u, v)
        print('{:<10}'.format(case), g.has_cycle())
