import os
import io
import ast
import inspect
import pandas as pd
import numpy as np
from collections import deque, Counter


class EndNode():
    def __init__(self):
        """
        represent the end of program
        """
        self._fields = ""
        
    def __str__(self):
        return '_ast.Program_End'

class DummyNode():
    def __init__(self):
        """
        represent the dummpy node
        """
        self._fields = ""
        
    def __str__(self):
        return '_ast.Dummy_Node'

"""

 traverse the Abstact Syntax Tree and collect the AST nodes using
 dfs algorithm.
 
"""

class TraverseAST():
    
    @classmethod
    def ast_neighbors(cls, node):
        if not node:
            return []
        if not isinstance(node, ast.AST):
            return []
        neighbor_nodes = []
        for attr in node._fields:
            if attr not in ['body', 'orelse']:
                continue
            attr_node = getattr(node, attr)
            if isinstance(attr_node, ast.AST):
                neighbor_nodes.append([attr_node])
            if isinstance(attr_node, list):
                neighbor_nodes.append(attr_node)  
        return neighbor_nodes
    
    
    @classmethod
    def adjacency_list(cls, root):
        adj_list = {}
        if not root:
            return adj_list
        return_node = EndNode() # denote the end of logic graph
        node_list = []
        queue = deque([root])
        
        while queue:
            node = queue.popleft() # FIFO
            if not isinstance(node, ast.AST):
                continue
            if isinstance(node, ast.Return):
                continue
            
            # the next node of current root node
            next_node = None
            if node in adj_list and adj_list[node]:
                next_node = adj_list[node][0]
            # dummy node indicate the end of subgraph: for/while/if etc
            dummy_node = None
            if isinstance(node, (ast.For, ast.While)):
                dummy_node = DummyNode()
            
            # build adjacency list
            neighbors = cls.ast_neighbors(node)
            for neighbor_group in neighbors:
                current_node = node
                
                # add the dummy node and subgraph end node to the neighbor list
                pop_count = 0
                if dummy_node:
                    neighbor_group.append(dummy_node)
                    adj_list[dummy_node] = deque([node])
                    pop_count += 1
                if next_node:
                    neighbor_group.append(next_node)
                    pop_count += 1

                # iterate the neighbor and insert edges 
                for adj_node in neighbor_group:
                    if isinstance(current_node, ast.Return):
                        continue
                    tmp_list = adj_list.get(current_node, deque([]))
                    if isinstance(adj_node, ast.Return):
                        tmp_list.append(return_node)
                    else:
                        tmp_list.append(adj_node)
                    adj_list[current_node] = tmp_list
                    current_node = adj_node
                    
                # remove dummy and return nodes
                for _ in range(pop_count):
                    neighbor_group.pop()
                    
                # add the nodes to the head of the queue
                neighbor_group.reverse()
                queue.extendleft(neighbor_group)
                
        return adj_list

def arw_embedding(source_code, min_length = 5, max_length = 20, measure_step = 5, sample_number = 1500):
    
    # init ARW parameters
    walk_length = min_length
    walk_samples = {}
    while walk_length <= max_length:
        walk_samples[walk_length] = [0] * walk_length
        walk_length += measure_step
    
    # parse graph
    node_list = {}
    node_choices = []
    try:
        root = ast.parse(source_code, mode='exec')
        node_list = TraverseAST.adjacency_list(root)
        node_choices = list(node_list.keys())
    except Exception:
        pass
    
    # ARW
    if not node_choices:
        embedding = []
        for _, values in walk_samples.items():
            embedding.extend(values)
        return embedding
    
    for _ in range(sample_number):
        index = 0
        current_node = np.random.choice(node_choices)
        visited = {current_node}

        for path_length in range(1, walk_length):
            if current_node in node_list:
                neighbors = list(node_list[current_node])
            else:
                neighbors = list(node_list.keys())
                #visited = set([])
            current_node = np.random.choice(neighbors)
            if current_node not in visited:
                index += 1
                visited.add(current_node)
            if path_length + 1 in walk_samples:
                key = path_length - index
                walk_samples[path_length + 1][key] += 1
                
    # output result
    embedding = []
    for _, values in walk_samples.items():
        embedding.extend(values)
    return embedding

class FeaturePipeline():
    def __init__(self, root):
        if not root:
            raise ValueError('> AST root node can not be empty.')
        self.node_list = self._dfs(root)
        self.call_count = {}
        self.ifs_count = 0
        self.loop_count = 0
        self.break_count = 0
        self.continue_count = 0
        self.variables = set([])
        self.recursions = 0
        self.loop_loop_count = 0
        self.loop_cond_count = 0
        self.nested_loop_depth = 0
        self.cond_loop_count = 0
        self.cond_cond_count = 0
        self.loop_statement_count = 0
        self.loop_fun_call_count = 0
        self.loop_return_count = 0
    
    def _countable_features(self):
        for node in self.node_list:
            # func call counts
            if isinstance(node, ast.Call):
                self._count_func_call(node)
            # if count
            if isinstance(node, ast.If):
                self.ifs_count += 1
                self._cond_features(node)
            # loop count
            if isinstance(node, (ast.For, ast.While)):
                self.loop_count += 1
                self._loop_features(node)
            # break count
            if isinstance(node, ast.Break):
                self.break_count += 1
            # continue count
            if isinstance(node, ast.Continue):
                self.continue_count += 1
            # variable count
            if isinstance(node, ast.Assign):
                self._count_variable(node)
            # check recursion
            if isinstance(node, ast.FunctionDef):
                self.recursions = self._has_recursion(node)

            
            
    def __call__(self):
        self._countable_features()
        call_count = sum(self.call_count.values())
        return [
            len(self.call_count),
            call_count,
            self.ifs_count,
            self.loop_count,
            self.break_count,
            self.continue_count,
            len(self.variables),
            self.recursions,
            len(self.node_list),
            self.nested_loop_depth,
            self.loop_loop_count / float(self.loop_count) if self.loop_count > 0 else 0,
            self.loop_cond_count / float(self.loop_count) if self.loop_count > 0 else 0,
            self.cond_cond_count / float(self.ifs_count) if self.ifs_count > 0 else 0,
            self.cond_loop_count / float(self.ifs_count) if self.ifs_count > 0 else 0,
            self.loop_statement_count / float(len(self.node_list)) if self.node_list else 0,
            self.loop_fun_call_count / float(call_count) if call_count > 0 else 0,
            self.loop_return_count
        ]
    
    def _cond_features(self, root):
        if not hasattr(root, 'body'):
            return
        loop_list = self._dfs(root, node_type = (ast.For, ast.While, ast.If))
        loop_count = 0
        cond_count = 0
        for node in loop_list:
            if isinstance(node, (ast.For, ast.While)):
                loop_count += 1
            if isinstance(node, ast.If):
                cond_count += 1
        self.cond_loop_count += (1 if loop_count > 0 else 0)
        self.cond_cond_count += (1 if cond_count > 1 else 0)
    
    def _loop_features(self, root):
        if not hasattr(root, 'body'):
            return
        loop_list = self._dfs(root)
        loop_count = 0
        cond_count = 0
        for node in loop_list:
            if isinstance(node, (ast.For, ast.While)):
                loop_count += 1
            if isinstance(node, ast.If):
                cond_count += 1
            if isinstance(node, ast.Call):
                self.loop_fun_call_count += 1
            if isinstance(node, ast.Return):
                self.loop_return_count += 1
            self.loop_statement_count += 1
        self.nested_loop_depth = max(loop_count, self.nested_loop_depth)
        self.loop_loop_count += (1 if loop_count > 1 else 0)
        self.loop_cond_count += (1 if cond_count > 0 else 0)
    
    def _count_variable(self, node):
        if not hasattr(node, 'targets'):
            return
        for var in node.targets:
            if not isinstance(var, ast.Name):
                continue
            self.variables.add(var.id)
            
    def _count_func_call(self, node):
        if isinstance(node.func, ast.Name):
            self.call_count[node.func.id] = self.call_count.get(node.func.id, 0) + 1
        if isinstance(node.func, ast.Attribute):
            self.call_count[node.func.attr] = self.call_count.get(node.func.attr, 0) + 1
        
    def _has_recursion(self, root):
        if not hasattr(root, 'body'):
            return
        call_list = self._dfs(root, node_type=(ast.Name, ast.Attribute))
        call_count = {}
        for node in call_list:
            if isinstance(node, ast.Name):
                call_count[node.id] = call_count.get(node.id, 0) + 1
            if isinstance(node, ast.Attribute):
                call_count[node.attr] = call_count.get(node.attr, 0) + 1
                
        if root.name not in call_count:
            return 0
        return call_count[root.name]
        
        
    """
    "
    " traverse the node using the dfs algorithm
    "
    """       
    def _dfs(self, root, node_type=(ast.AST)):
        if not root:
            return node_list
        node_list = []
        queue = deque([root])
        while queue:
            node = queue.pop() # FIFO
            if isinstance(node, node_type):
                node_list.append(node)
            queue.extend(self._ast_neighbors(node))
        return node_list
    

    def _ast_neighbors(self, node):
        if not node:
            return []
        if not isinstance(node, ast.AST):
            return []
        neighbor_nodes = []
        for attr in node._fields:
            attr_node = getattr(node, attr)
            if isinstance(attr_node, ast.AST):
                neighbor_nodes.append(attr_node)
            if isinstance(attr_node, list):
                neighbor_nodes.extend(attr_node)
        return neighbor_nodes

def code_pattern_embedding(source_code):
  root = ast.parse(source_code, mode='exec')
  fp = FeaturePipeline(root)
  fp_emb = fp()
  return fp_emb