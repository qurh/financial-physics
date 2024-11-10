# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 17:24:44 2024

@author: User
"""
import igraph as ig

g = ig.Graph(n=10, edges=[[0,1],[1,3]])

g.add_vertices(3)

g.add_edges([(2,4), (1,8)])

print(g.get_eid(1, 3))

g.delete_edges(1)

g1 = ig.Graph.Tree(32, 2)

ge = g1.get_edgelist()

print(ge[1:10:2])

g3 = ig.Graph.GRG(1,10)

print(g3)

