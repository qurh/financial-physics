# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 00:10:14 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define the model parameters
q_values = np.linspace(0.05, 0.25, 100)  # Range of noise parameter q
L_values = [40, 60, 80, 100, 120, 140]  # System sizes
p_values = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]  # Rewiring probabilities
num_steps = 10 ** 1  # Number of Monte Carlo steps

# Function to build the small-world network
def build_network(L, p):
    N = L ** 2  # Total number of nodes
    network = defaultdict(list)  # Adjacency list for the network

    # Build the initial square lattice
    for node in range(N):
        row, col = node // L, node % L
        neighbors = [(row, (col + 1) % L), ((row + 1) % L, col)]
        for neighbor_row, neighbor_col in neighbors:
            neighbor = neighbor_row * L + neighbor_col
            network[node].append(neighbor)

    # Rewire the links
    for node in range(N):
        neighbors_to_remove = []  # Store indices of neighbors to be removed
        for neighbor_idx in [0, 1]:
            if np.random.rand() < p:
                neighbor = network[node][neighbor_idx]
                # Calculate the total number of available nodes for rewiring
                available_nodes = set(range(N)) - set(network[node]) - {node} - set(network[neighbor])
                total_prob = len(available_nodes)
                if total_prob > 0:
                    new_neighbor = np.random.choice(list(available_nodes))
                    neighbors_to_remove.append(neighbor_idx)  # Store index of neighbor to be removed
                    network[node].append(new_neighbor)
                    network[new_neighbor].append(node)
    # Remove neighbors outside of the iteration
    for idx in sorted(neighbors_to_remove, reverse=True):
        del network[node][idx]
    return network

def majority_vote_dynamics(L, q, p):
    N = L ** 2  # Total number of nodes
    
    # Build the network
    network = build_network(L, p)
    
    # Initialize all states to be random
    states = np.random.choice([1, 2, 3], size=N)
    
    for step in range(num_steps):
        node = np.random.randint(N)
        neighbor_states = [states[neighbor] for neighbor in network[node]]
        
        # Count the number of neighbors in each state
        state_counts = [neighbor_states.count(state) for state in [1, 2, 3]]

        # Update the state of the current node based on the majority-vote rule
        max_count = max(state_counts)
        majority_states = [state for state, count in enumerate(state_counts, start=1) if count == max_count]

        if len(majority_states) == 1:
            p_new_state = 1 - q
        else:  # len(majority_states) == 2
            p_new_state = (1 - q) / len(majority_states)

        # Update the state of the current node
        if np.random.rand() < p_new_state:
            states[node] = np.random.choice(majority_states)
        else:
            other_states = [state for state in [1, 2, 3] if state not in majority_states]
            if other_states:  # Check if other_states is not empty
                states[node] = np.random.choice(other_states)
    
    return states


# Function to calculate the average magnetization
def calculate_magnetization(states, N):
    state_counts = [np.count_nonzero(states == state) for state in [1, 2, 3]]
    m1 = (3 / 2) * (state_counts[0] / N - 1 / 3)
    m2 = (3 / 2) * (state_counts[1] / N - 1 / 3)
    m3 = (3 / 2) * (state_counts[2] / N - 1 / 3)
    return np.sqrt(m1 ** 2 + m2 ** 2 + m3 ** 2)

# Function to calculate the magnetic susceptibility
def calculate_susceptibility(states, N, M):
    state_counts = [np.count_nonzero(states == state) for state in [1, 2, 3]]
    m1 = (3 / 2) * (state_counts[0] / N - 1 / 3)
    m2 = (3 / 2) * (state_counts[1] / N - 1 / 3)
    m3 = (3 / 2) * (state_counts[2] / N - 1 / 3)
    m_squared = m1 ** 2 + m2 ** 2 + m3 ** 2
    return N * (m_squared - M ** 2)

# Function to calculate the Binder cumulant
def calculate_binder_cumulant(states, N, M):
    state_counts = [np.count_nonzero(states == state) for state in [1, 2, 3]]
    m1 = (3 / 2) * (state_counts[0] / N - 1 / 3)
    m2 = (3 / 2) * (state_counts[1] / N - 1 / 3)
    m3 = (3 / 2) * (state_counts[2] / N - 1 / 3)
    m_fourth = m1 ** 4 + m2 ** 4 + m3 ** 4
    return 1 - m_fourth / (3 * M ** 4)

# Finite-size scaling analysis
for p in p_values:
    M_data = []
    chi_data = []
    U_data = []
    q_c_estimates = []
    
    for L in L_values:
        N = L ** 2
        M_L = []
        chi_L = []
        U_L = []
        
        for q in q_values:
            states = majority_vote_dynamics(L, q, p)
            M = calculate_magnetization(states, N)
            chi = calculate_susceptibility(states, N, M)
            U = calculate_binder_cumulant(states, N, M)
            
            M_L.append(M)
            chi_L.append(chi)
            U_L.append(U)
        
        M_data.append(M_L)
        chi_data.append(chi_L)
        U_data.append(U_L)
        
        # Estimate q_c from the crossing of Binder cumulant curves
        min_diff = np.inf
        q_c_estimate = None
        for q in q_values:
            diff = np.abs(U_L[np.searchsorted(q_values, q)] - 2/3)
            if diff < min_diff:
                min_diff = diff
                q_c_estimate = q
        q_c_estimates.append(q_c_estimate)
    
    # Plot the average magnetization
    plt.figure()
    for i, L in enumerate(L_values):
        plt.plot(q_values, M_data[i], label=f'L={L}')
    plt.xlabel('q')
    plt.ylabel('M(q, p)')
    plt.title(f'Average Magnetization (p={p})')
    plt.legend()
    plt.show()
    
    # Plot the magnetic susceptibility
    plt.figure()
    for i, L in enumerate(L_values):
        plt.plot(q_values, chi_data[i], label=f'L={L}')
    plt.xlabel('q')
    plt.ylabel(r'$\chi(q, p)$')
    plt.title(f'Magnetic Susceptibility (p={p})')
    plt.legend()
    plt.show()
    
    # Plot the magnetic susceptibility peaks (Fig. 6)
plt.figure()
for p_value in [1.0, 0.0001]:  # Modify these values as needed
    for i, L in enumerate(L_values):
        plt.plot(q_values, chi_data[i], label=f'L={L}', alpha=0.5)
    plt.xlabel('q')
    plt.ylabel(r'$\chi(q, p)$')
    plt.title(f'Magnetic Susceptibility Peaks (p={p_value})')
    plt.legend()
    plt.show()
    
    # Plot the Binder cumulant
    plt.figure()
    for i, L in enumerate(L_values):
        plt.plot(q_values, U_data[i], label=f'L={L}')
    plt.xlabel('q')
    plt.ylabel('U(q, p)')
    plt.title(f'Binder Cumulant (p={p})')
    plt.legend()
    plt.show()
    
    # Plot the Binder cumulant crossing for q_c estimation (Fig. 8a)
    p_value = 0.0001  # Modify this value as needed
    plt.figure()
    for L in [60, 80, 100, 120, 140]:
        U_L = U_data[L_values.index(L)]
        plt.plot(q_values, U_L, label=f'L={L}')
        plt.axhline(y=2/3, linestyle='--', color='k', label=r'$U=2/3$')
        plt.xlabel('q')
        plt.ylabel('U(q, p)')
        plt.title(f'Binder Cumulant Crossing (p={p_value})')
        plt.legend()
        plt.show()
    
    # Estimate critical exponents
    q_c = np.mean(q_c_estimates)
    print(f'p={p}, q_c={q_c}')
    
    # Estimate 1/nu
    ln_L = np.log(L_values)
    ln_q_c_diff = np.log(np.array(q_c_estimates) - q_c+ 1e-10)
    coeff = np.polyfit(ln_L, ln_q_c_diff, 1)
    nu_inv = -coeff[0]
    print(f'1/nu = {nu_inv}')
    
    # Estimate beta/nu
    ln_M = np.log(np.array([M_data[i][np.searchsorted(q_values, q_c)] for i in range(len(L_values))]))
    coeff = np.polyfit(ln_L, ln_M, 1)
    beta_nu = -coeff[0]
    print(f'beta/nu = {beta_nu}')
    
    # Estimate gamma/nu
    ln_chi = np.log(np.array([chi_data[i][np.searchsorted(q_values, q_c)] for i in range(len(L_values))])+ 1e-10)
    coeff = np.polyfit(ln_L, ln_chi, 1)
    gamma_nu = coeff[0]
    print(f'gamma/nu = {gamma_nu}')
    
    # Plot the phase diagram (Fig. 8b)
    plt.figure()
    plt.plot(p_values, [q_c] * len(p_values), 'o-', label='Three-state')
    plt.xlabel('p')
    plt.ylabel(r'$q_c$')
    plt.title('Phase Diagram')
    plt.legend()
    plt.show()
    
    # Data collapse for magnetization
    plt.figure()
    for i, L in enumerate(L_values):
        scaled_q = np.abs(q_values - q_c) * (L ** (1/nu_inv))
        scaled_M = [M * np.power(L, beta_nu) for M in M_data[i]]
        plt.plot(scaled_q, scaled_M, label=f'L={L}')
    plt.xlabel(r'$\epsilon L^{1/\nu}$')
    plt.ylabel(r'$M L^{\beta/\nu}$')
    plt.title(f'Data Collapse for Magnetization (p={p})')
    plt.legend()
    plt.show()
    
    # Data collapse for susceptibility
    plt.figure()
    for i, L in enumerate(L_values):
        scaled_q = np.abs(q_values - q_c) * (L ** (1/nu_inv))
        scaled_chi = chi_data[i] * (L ** (-gamma_nu))
        plt.plot(scaled_q, scaled_chi, label=f'L={L}')
    plt.xlabel(r'$\epsilon L^{1/\nu}$')
    plt.ylabel(r'$\chi L^{-\gamma/\nu}$')
    plt.title(f'Data Collapse for Susceptibility (p={p})')
    plt.legend()
    plt.show()
    
    # Data collapse for Binder cumulant
    plt.figure()
    for i, L in enumerate(L_values):
        scaled_q = np.abs(q_values - q_c) * (L ** (1/nu_inv))
        plt.plot(scaled_q, U_data[i], label=f'L={L}')
    plt.xlabel(r'$\epsilon L^{1/\nu}$')
    plt.ylabel('U(q, p)')
    plt.title(f'Data Collapse for Binder Cumulant (p={p})')
    plt.legend()
    plt.show()