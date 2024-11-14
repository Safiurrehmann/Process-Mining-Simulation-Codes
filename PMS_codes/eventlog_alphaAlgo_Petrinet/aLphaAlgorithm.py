import pandas as pd
from collections import defaultdict
import numpy as np

# Load the event log from an external CSV file
file_path = 'event_log.csv'  # Change this to the path of your CSV file
df = pd.read_csv(file_path)

# Step 1: Convert to unique traces and calculate frequencies
df['trace'] = df.groupby('order number')['activity'].transform(lambda x: ','.join(x))
unique_traces = df[['order number', 'trace']].drop_duplicates().drop(columns='order number')
trace_counts = unique_traces['trace'].value_counts().reset_index()
trace_counts.columns = ['Trace', 'Frequency']
print("Unique Traces and Frequencies:")
print(trace_counts)

# Step 2: Extract unique events, initial and final events
unique_events = df['activity'].unique()
initial_events = df.groupby('order number')['activity'].first().unique()
final_events = df.groupby('order number')['activity'].last().unique()

print("Unique Events (TL):", unique_events)
print("Initial Events (TI):", initial_events)
print("Final Events (TO):", final_events)

# Step 3: Construct footprint matrix
# Determine direct succession for footprint matrix
direct_succession = defaultdict(set)
for order in df['order number'].unique():
    order_activities = df[df['order number'] == order]['activity'].values
    for i in range(len(order_activities) - 1):
        direct_succession[order_activities[i]].add(order_activities[i + 1])

# Initialize footprint matrix
footprint_matrix = pd.DataFrame('-', index=unique_events, columns=unique_events)
for a in unique_events:
    for b in unique_events:
        if b in direct_succession[a]:
            if a in direct_succession[b]:
                footprint_matrix.at[a, b] = '||'  # Parallel
            else:
                footprint_matrix.at[a, b] = '->'  # Causal
        elif a in direct_succession[b]:
            footprint_matrix.at[a, b] = '<-'  # Inverse Causal

print("Footprint Matrix:")
print(footprint_matrix)

# Step 4: Create sets XL, YL, PL, FL
# XL: Set of causal relationships
XL = {(a, b) for a in unique_events for b in unique_events if footprint_matrix.at[a, b] == '->'}

# YL: Set of maximal pairs
YL = {(a, b) for a, b in XL if (b, a) not in XL}

# PL: Place set (for simplicity, using the causal pairs as places)
PL = {f'p_{i}': (a, b) for i, (a, b) in enumerate(YL)}

# FL: Flow relation
FL = set()
for (a, b) in YL:
    FL.add((a, f'p_{a}_{b}'))
    FL.add((f'p_{a}_{b}', b))

print("Causal Relationships (XL):", XL)
print("Maximal Pairs (YL):", YL)
print("Place Set (PL):", PL)
print("Flow Relation (FL):", FL)

# Step 5: Build Petri Net (Visual Representation - requires graphing library)
# Using graphviz to illustrate (optional)

try:
    from graphviz import Digraph

    petri_net = Digraph('PetriNet', format='png')

    # Define main activities to be circles
    main_activities = {'register order', 'check stock', 'ship order', 'handle payment'}
    for event in unique_events:
        if event in main_activities:
            petri_net.node(event, shape='circle')
        else:
            petri_net.node(event, shape='rectangle')

    # Add causal relationships (places) as rectangles
    for p in PL.keys():
        petri_net.node(p, shape='rectangle')

    # Add edges for all causal relationships
    for (a, b) in FL:
        petri_net.edge(a, b)

    petri_net.render('PetriNet')
    print("Petri Net created and saved as PetriNet.png")

except ImportError:
    print("Graphviz is not installed, please install it to visualize the Petri Net.")
