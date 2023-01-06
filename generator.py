def generate_dataset(lg_N):
  N = 2**lg_N #N: Max number of nodes in a cycle
   #lg_N: Number of bits to store node index
  half_lg_N = lg_N/2

  dataset = []
  dataset_2 = dict()
  dataset_3 = {}

  node_id = {}
  counter = 0
  for n in range(3, N): # For each cycle
    for i in range(n): # For each node in each cycle
      node_id[(n, i)] = counter
      counter += 1

  for n in range(3, N): # For each cycle
    for i in range(n): # For each node in each cycle
      for j in range(n):
        if abs(i-j) <= 4:
          for k in range(5):
            dataset.append([node_id[(n,i)], node_id[(n,j)], min(abs(i-j), n-abs(i-j))])
        dataset.append([node_id[(n,i)], node_id[(n,j)], min(abs(i-j), n-abs(i-j))])
        dataset_2[(node_id[(n,i)], node_id[(n,j)])] = min(abs(i-j), n-abs(i-j))
        dataset_3[node_id[(n,i)]] = n, i

  return dataset, dataset_2, dataset_3, node_id
