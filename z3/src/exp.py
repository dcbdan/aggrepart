from problem import *

def exp01(max_time):
  all_subsets = ["a", "b", "c", "d", "ab", "cd", "abcd"]
  _all_partitions = {
    "a": [["a"]],
    "b": [["b"]],
    "c": [["c"]],
    "d": [["d"]],
    "ab": [["a","b"], ["ab"]],
    "cd": [["c","d"], ["cd"]],
    "abcd": [["a","b","c","d"], ["ab", "cd"]] }
  get_all_partitions = lambda elems: _all_partitions[elems]
  get_subset_size = lambda elems: len(elems)

  init_elems_locs_dict = {
    "a": [0],
    "b": [1],
    "c": [2],
    "d": [3] }
  fini_elems_locs_list = [("abcd", [0,1,2,3])]

  resources = {}
  for i in range(4):
    for j in range(4):
      if i != j:
        resources[(i,j)] = 1

  problem = Problem(
    all_subsets,
    get_all_partitions,
    get_subset_size,
    init_elems_locs_dict,
    fini_elems_locs_list,
    resources,
    max_time)

  use_full_solver_interface = True
  solve_problem(problem, use_full_solver_interface)

def exp02(max_time):
  all_subsets = ["a", "b", "c", "d", "e", "f", "g", "h",
                 "ab", "cd", "ef", "gh", "abcd", "efgh",
                 "abcdefgh"]
  _all_partitions = {
    "a": [["a"]],
    "b": [["b"]],
    "c": [["c"]],
    "d": [["d"]],
    "ab": [["a","b"], ["ab"]],
    "cd": [["c","d"], ["cd"]],
    "abcd": [["a","b","c","d"], ["ab", "cd"]],
    "e": [["e"]],
    "f": [["f"]],
    "g": [["g"]],
    "h": [["h"]],
    "ef": [["e","f"], ["ef"]],
    "gh": [["g","h"], ["gh"]],
    "efgh": [["e","f","g","h"], ["ef", "gh"]],
    "abcdefgh": [["a","b","c","d","e","f","g","h"],
                 ["ab", "cd", "ef", "gh"],
                 ["abcd", "efgh"]]
  }

  get_all_partitions = lambda elems: _all_partitions["".join(sorted(elems))]
  get_subset_size = lambda elems: 2

  init_elems_locs_dict = {
    "a": [0],
    "b": [1],
    "c": [2],
    "d": [3],
    "e": [4],
    "f": [5],
    "g": [6],
    "h": [7]
  }
  fini_elems_locs_list = [("abcdefgh", [0,1,2,3,4,5,6,7])]

  resources = {}
  for i in range(8):
    for j in range(8):
      if i != j:
        resources[(i,j)] = 1

  problem = Problem(
    all_subsets,
    get_all_partitions,
    get_subset_size,
    init_elems_locs_dict,
    fini_elems_locs_list,
    resources,
    max_time)

  use_full_solver_interface = True
  solve_problem(problem, use_full_solver_interface)

def exp03(max_time = 4, all_connected = False):
  init_info = {
    'a': [0], 'b': [1], 'c': [2], 'd': [3], 'e': [4], 'f': [5], 'g': [6], 'h': [7] }
  fini_info = [ ("abcdefgh", [0,1,2,3,4,5,6,7]) ]

  resources = {}
  if all_connected:
    for i in range(8):
      for j in range(8):
        if i != j:
          resources[(i,j)] = 1
  else:
    for locs in [ [0,1,2,3], [4,5,6,7] ]:
      for i in locs:
        for j in locs:
          if i != j:
            resources[(i,j)] = 1

    for (i,j) in [ (0,4), (1,5), (2,6), (3,7) ]:
      resources[(i,j)] = 1
      resources[(j,i)] = 1

  get_subset_size = lambda _: 1

  all_subsets = set()
  ps = all_partitions("abcdefgh")
  for part in ps:
    all_subsets.update(part)

  get_all_partitions = lambda elems: list(all_partitions(elems))

  problem = Problem(
    all_subsets,
    get_all_partitions,
    get_subset_size,
    init_info,
    fini_info,
    resources,
    max_time)

  use_full_solver_interface = True
  solve_problem(problem, use_full_solver_interface)

def exp04(max_time):
  init_info = {
    'a': [0], 'b': [1], 'c': [2], 'd': [3] }
  fini_info = [ ("abcd", [0,1,2,3]) ]

  resources = {}
  resources[(0,1)] = 1
  resources[(1,0)] = 1

  resources[(2,3)] = 1
  resources[(3,2)] = 1

  resources[(0,2)] = 1
  resources[(2,0)] = 1

  get_subset_size = lambda elems: 1

  #all_subsets = set()
  #ps = all_partitions("abcd")
  #for part in ps:
  #  all_subsets.update(part)
  #get_all_partitions = lambda elems: list(all_partitions(elems))

  all_subsets = set(["a","b","c","d","ab","cd","abcd"])
  _all_partitions = {
    "a": [["a"]],
    "b": [["b"]],
    "c": [["c"]],
    "d": [["d"]],
    "ab": [["a","b"], ["ab"]],
    "cd": [["c","d"], ["cd"]],
    "abcd": [["a","b","c","d"], ["ab", "cd"]] }
  get_all_partitions = lambda elems: _all_partitions[elems]

  problem = Problem(
    all_subsets,
    get_all_partitions,
    get_subset_size,
    init_info,
    fini_info,
    resources,
    max_time)

  solve_problem(problem, use_full_solver_interface = True)

#exp02(2)
#exp03(max_time = 1, all_connected = False)
exp04(3)
