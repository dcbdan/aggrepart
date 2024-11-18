class Sol:
  class Node:
    def __init__(self, sol, elems, loc):
      self.sol = sol
      self.elems = elems
      self.loc = loc
      self.inns = [] # list of subset, loc pairs
      self.time = None

    def is_set(self):
      if len(self.inns) == 0:
        if self.sol.is_init(self.elems, self.loc):
          return True
        else:
          return False
      else:
        return True

  def __init__(self, init_elem_locs_dict, fini_elems_locs_list):
    self.init_elem_locs_dict = init_elem_locs_dict

    self.nodes = []
    for elems, locs in fini_elems_locs_list:
      for loc in locs:
        self.nodes.append(Sol.Node(self, elems, loc))

  def is_init(self, elems, loc):
    if len(elems) != 1:
      return False
    if elems in self.init_elem_locs_dict:
      return loc in self.init_elem_locs_dict[elems]
    else:
      return False

  def add_node_after(self, idx, elems, loc):
    for node in self.nodes[idx:]:
      if node.elems == elems and node.loc == loc:
        return
    self.nodes.append(Sol.Node(self, elems, loc))

  def solve(self, f):
    """
    f: given (elems, locs), return
       1. list of (elems,loc) pairs that are the nodes inputs
       2. the time that the node occurs
    """
    # TODO TODO: this is no good, things can be out of order with respect to time
    idx = 0
    while idx < len(self.nodes):
      node = self.nodes[idx]
      if self.is_init(node.elems, node.loc):
        node.time = 0
      elif len(node.inns) != 0 or node.time is not None:
        raise ValueError("node already set...")
      else:
        node.inns, node.time = f(node.elems, node.loc)
        for elems, loc in node.inns:
          self.add_node_after(idx, elems, loc)
      idx += 1
    self.nodes.sort(key = lambda node: -1*node.time)
