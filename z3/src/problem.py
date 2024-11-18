from setup import *
from sol import *

class Problem:
  def __init__(
    self,
    all_subsets,
    get_all_partitions,
    get_subset_size,
    init_elem_locs_dict,
    fini_elems_locs_list,
    resource_counts_dict,
    max_time):

    self.all_subsets = all_subsets
    self.get_all_partitions = get_all_partitions
    self.get_subset_size = get_subset_size

    self.all_locs = set()
    for locs in init_elem_locs_dict.values():
      self.all_locs.update(locs)
    for _, locs in fini_elems_locs_list:
      self.all_locs.update(locs)
    for edge in resource_counts_dict.keys():
      self.all_locs.update(edge)

    self.init_elem_locs_dict = init_elem_locs_dict
    self.fini_elems_locs_list = fini_elems_locs_list

    self.max_time = max_time

    self.resource_counts_dict = resource_counts_dict

  def mk_node(self, elems, loc, time):
    name = "".join(elems) + "@" + str(loc) + "|" + str(time)
    return zz.Bool(name)
  def mk_move(self, elems, src, dst, time):
    if (src,dst) not in self.resource_counts_dict:
      return False # just in case, never create a variable when this can't happen
    name = "".join(elems) + "@" + str(src) + "->" + str(dst) + "|" + str(time)
    return zz.Bool(name)

  def parse_node(self, name):
    elems, name = name.split("@")
    loc, time = name.split("|")
    return elems, int(loc), int(time)
  def parse_move(self, name):
    elems, name = name.split("@")
    src, name = name.split("->")
    dst, time = name.split("|")
    return elems, int(src), int(dst), int(time)

  # TODO: remove?
  def _all_moves(self):
    """
    Return all (elems, sz_per_time, ntime, src, dst, time_lst).
    This implies that the valid moves are (elems, sz, ntime, src, dst, time) for time
    in [0, time_lst].
    """
    for elems in self.all_subsets:
      sz = self.get_subset_size(elems)
      for src in self.all_locs:
        for dst in self.all_locs:
          if src == dst:
            continue
          if (src, dst) not in self.resource_counts_dict:
            continue
          resource = self.resource_counts_dict[(src,dst)]
          if resource == 0:
            continue
          ntime = (sz + resource - 1) // resource
          sz_per_time = min(sz, resource)
          time_lst = self.max_time - ntime
          if time_lst >= 0:
            yield (elems, sz_per_time, ntime, src, dst, time_lst)

  # TODO: too much code duplication
  def _all_moves_starting_at(self, src, dst, time):
    if src == dst:
      return []
    if (src, dst) not in self.resource_counts_dict:
      return []
    resource = self.resource_counts_dict[(src,dst)]
    if resource == 0:
      return []
    ret = []
    for elems in self.all_subsets:
      sz = self.get_subset_size(elems)
      ntime = (sz + resource - 1) // resource
      if time + ntime <= self.max_time:
        ret.append(self.mk_move(elems, src, dst, time))
    return ret

  def init_time_zero(self):
    """
    At time zero, we
    1) have singleton elems from init_elem_locs_dict
    2) have unions of singleton elems at loc
    """
    ret = []
    for elems in self.all_subsets:
      if len(elems) == 1:
        elem = elems[0]
        has_locs = self.init_elem_locs_dict[elem]
        for loc in self.all_locs:
          has_loc = loc in has_locs
          ret.append(self.mk_node(elem, loc, 0) == has_loc)
      else:
        for loc in self.all_locs:
          has_it = all(loc in self.init_elem_locs_dict[elem] for elem in elems)
          ret.append(self.mk_node(elems, loc, 0) == has_it)
    return zz.And(*ret)

  # TODO: remove?
  def always_have_at_next_time(self):
    """
    If you have a (elems,loc,time), you also have it at (elems,loc,time+1)
    """
    for elems in self.all_subsets:
      for loc in self.all_locs:
        for time in range(self.max_time):
          has_prev = self.mk_node(elems, loc, time)
          has_next = self.mk_node(elems, loc, time + 1)
          ret.append(zz.Implies(has_prev, has_next))
    return zz.And(*ret)

  def _moved_to(self, elems, sz, dst, time, as_list = False):
    # Ok, find all the moves that arrive at time t
    ret = []
    for src in self.all_locs:
      if src == dst:
        continue
      if (src, dst) not in self.resource_counts_dict:
        continue
      resource = self.resource_counts_dict[(src,dst)]
      if resource == 0:
        continue
      ntime = (sz + resource - 1) // resource
      time_start = time - ntime
      if time_start >= 0:
        ret.append(self.mk_move(elems, src, dst, time_start))
    if as_list:
      return ret
    if len(ret) == 0:
      return False
    elif len(ret) == 1:
      ret = ret[0]
    elif len(ret) > 1:
      ret = zz.Or(*ret)
    return ret

  def _has_partition_here(self, partitions, loc, time, as_list = False):
    ret = []
    for partition in partitions:
      if len(partition) == 1:
        # Not including singleton partitions sincs
        continue
      has_partition = zz.And(*
        [self.mk_node(subset, loc, time) for subset in partition])
      ret.append(has_partition)

    if as_list:
      return ret

    if len(ret) == 0:
      return False
    return zz.Or(*ret)

  # TODO: too much code duplication
  def iff_vars(self, elems, loc, time):
    partitions = self.get_all_partitions(elems)
    sz = self.get_subset_size(elems)

    moved_here = self._moved_to(elems, sz, loc, time, as_list = True)
    partition_here = self._has_partition_here(partitions, loc, time, as_list = True)

    return moved_here, partition_here

  def has_node_iff(self):
    """
    for time > 0, (elems, loc, time) is true iff and only if
    either (1) a partition is available at time (not including singleton partitions),
           (2) a move of elems finishes here,
           (3) elems was available at the previous time
    """
    ret = []
    for elems in self.all_subsets:
      partitions = self.get_all_partitions(elems)
      sz = self.get_subset_size(elems)
      for loc in self.all_locs:
        for time in range(1, self.max_time + 1):
          avail_at_prev_time = self.mk_node(elems, loc, time-1)

          moved_here = self._moved_to(elems, sz, loc, time)

          partition_here = self._has_partition_here(partitions, loc, time)

          node = self.mk_node(elems, loc, time)
          node_val = zz.Or(avail_at_prev_time, moved_here, partition_here)
          ret.append(node == node_val)
    return zz.And(*ret)

  # TODO: remove?
  def has_partition_implies_has_elems(self):
    """
    If a partition of elems is here at loc, we have the union.
    This applies to all times > 0 since the constraint for time zero
    is encoded explicitly in `init_time_zero`.
    """
    ret = []
    for elems in self.all_subsets:
      if len(elems) == 1:
        # only 1 singleton partition here
        continue
      for partition in self.get_all_partitions(elems):
        if len(partition) == 1:
          # this is the singleton partition, so the condition
          # would be if has_here, then has_here
          continue
        for time in range(1, self.max_time + 1):
          for loc in self.all_locs:
            has_partition = zz.And(*
              [self.mk_node(subset, loc, time) for subset in partition])
            has_elems = self.mk_node(elems, loc, time)
            ret.append(zz.Implies(has_partition, has_elems))
    return zz.And(*ret)

  def has_before_and_after_move(self):
    """
    Note: all moves saturate resources. So if a sz = 4, resource from src to dst is 2,
          it will take two units of time, not 4.

    A move starting at time t finishes at time t + (sz + resource - 1) // resource.
    At each time a move occurs, it will use sz.. so if resource = 2, sz = 5, it'll use
    6 resources even though sz is 5. (this is for simplicity)

    Here, if we do a move, then we have the data at the src and the dst
    """
    ret = []
    for elems, sz_per_time, ntime, src, dst, time_lst in self._all_moves():
      for time in range(time_lst + 1):
        does_move  = self.mk_move(elems, src, dst, time)
        has_at_src = self.mk_node(elems, src, time)
        has_at_dst = self.mk_node(elems, dst, time + ntime)
        ret.append(zz.Implies(does_move, zz.And(has_at_src, has_at_dst)))
    return zz.And(*ret)

  def constrain_resources(self):
    """
    Each move has a size and occurs at times t, t + 1, ..., t + last.
    Make sure that at no point does a resource get used more than available
    """
    # TODO: double check
    totals = defaultdict(list)
    for elems, sz_per_time, ntime, src, dst, time_last_fst in self._all_moves():
      for time_fst in range(time_last_fst + 1):
        move = self.mk_move(elems, src, dst, time_fst)
        for time in range(time_fst, time_fst + ntime):
          totals[(src, dst, time)].append((move, sz_per_time))
    ret = []
    for (src, dst, time), move_szs in totals.items():
      total = zz.Sum(*[zz.If(move, sz_per_time, 0) for move, sz_per_time in move_szs])
      limit = self.resource_counts_dict[(src,dst)]
      ret.append(total <= limit)
    return zz.And(*ret)

  def init_fini_time(self):
    """
    Make sure that all fini_elems_locs_list are satisified at max time
    """
    ret = []
    for elems, locs in self.fini_elems_locs_list:
      for loc in locs:
        ret.append(self.mk_node(elems, loc, self.max_time) == True)
    return zz.And(*ret)

  def full_problem(self):
    return zz.And(
      self.init_time_zero(),
      self.has_node_iff(),
      self.has_before_and_after_move(),
      self.constrain_resources(),
      self.init_fini_time())

def solve_problem(problem, use_full_solver_interface = True):
  if not use_full_solver_interface:
    sat = problem.full_problem()
    zz.solve(sat)
  else:
    s = zz.Solver()
    s.add(problem.full_problem())

    if s.check() == zz.sat:
      model = s.model()
      #print(model)
      #print("----------------------------------------------")
      decls = model.decls()
      has_vars = set(x.name() for x in decls)

      def has_var(var):
        name = var.decl().name()
        return name in has_vars
      def print_vars(time):
        print("vars:")
        for subset in problem.all_subsets:
          for loc in sorted(problem.all_locs):
            var = problem.mk_node(subset, loc, time)
            if has_var(var) and model.eval(var):
              print("  ", var)
      def print_moves(time):
        print("moves:")
        for (src,dst) in problem.resource_counts_dict.keys():
          for var in problem._all_moves_starting_at(src, dst, time):
            if has_var(var) and model.eval(var):
              print("  ", var)

      print_vars(0)
      print("")
      for time in range(1, problem.max_time + 1):
        print_moves(time-1)
        print_vars(time)
        print("")

      # now, attempt to create the solution object
      def get_inns(elems, loc):
        if len(elems) == 1:
          if loc in problem.init_elem_locs_dict[elems]:
            return [], 0

        for time in range(1, problem.max_time + 1):
          moved_here_vars, partition_here_vars = problem.iff_vars(elems, loc, time)
          for did_move in moved_here_vars:
            if model.eval(did_move):
              name = did_move.decl().name()
              _, src, dst, time = problem.parse_move(name)
              return [(elems, src)], time
          for this_part_here in partition_here_vars:
            # this_part_here = zz.And(...)
            if model.eval(this_part_here):
              ret = []
              for var in this_part_here.children():
                name = var.decl().name()
                subset, _, _ = problem.parse_node(name)
                ret.append((subset, loc))
              return ret, time
        raise ValueError(f"get_inns passed to Sol.solve: could not find \"{elems}\" @ {loc}")

      pending = []
      for elems, locs in problem.fini_elems_locs_list:
        for loc in locs:
          inns, time = get_inns(elems, loc)
          pending.append((elems, loc, inns, time))
      pending.sort(key = lambda x: x[-1])

      items = []
      while len(pending) > 0:
        items.append(pending.pop())
        for inn_elems, inn_loc in items[-1][2]:
          found = False
          for p_elems, p_loc, _, _ in pending:
            if inn_elems == p_elems and inn_loc == p_loc:
              found = True
              break
          if not found:
            inn_inns, inn_time = get_inns(inn_elems, inn_loc)
            pending.append((inn_elems, inn_loc, inn_inns, inn_time))
        pending.sort(key = lambda x: x[-1])

      for elems, loc, inns, time in items:
        print(f"{elems}@{loc}|t={time}")
        for inn_elems, inn_loc in inns:
          print(f"  {inn_elems}@{inn_loc}")

      #########print("")
      #########sol = Sol(problem.init_elem_locs_dict, problem.fini_elems_locs_list)
      #########sol.solve(get_inns)
      #########print("")

      #########for node in sol.nodes:
      #########  print(f"{node.elems} @ {node.loc}")
      #########  for elems, loc in node.inns:
      #########    print(f"  {elems} @ {loc}")
      #########  print("")
    else:
      print("no-sat")

