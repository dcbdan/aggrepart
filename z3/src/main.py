from sys import stdin, stdout, stderr
from problem import Problem, solve_problem_v2

def correct_io_object(obj):
  obj.print = lambda msg: obj.write(msg + "\n")

  # the default readline method always returns the end newline
  # in the resulting string, which is annoying, so rewrite it
  obj._readline = obj.readline
  obj.readline = lambda: obj._readline()[:-1]
  return obj

stdin  = correct_io_object(stdin)
stdout = correct_io_object(stdout)
stderr = correct_io_object(stderr)

class ReaderWriter:
  def __init__(self):
    self.letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,./<>?"
    self.from_letters = {c: i for i, c in enumerate(self.letters)}

  def to_char(self, int_str):
    return self.letters[int(int_str)]

  def parse_elems_str(self, ss):
    ss = ss[1:-1]
    ret = "".join(sorted(self.to_char(s) for s in ss.split(",")))
    return ret
  def parse_locs_str(self, ss):
    ss = ss[1:-1]
    return list(sorted(map(int, ss.split(","))))

  def write_elems(self, elems):
    return "[" + ",".join([str(self.from_letters[e]) for e in elems]) + "]"

  def read_subset_line(self, line):
    line = line.split("|")
    size, parts_str = line[0], line[1:]
    size = int(size)
    all_partitions = []
    for parts in parts_str:
      ret = []
      for part in parts.split("@"):
        elems = self.parse_elems_str(part)
        ret.append(elems)
      all_partitions.append(ret)
    return size, all_partitions

  def read_subsets(self):
    subsets = set()
    sizes = {}
    partitions = {}

    while True:
      line = stdin.readline()
      if line == "stop":
        return subsets, sizes, partitions

      size, all_partitions = self.read_subset_line(line)
      elems = all_partitions[0][0]

      subsets.add(elems)
      sizes[elems] = size
      partitions[elems] = all_partitions

  def read_resources(self):
    ret = {}
    while True:
      line = stdin.readline()
      if line == "stop":
        return ret

      i,j = map(int, line.split(","))
      key = (i,j)
      if key in ret:
        ret[(i,j)] += 1
      else:
        ret[(i,j)] = 1

  def read_init(self):
    ret = {}
    while True:
      line = stdin.readline()
      if line == "stop":
        return ret
      # elem_as_int | {loc1,loc2,...}
      elem_str, locs_str = line.split("|")
      elem = self.to_char(elem_str)
      locs = self.parse_locs_str(locs_str)
      ret[elem] = locs

  def read_fini(self):
    ret = []
    while True:
      line = stdin.readline()
      if line == "stop":
        return ret
      # {elem1,elem2,...}|{loc1,loc2,...}
      elems_str, locs_str = line.split("|")
      elems = self.parse_elems_str(elems_str)
      locs  = self.parse_locs_str(locs_str)
      ret.append((elems, locs))

  def read_inputs(self):
    init_info  = None
    fini_info  = None
    subsets    = None
    sizes      = None
    partitions = None
    resources  = None
    while True:
      line = stdin.readline()
      if line == "start-subsets":
        subsets, sizes, partitions = self.read_subsets()
      elif line == "start-resources":
        resources = self.read_resources()
      elif line == "start-init":
        init_info = self.read_init()
      elif line == "start-fini":
        fini_info = self.read_fini()
      elif line == "end":
        break
    is_none = lambda x: x is None
    if any(map(is_none, [init_info, fini_info, subsets, sizes, partitions, resources])):
      raise ValueError("did not read everything")

    return init_info, fini_info, subsets, sizes, partitions, resources

  def print_solution(self, info):
    # each info contains elems, loc, inns, time
    #   where elems = set of locs
    #         loc   = location
    #         inns  = list of (elems, loc)
    #         time  = int
    for elems, loc, inns, time in info:
      msg = str(loc) + "|" + str(time) + "|" + self.write_elems(elems) + "|"
      for inn_elems, inn_loc in inns:
        msg += self.write_elems(inn_elems) + "@" + str(inn_loc) + "|"
      msg = msg[:-1]
      stdout.print(msg)
    stdout.print("done")

def example_debugger():
  with open("debug", "w") as debug:
    debug = correct_io_object(debug)
    while True:
      line = stdin.readline()
      if line == "end":
        break
      debug.print(line)
      debug.flush()
  stdout.print("done")

if __name__ == "__main__":
  # Here is the idea:
  # 1. read in all the inputs
  # 2. get a solution
  # 3. write out the solution

  self = ReaderWriter()
  init_info, fini_info, subsets, sizes, partitions, resources = self.read_inputs()
  with open("debug", "w") as debug:
    debug = correct_io_object(debug)
    debug.print("init_info")
    debug.print(str(init_info))
    debug.print("")

    debug.print("fini_info")
    debug.print(str(fini_info))
    debug.print("")

    debug.print("subsets")
    debug.print(str(subsets))
    debug.print("")

    debug.print("sizes")
    debug.print(str(sizes))
    debug.print("")

    debug.print("partitions")
    debug.print(str(partitions))
    debug.print("")

    debug.print("resources")
    debug.print(str(resources))
    debug.print("")

  success = False
  for max_time in range(1, 8 + 1):
    problem = Problem(
      subsets,
      lambda x: partitions[x],
      lambda x: sizes[x],
      init_info,
      fini_info,
      resources,
      max_time)

    maybe = solve_problem_v2(problem)
    if maybe is not None:
      self.print_solution(maybe)
      success = True
      break
  if not success:
    stdout.print("no-sat")


