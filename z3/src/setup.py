import z3 as zz

from itertools import combinations
import itertools

from functools import lru_cache

from collections import defaultdict

import heapq

# https://stackoverflow.com/questions/33987060/python-context-manager-that-measures-time
from time import perf_counter
class catchtime:
  def __init__(self, msg):
    self.msg = msg

  def __enter__(self):
    self.start = perf_counter()
    return self

  def __exit__(self, type, value, traceback):
    self.time = perf_counter() - self.start
    self.readout = f'Time: {self.time:.3f} seconds'
    print(self.msg, "|", self.readout)

def all_partitions(xs):
  if "," in xs:
    raise ValueError("can't have comma in all_partitions...")

  for partition_str in _all_partitions(xs):
    yield partition_str.split(",")

@lru_cache(maxsize=None)
def _all_partitions(xs):
  n = len(xs)
  if n == 0:
    raise ValueError("Uh oh")
  if n == 1:
    return set([xs])

  def combine(ss):
    return ",".join(sorted(ss.split(",")))

  ret = set()
  ret.add(xs)
  for i in range(1, n):
    if i > n-i:
      break
    for us in combinations(xs, i):
      us = "".join(us)
      vs = "".join(v for v in xs if v not in us)
      for vvs in _all_partitions(vs):
        ret.add(combine(vvs + "," + us))

  return ret

