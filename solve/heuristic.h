#pragma once
#include "../utils/setup.h"

#include "sol.h"

// Given an unsolved sol_t, solve it.
// 
// If any of subsueqent nodes are a subset, depend on that subset
// Otherwise, if there are local elements, pop them off and depend on
//   the previous location
// Otherwise, solve right away
////void heuristic01(sol_t& sol);

// Always chain if possible otherwise do a naive solution
void heuristic02(sol_t& sol);

// pop off all local objects and recv the rest from the previous loc
void heuristic03_ring(sol_t& sol, vector<int> loc_order);
