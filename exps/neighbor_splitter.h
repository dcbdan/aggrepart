struct splitter_from_neighbors_t {
  map<int, set<int>> const all_neighbors;

  int cross_numa_node(int nlocs, int src) {
    if(nlocs != 8) {
      throw std::runtime_error("implement shortest path");
    }
    if(src == 0) { return 4; }
    if(src == 1) { return 5; }
    if(src == 2) { return 6; }
    if(src == 3) { return 7; }
    if(src == 4) { return 0; }
    if(src == 5) { return 1; }
    if(src == 6) { return 2; }
    if(src == 7) { return 3; }
    throw std::runtime_error("should not reach...");
  }

  vector<sol_t::info_t> operator()(
    sol_t const& sol_init, int start_id) 
  {
    using info_t  = sol_t::info_t;
    using node_t  = sol_t::node_t;
    using which_t = sol_t::which_t;

    node_t const& node = sol_init.nodes[start_id];

    set<int> rem_elems = node.elems();
    int src = node.loc();

    auto pop = [&](int loc) {
      set<int> ret;
      set<int> es = rem_elems;
      for(int const& elem: es) {
        if(sol_init.has_input_elem(elem, loc)) {
          ret.insert(elem);
          rem_elems.erase(elem);
        }
      }
      return ret;
    };

    vector<info_t> ret;
    for(int const& local_elem: pop(src)) {
      ret.push_back(info_t { .elems = { local_elem }, .loc = src });
    }

    set<int> const& neighbors = all_neighbors.at(src);

    map<int, set<int>> to_elems;
    for(int const& dst: all_neighbors.at(src)) {
      to_elems[dst] = pop(dst);
    }
    int extra = cross_numa_node(8, src);
    for(int const& elem: rem_elems) {
      to_elems[extra].insert(elem);
    }
    
    for(auto const& [dst, es]: to_elems) {
      if(es.size() > 0) {
        ret.push_back(info_t { .elems = es, .loc = dst });
      }
    }

    //auto iter = neighbors.begin();
    //int first_dst = *iter++;
    //for(; iter != neighbors.end(); ++iter) {
    //  int dst = *iter;
    //  set<int> recv = pop(dst);
    //  if(recv.size() > 0) {
    //    info_t info { .elems = recv, .loc = dst };
    //    ret.push_back(info);
    //  }
    //}

    //if(rem_elems.size() > 0) {
    //  info_t info { .elems = rem_elems, .loc = first_dst };
    //  ret.push_back(info);
    //}

    return ret;
  }

  static splitter_from_neighbors_t make_v100(int nlocs) {
    map<int, set<int>> all_neighbors;

    auto ii = [&](int src, int dst) {
      if(src < nlocs && dst < nlocs) {
        all_neighbors[src].insert(dst);
      }
    };
    auto iii = [&](int src, set<int> dsts) {
      for(int const& d: dsts) {
        ii(src, d);
      }
    };
  
    iii(0, {1,2,3,4});
    iii(1, {0,2,3,5});
    iii(2, {0,1,3,6});
    iii(3, {0,1,2,7});
    iii(4, {0,5,6,7});
    iii(5, {1,4,6,7});
    iii(6, {2,4,5,7});
    iii(7, {3,4,5,6});
    
    return splitter_from_neighbors_t {
      .all_neighbors = all_neighbors
    };
  }
};
