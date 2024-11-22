#pragma once
#include "../utils/setup.h"

struct exec_item_t {
  struct move_t {
    set<int> elems;
    int src;
    int dst;
    int start_time;
    int end_time;
  };
  struct form_t {
    set<int> elems;
    int loc;
    int time;
    vector<set<int>> inn_elems;
  };

  std::variant<move_t, form_t> item;

  bool is_move() const { return std::holds_alternative<move_t>(item); }
  bool is_form() const { return std::holds_alternative<form_t>(item); }

  move_t const& get_move() const { return std::get<move_t>(item); }
  form_t const& get_form() const { return std::get<form_t>(item); }

  set<int> const& elems() const {
    if(is_move()) { return get_move().elems; }
    if(is_form()) { return get_form().elems; }
    throw std::runtime_error("should not reach");
  }
};

using exec_list_t = vector<exec_item_t>;

