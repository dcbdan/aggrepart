#include "exec_list.h"

#include "exec_list.h"

std::ostream& operator<<(std::ostream& out, exec_item_t const& item) {
  if(item.is_move()) {
    auto const& m = item.get_move();
    out << "move|elems=" << m.elems << "|src->dst=" << m.src << "->" << m.dst;
    out << "|" << "time=" << m.start_time << "," << m.end_time;
  } else {
    auto const& f = item.get_form();
    out << "form|elems=" << f.elems << "|loc=" << f.loc << "|time=" << f.time;
    for(auto const& es: f.inn_elems) {
      out << "|" << es;
    }
  }
  return out;
}
