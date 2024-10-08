#include "touch.h"

touch_t touch_t::inn_to_write() const {
  vector<dim_t> ds;
  for(dim_t const& x: dims) {
    ds.push_back(dim_t {
      .d_inn      = x.d_inn,
      .d_out      = x.size,
      .offset_inn = x.offset_out,
      .offset_out = 0,
      .size       = x.size
    });
  }
  return touch_t { .dims = ds, .op = op };
}

touch_t touch_t::write_to_out() const {
  vector<dim_t> ds;
  for(dim_t const& x: dims) {
    ds.push_back(dim_t {
      .d_inn      = x.size,
      .d_out      = x.d_out,
      .offset_inn = 0,
      .offset_out = x.offset_out,
      .size       = x.size
    });
  }
  return touch_t { .dims = ds, .op = op };
}

bool touch_t::is_identity() const {
  if(bool(op)) {
    return false;
  }

  for(dim_t const& d: dims) {
    if(d.d_inn == d.d_out && d.offset_inn == d.size &&
                             d.offset_out == d.size)
    {
      // ok, this is the same
    } else {
      return false;
    }
  }

  return true;
}

touch_t touch_t::intersect(
  hrect_t<uint64_t> inn_region,
  hrect_t<uint64_t> out_region,
  optional<castable_t> op)
{
  if(inn_region.size() != out_region.size()) {
    throw std::runtime_error("invalid touch intersect args");
  }

  mid_region = hrect_intersect(inn_region, out_region);

  vector<dim_t> dims;
  for(int idx = 0; idx != inn_region.size(); ++idx) {
    auto const& [ib,ie] = inn_region[i];
    auto const& [ob,oe] = out_region[i];
    auto const& [mb,me] = mid_region[i];

    dims.push_back(dim_t {
      .d_inn      = ie-ib,
      .d_out      = oe-ob,
      .offset_inn = mb-ib,
      .offset_out = mb-ob,
      .size       = me-mb
    });
  }

  return touch_t {
    .dims = dims,
    .op = op
  };
}

