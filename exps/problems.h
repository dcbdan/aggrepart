tuple<placement_t, placement_t>
make_pls_matrix_all_reduce(
  uint64_t nrow,
  uint64_t ncol,
  int nlocs,
  bool split_further = false)
{
  partition_t partition { vector<partdim_t> {
    partdim_t::singleton(nrow),
    partdim_t::singleton(ncol)
  }};

  placement_t init = placement_t::make(partition, nlocs);
  placement_t fini = placement_t::make(partition, 1);

  for(int i = 0; i != nlocs; ++i) {
    set<int>& s = init.locations.get()[i];
    s.insert(i);
  }

  {
    set<int>& s = fini.locations.get()[0];
    for(int i = 0; i != nlocs; ++i) {
      s.insert(i);
    }
  }

  if(split_further) {
    partition_t p {
      .partdims = vector<partdim_t> {
        partdim_t::split(nrow, nlocs),
        partdim_t::singleton(ncol)
      }
    };
    return { init.construct_refinement(p), fini };
  } else {
    return { init, fini };
  }
}

// -------------    -------------
// | a,e | b,f |    | a,e,b,f    |
// | 0 1 | 2 3 |    |        @0,1|
// |-----------| -> |------------|
// | c,g | d,h |    | c,g,d,h    |
// | 0 1 | 2 3 |    |        @2,3|
// ------------     -------------
//
// Here, the init relation is replicated
// twice into column strips and the fini
// relation is partitioned into rows,
// each row duplicated on two locs
tuple<placement_t, placement_t>
make_pls_canonical_4locs_rows_to_cols(
  uint64_t nrow,
  uint64_t ncol)
{
  partition_t part_init { vector<partdim_t> {
    partdim_t::split(nrow, 2),
    partdim_t::split(ncol, 2)
  }};

  partition_t part_fini { vector<partdim_t> {
    partdim_t::split(nrow, 2),
    partdim_t::split(ncol, 1)
  }};

  placement_t init = placement_t::make(part_init, 2);
  init.get_locs({ 0, 0 }, 0).insert(0); // a
  init.get_locs({ 0, 0 }, 1).insert(1); // e
  init.get_locs({ 0, 1 }, 0).insert(0); // b
  init.get_locs({ 0, 1 }, 1).insert(1); // f
  init.get_locs({ 1, 0 }, 0).insert(0); // c
  init.get_locs({ 1, 0 }, 1).insert(1); // g
  init.get_locs({ 1, 1 }, 0).insert(0); // d
  init.get_locs({ 1, 1 }, 1).insert(1); // h

  placement_t fini = placement_t::make(part_fini, 1);
  fini.get_locs({ 0, 0 }, 0).insert(0);
  fini.get_locs({ 0, 0 }, 0).insert(1);
  fini.get_locs({ 1, 0 }, 0).insert(2);
  fini.get_locs({ 1, 0 }, 0).insert(3);

  return { init, fini };
}

tuple<placement_t, placement_t>
make_pls_row_strip_to_col_strip(
 uint64_t nrow,
 uint64_t ncol,
 int nlocs)
{
  partition_t part_init { vector<partdim_t> {
    partdim_t::split(nrow, nlocs),
    partdim_t::split(ncol, nlocs)
  }};

  partition_t part_fini { vector<partdim_t> {
    partdim_t::split(nrow, 1),
    partdim_t::split(ncol, nlocs)
  }};

  placement_t init = placement_t::make(part_init, 1);
  placement_t fini = placement_t::make(part_fini, 1);

  for(int r = 0; r != nlocs; ++r) {
  for(int c = 0; c != nlocs; ++c) {
    init.get_locs({r,c}, 0).insert(r);
  }}

  for(int c = 0; c != nlocs; ++c) {
    fini.get_locs({0,c}, 0).insert(c);
  }

  return { init, fini };
}
