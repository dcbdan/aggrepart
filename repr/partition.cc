#include "partition.h"
#include "../utils/indexer.h"

partition_t partition_t::singleton(vector<uint64_t> shape) {
  vector<partdim_t> partdims;
  partdims.reserve(shape.size());
  for(auto const& sz: shape) {
    partdims.push_back(partdim_t::singleton(sz));
  }
  return partition_t { .partdims = partdims };
};

partition_t partition_t::intersect(partition_t lhs, partition_t rhs) {
  if(!vector_equal(lhs.total_shape(), rhs.total_shape())) {
    throw std::runtime_error("invalid args: partition_t::intersect");
  }
  int rank = lhs.partdims.size();

  vector<partdim_t> ret;
  ret.reserve(rank);
  for(int i = 0; i != rank; ++i) {
    ret.push_back(partdim_t::unions({
      lhs.partdims[i],
      rhs.partdims[i]
    }));
  }

  return partition_t {
    .partdims = ret
  };
}

vector<uint64_t> partition_t::total_shape() const {
  return vector_from_each_method(partdims, uint64_t, total);
}

int partition_t::num_parts() const {
  return product(this->block_shape());
}

vector<int> partition_t::block_shape() const {
  return vector_from_each_method(partdims, int, num_parts);
}

int partition_t::rank() const {
  return partdims.size();
}

hrect_t<uint64_t>
partition_t::get_region(vector<int> const& idxs) const
{
  if(idxs.size() != partdims.size()) {
    throw std::runtime_error("partition_t::get_hrect");
  }

  vector<tuple<uint64_t, uint64_t>> ret;
  ret.reserve(idxs.size());
  for(int i = 0; i != partdims.size(); ++i) {
    ret.push_back(partdims[i].which_vals(idxs[i]));
  }

  return ret;
}

hrect_t<int>
partition_t::get_exact_covering_blocks(
  hrect_t<uint64_t> const& region) const
{
  if(region.size() != partdims.size()) {
    throw std::runtime_error("partition_t::get_exact_region");
  }
  vector<tuple<int,int> > ret;
  ret.reserve(region.size());
  for(int i = 0; i != partdims.size(); ++i) {
    auto const& [beg,end] = region[i];
    ret.push_back(partdims[i].exact_region(beg,end));
  }
  return ret;
}

vector<tuple<int,int> >
partition_t::get_covering_blocks(
  vector<tuple<uint64_t,uint64_t>> const& region) const
{
  if(region.size() != partdims.size()) {
    throw std::runtime_error("partition_t::get_region");
  }
  vector<tuple<int,int> > ret;
  ret.reserve(region.size());
  for(int i = 0; i != partdims.size(); ++i) {
    auto const& [beg,end] = region[i];
    ret.push_back(partdims[i].region(beg,end));
  }
  return ret;
}

vector<int> partition_t::get_covering_block(
  vector<tuple<uint64_t,uint64_t>> const& hrect) const
{
  vector<tuple<int,int>> ret = get_covering_blocks(hrect);

  if(ret.size() != hrect.size()) {
    throw std::runtime_error("get index covering should not happen");
  }

  for(auto const& [b,e]: ret) {
    if(b >= e) {
      throw std::runtime_error("get index covering should not happen");
    }
    if(b + 1 != e) {
      throw std::runtime_error(
        "get_index_covering: cannot have multiple index");
    }
  }

  return vector_mapfst(ret);
}

tuple<vector<int>, touch_t>
partition_t::subset_covering_block(hrect_t<uint64_t> const& region) const
{
  vector<int> idx = get_covering_block(region);
  hrect_t<uint64_t> full_region = get_region(idx);

  vector<touch_t::dim_t> dims;
  for(int i = 0; i != region.size(); ++i) {
    auto const& [a,d] = full_region[i];
    auto const& [b,c] = region[i];
    // ------------------------------------------
    //   ^          ^              ^       ^
    //   a          b              c       d
    //   ---------------------------------- input
    //              ---------------         output
    dims.push_back(touch_t::dim_t {
      .d_inn = d-a,
      .d_out = c-b,
      .offset_inn = b-a,
      .offset_out = 0,
      .size = c-b
    });
  }

  touch_t op {
    .dims = dims,
    .op = std::nullopt
  };

  return {idx, op};
}

tuple<vector<int>, vector<int>, touch_t>
touch_from_covered_region(
  hrect_t<uint64_t> region,
  partition_t const& inn,
  partition_t const& out)
{
  if(inn.total_shape() != out.total_shape()) {
    throw std::runtime_error("touch_from_region: expect partitions with the same shape");
  }

  vector<int> inn_idx = inn.get_covering_block(region);
  hrect_t<uint64_t> inn_full_region = inn.get_region(inn_idx);

  vector<int> out_idx = out.get_covering_block(region);
  hrect_t<uint64_t> out_full_region = out.get_region(out_idx);

  vector<touch_t::dim_t> dims;
  for(int i = 0; i != region.size(); ++i) {
    auto const& [ibeg,iend] = inn_full_region[i];
    auto const& [obeg,oend] = out_full_region[i];
    auto const& [rbeg,rend] = region[i];
    dims.push_back(touch_t::dim_t {
      .d_inn = iend-ibeg,
      .d_out = oend-obeg,
      .offset_inn = rbeg-ibeg,
      .offset_out = rbeg-obeg,
      .size = rend-rbeg
    });
  }

  touch_t op {
    .dims = dims,
    .op = std::nullopt
  };

  return {inn_idx, out_idx, op};
}

bool operator==(partition_t const& lhs, partition_t const& rhs) {
  return vector_equal(lhs.partdims, rhs.partdims);
}
bool operator!=(partition_t const& lhs, partition_t const& rhs) {
  return !(lhs == rhs);
}

std::ostream& operator<<(std::ostream& out, partition_t const& p) {
  out << "partition" << p.partdims;
  return out;
}

vector<int> partition_t::block_to_index(int block) const {
  return index_to_idxs(block_shape(), block);
}

int partition_t::index_to_block(vector<int> const& index) const {
  return idxs_to_index(block_shape(), index);
}

std::function<hrect_t<int>(vector<int> const&)>
build_get_refi_index_region(
  partition_t const& coarse_part,
  partition_t const& refined_part)
{
  return [&](vector<int> const& coarse_bid) {
    hrect_t<uint64_t> coarse_region = coarse_part.get_region(coarse_bid);
    return refined_part.get_exact_covering_blocks(coarse_region);
  };
}

