#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <variant>
#include <tuple>
#include <set>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <random>
#include <queue>
#include <chrono>
#include <fstream>

using std::vector;
using std::tuple;
using std::set;
using std::map;
using std::optional;
using std::string;

#define DOUT(x) \
  std::cout << x << std::endl;
#define DLINEOUT(x) \
  std::cout << "Line " << __LINE__ << " | " << x << std::endl;
#define DLINE \
  DLINEOUT(' ')
#define DLINEFILEOUT(x) \
  std::cout << __FILE__ << " @ " << __LINE__ << " | " << x << std::endl;
#define DLINEFILE \
  DLINEFILEOUT(' ')

#define vector_from_each_member(items, member_type, member_name) [](auto const& xs) { \
    std::vector<member_type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return x.member_name; }); \
    return ret; \
  }(items)

#define vector_from_each_method(items, type, method) [](auto const& xs) { \
    std::vector<type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return x.method(); }); \
    return ret; \
  }(items)

#define vector_from_each_tuple(items, which_type, which) [](auto const& xs) { \
    std::vector<which_type> ret; \
    ret.reserve(xs.size()); \
    std::transform( \
      xs.begin(), \
      xs.end(), \
      std::back_inserter(ret), \
      [](auto const& x){ return std::get<which>(x); }); \
    return ret; \
  }(items)


template <typename T>
bool set_equal(set<T> const& xs, set<T> const& ys) {
  if(xs.size() != ys.size()) {
    return false;
  }
  auto xiter = xs.begin();
  auto yiter = ys.begin();
  for(; xiter != xs.end(); ++xiter, ++yiter) {
    if(*xiter != *yiter) {
      return false;
    }
  }

  return true;
}

template <typename T>
bool vector_equal(vector<T> const& xs, vector<T> const& ys) {
  if(xs.size() != ys.size()) {
    return false;
  }
  for(int i = 0; i != xs.size(); ++i) {
    if(xs[i] != ys[i]) {
      return false;
    }
  }

  return true;
}

// Remove the duplicates in a sorted list
template <typename T>
void vector_remove_duplicates(vector<T>& xs) {
  std::size_t i = 0;
  std::size_t j = 0;
  while(j != xs.size()) {
    xs[i++] = xs[j++];
    while(j != xs.size() && xs[i-1] == xs[j]) {
      ++j;
    }
  }
  xs.resize(i);
}

template <typename T>
T product(vector<T> const& xs)
{
  T ret = 1;
  for(T const& x: xs) {
    ret *= x;
  }
  return ret;
}

template <typename T>
void print_vec(std::ostream& out, vector<T> const& xs)
{
  out << "{";
  if(xs.size() >= 1) {
    out << xs[0];
  }
  if(xs.size() > 1) {
    for(int i = 1; i != xs.size(); ++i) {
      out << "," << xs[i];
    }
  }
  out << "}";
}

template <typename T>
void print_set(std::ostream& out, set<T> const& xs)
{
  auto iter = xs.begin();
  out << "{";
  if(xs.size() >= 1) {
    out << (*iter++);
  }
  for(; iter != xs.end(); ++iter) {
    out << "," << (*iter);
  }
  out << "}";
}

template <typename T>
void print_vec(vector<T> const& xs)
{
  print_vec(std::cout, xs);
}

template <typename T>
void print_set(set<T> const& xs) {
  print_set(std::cout, xs);
}

vector<int> divide_evenly_int(int num_parts, int n);
vector<uint64_t> divide_evenly(int num_parts, uint64_t n);

template <typename T>
[[nodiscard]] vector<T> vector_concatenate(vector<T> vs, vector<T> const& add_these) {
  vs.reserve(vs.size() + add_these.size());
  for(auto const& x: add_these) {
    vs.push_back(x);
  }
  return vs;
}
template <typename T>
void vector_concatenate_into(vector<T>& vs, vector<T> const& add_these) {
  vs.reserve(vs.size() + add_these.size());
  for(auto const& x: add_these) {
    vs.push_back(x);
  }
}

// Take a bunch of sorted lists and merge em into a single sorted list
// This is nlogn, but for n < 100, it's pretty fast cuz std::sort is fast.
// For lorge n, use a nlogk algorthm where k = xs.size(). An implementation
// tested wasnt faster until n > 5000.
template <typename T>
vector<T> vector_sorted_merges(vector<vector<T>> const& xs) {
  vector<T> ret;
  for(auto const& x: xs) {
    vector_concatenate_into(ret, x);
  }
  std::sort(ret.begin(), ret.end());
  return ret;
}

template <typename T>
vector<T> vector_iota(int n) {
  return vector_iota<T>(n, 0);
}

template <typename T>
vector<T> vector_iota(int size, int start) {
  vector<T> ret(size);
  std::iota(ret.begin(), ret.end(), T(start));
  return ret;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, vector<T> const& ts) {
  print_vec(out, ts);
  return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, set<T> const& ts) {
  print_set(out, ts);
  return out;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& out, tuple<T, U> const& x12) {
  auto const& [x1,x2] = x12;
  out << "tup[" << x1 << "|" << x2 << "]";
  return out;
}

std::mt19937& random_gen();

void set_seed(int seed);

template <typename T>
using priority_queue_least = std::priority_queue<T, vector<T>, std::greater<T>>;
// For priority_queue_least, the top most element is the smallest,
// which is the opposite behaviour of priority_queue which puts the
// largest element at the top.

#define clock_now std::chrono::high_resolution_clock::now

using timestamp_t = decltype(clock_now());

struct raii_print_time_elapsed_t {
  raii_print_time_elapsed_t(string msg):
    msg(msg), start(clock_now()), out(std::cout)
  {}

  raii_print_time_elapsed_t():
    msg(), start(clock_now()), out(std::cout)
  {}

  ~raii_print_time_elapsed_t() {
    auto end = clock_now();
    using namespace std::chrono;
    auto duration = (double) duration_cast<microseconds>(end - start).count()
                  / (double) duration_cast<microseconds>(1s         ).count();

    if(msg.size() > 0) {
      out << msg << " | ";
    }
    out << "Total Time (seconds): " << duration << std::endl;
  }

  string const msg;
  timestamp_t const start;
  std::ostream& out;
};

using gremlin_t = raii_print_time_elapsed_t;

void hash_combine_impl(std::size_t& seed, std::size_t value);

template <typename T>
vector<T> _reverse_variadic_to_vec(T i) {
  vector<T> x(1, i);
  return x;
}
template <typename T, typename... Args>
vector<T> _reverse_variadic_to_vec(T i, Args... is) {
  vector<T> x = _reverse_variadic_to_vec(is...);
  x.push_back(i);
  return x;
}

template <typename T, typename... Args>
vector<T> variadic_to_vec(Args... is) {
  vector<T> x = _reverse_variadic_to_vec(is...);
  std::reverse(x.begin(), x.end());
  return x;
}

template <typename T>
optional<string> check_concat_shapes(
  int dim,
  vector<vector<T>> const& shapes)
{
  if(shapes.size() == 0) {
    return "cannot be empty list of shapes";
  }

  // they should all have the same rank
  int rank = shapes[0].size();
  for(int i = 1; i != shapes.size(); ++i) {
    if(shapes[i].size() != rank) {
      return "invalid input size";
    }
  }

  if(dim < 0 || dim >= rank) {
    return "invalid dim";
  }

  // every dim should be the same, except dim
  vector<T> dim_parts;
  for(int r = 0; r != rank; ++r) {
    if(r != dim) {
      T d = shapes[0][r];
      for(int i = 1; i != shapes.size(); ++i) {
        if(shapes[i][r] != d) {
          return "non-concat dimensions do not line up";
        }
      }
    }
  }

  return std::nullopt;
}

template <typename T, typename U>
vector<T> vector_mapfst(vector<tuple<T, U>> const& xys) {
  return vector_from_each_tuple(xys, T, 0);
}

template <typename T, typename U>
vector<U> vector_mapsnd(vector<tuple<T, U>> const& xys) {
  return vector_from_each_tuple(xys, T, 1);
}

template <typename T>
void set_append(set<T>& out, set<T> const& other) {
  for(T const& v: other) {
    out.insert(v);
  }
}

template <typename T>
[[nodiscard]] set<T> set_union(set<T> const& lhs, set<T> const& rhs) {
  set<T> ret = lhs;
  set_append(ret, rhs);
  return ret;
}
template <typename T>
void set_union_into(set<T>& ret, set<T> const& xs) {
  set_append(ret, xs);
}

template <typename T>
set<T> set_from_vector(vector<T> const& xs) {
  return set<T>(xs.begin(), xs.end());
}

template <typename T>
T parse_with_ss(string const& s)
{
  T out;
  std::istringstream ss(s);
  ss >> out;
  return out;
}

// Parse [], [T], [T,...,T] with parse_with_ss for each element
template <typename T>
vector<T> parse_vector(string const& s, char sep = ',', char open = '[', char close = ']')
{
  vector<T> ret;

  if(s.size() < 2) {
    throw std::runtime_error("failed to parse vector: len < 2");
  }
  if(*s.begin() != open or *(s.end() - 1) != close) {
    throw std::runtime_error("parse vector: needs brackets");
  }

  auto xx = s.begin() + 1;
  auto end = s.end() - 1;
  while(xx != end) {
    auto yy = std::find(xx, end, sep);
    if(xx == yy) {
      throw std::runtime_error("parse_vector: empty substring");
    }
    ret.push_back(parse_with_ss<T>(std::string(xx,yy)));
    if(yy == end) {
      xx = end;
    } else {
      xx = yy + 1;
    }
  }

  return ret;
}

template <typename T>
string write_with_ss(T const& val)
{
  std::ostringstream ss;
  ss << val;
  return ss.str();
}

// Find the last true element
// Assumption: evaluate returns all trues then all falses.
// If there are no trues: return end
// If there are all trues: return end-1
template <typename Iter, typename F>
Iter binary_search_find(Iter beg, Iter end, F evaluate)
{
  if(beg == end) {
    return end;
  }
  if(!evaluate(*beg)) {
    return end;
  }

  decltype(std::distance(beg,end)) df;
  while((df = std::distance(beg, end)) > 2) {
    Iter mid = beg + (df / 2);
    if(evaluate(*mid)) {
      beg = mid;
    } else {
      end = mid;
    }
  }

  if(df == 1) {
    return beg;
  }

  if(evaluate(*(end - 1))) {
    return end-1;
  } else {
    return beg;
  }
}

void* increment_void_ptr(void* ptr, uint64_t size);
void const* increment_void_ptr(void const* ptr, uint64_t size);
