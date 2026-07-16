// Replays fixtures exported from the Python reference (async_infer.aggregate_chunks +
// AsyncRunner._merge ramp_in) against the C++ port. Regenerate fixtures with
// test/gen_fixtures.py in the sim container.
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "smolvla_client/algorithm.hpp"

namespace
{

smolvla::Chunk read_chunk(std::ifstream & in, std::size_t rows, std::size_t dim)
{
  smolvla::Chunk c;
  c.rows = rows;
  c.dim = dim;
  c.data.resize(rows * dim);
  for (auto & v : c.data) {in >> v;}
  return c;
}

}  // namespace

TEST(Aggregate, MatchesPythonReference)
{
  std::ifstream in(FIXTURE_PATH);
  ASSERT_TRUE(in.good()) << "fixture file missing: " FIXTURE_PATH
                         << " (run test/gen_fixtures.py in the sim container)";
  int n_cases = 0;
  in >> n_cases;
  ASSERT_GT(n_cases, 0);

  for (int c = 0; c < n_cases; ++c) {
    std::string how;
    std::size_t old_rows, new_rows, dim;
    int ramp, has_last;
    in >> how >> old_rows >> new_rows >> dim >> ramp >> has_last;

    smolvla::Chunk old_q = read_chunk(in, old_rows, dim);
    smolvla::Chunk new_c = read_chunk(in, new_rows, dim);
    std::vector<float> last(dim);
    if (has_last) {
      for (auto & v : last) {in >> v;}
    }
    smolvla::Chunk exp = read_chunk(in, new_rows, dim);

    smolvla::Chunk got = smolvla::aggregate_chunks(old_q, new_c, how);
    if (ramp > 0 && has_last) {
      smolvla::ramp_in(got, last, ramp);
    }

    ASSERT_EQ(got.rows, exp.rows) << "case " << c << " (" << how << ")";
    ASSERT_EQ(got.dim, exp.dim) << "case " << c;
    for (std::size_t i = 0; i < exp.data.size(); ++i) {
      EXPECT_NEAR(got.data[i], exp.data[i], 1e-5f)
        << "case " << c << " (" << how << ", ramp=" << ramp << ") element " << i;
    }
  }
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
