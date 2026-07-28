import os
import subprocess
import textwrap
from pathlib import Path

import pytest


def test_target_tensor_pool_with_fake_adapter(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    source = tmp_path / "target_tensor_pool_test.cc"
    binary = tmp_path / (
        "target_tensor_pool_test.exe" if os.name == "nt" else "target_tensor_pool_test"
    )
    source.write_text(_TARGET_TENSOR_POOL_TEST_SOURCE)

    _run(
        [
            os.environ.get("CXX") or "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{repo_root / 'src'}",
            str(source),
            "-o",
            str(binary),
        ],
        skip_if_missing=True,
    )
    _run([str(binary)])


def _run(command, skip_if_missing=False):
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as error:
        if skip_if_missing:
            pytest.skip(f"`{command[0]}` is not available: {error}")

        raise AssertionError(f"`{command[0]}` was not produced") from error
    except subprocess.CalledProcessError as error:
        output = "\n".join((error.stdout, error.stderr)).strip()
        raise AssertionError(output) from error


_TARGET_TENSOR_POOL_TEST_SOURCE = textwrap.dedent(
    r"""
    #include "target_tensor_pool.h"

    #include <cassert>
    #include <cstddef>
    #include <memory>
    #include <type_traits>
    #include <utility>
    #include <vector>

    struct FakeTensor {
      explicit FakeTensor(void* data) : data{data} {}

      void* data;
    };

    struct FakeAdapter {
      struct State {
        State(int id, void* data) : id{id}, data{data} {
          ++states_created;
          ++live_states;
        }

        State(const State&) = delete;
        State& operator=(const State&) = delete;

        ~State() {
          ++states_destroyed;
          --live_states;
        }

        int id;
        void* data;
      };

      using NativeHandle = State&;

      static std::unique_ptr<State> Create(const FakeTensor& tensor) {
        return std::make_unique<State>(next_id++, tensor.data);
      }

      static void Rebind(State& state, const FakeTensor& tensor) noexcept {
        state.data = tensor.data;
      }

      static NativeHandle Native(State& state) { return state; }

      inline static int next_id = 1;
      inline static int states_created = 0;
      inline static int states_destroyed = 0;
      inline static int live_states = 0;
    };

    using Pool = infini::ops::TargetTensorPool<FakeAdapter>;
    using Lease = Pool::Lease;

    static_assert(std::is_default_constructible_v<Lease>);
    static_assert(!std::is_copy_constructible_v<Lease>);
    static_assert(!std::is_copy_assignable_v<Lease>);
    static_assert(std::is_nothrow_move_constructible_v<Lease>);
    static_assert(std::is_nothrow_move_assignable_v<Lease>);
    static_assert(std::is_nothrow_destructible_v<Lease>);
    static_assert(
        std::is_same_v<decltype(std::declval<const Lease&>().Native()),
                       FakeAdapter::State&>);
    static_assert(!std::is_copy_constructible_v<Pool>);
    static_assert(!std::is_copy_assignable_v<Pool>);
    static_assert(std::is_nothrow_move_constructible_v<Pool>);
    static_assert(!std::is_move_assignable_v<Pool>);
    static_assert(noexcept(FakeAdapter::Rebind(
        std::declval<FakeAdapter::State&>(),
        std::declval<const FakeTensor&>())));

    void TestSequentialReuseAndRebind() {
      Pool pool;
      int first_data = 1;
      int second_data = 2;
      int first_id = 0;
      const int created_before = FakeAdapter::states_created;

      {
        auto first = pool.Acquire(FakeTensor{&first_data});
        first_id = first.Native().id;
        assert(first.Native().data == &first_data);
      }

      auto reused = pool.Acquire(FakeTensor{&second_data});
      assert(reused.Native().id == first_id);
      assert(reused.Native().data == &second_data);
      assert(FakeAdapter::states_created == created_before + 1);
    }

    void TestNestedAcquireAndStableGrowth() {
      Pool pool;
      int outer_data = 1;
      int nested_data[64] = {};
      int extra_data = 2;
      auto outer = pool.Acquire(FakeTensor{&outer_data});
      auto* outer_state = &outer.Native();
      const int outer_id = outer.Native().id;
      std::vector<Lease> nested;

      for (std::size_t i = 0; i < 64; ++i) {
        nested.push_back(pool.Acquire(FakeTensor{&nested_data[i]}));
        assert(nested.back().Native().id != outer_id);
        assert(&outer.Native() == outer_state);
        assert(outer.Native().data == &outer_data);
      }

      const int created_before_extra = FakeAdapter::states_created;
      auto extra = pool.Acquire(FakeTensor{&extra_data});
      assert(FakeAdapter::states_created == created_before_extra + 1);
      assert(&outer.Native() == outer_state);
      assert(extra.Native().id != outer_id);
    }

    void TestLeaseMovesReleaseExactlyOneEntry() {
      Pool pool;
      int source_data = 1;
      int destination_data = 2;
      int reused_data = 3;
      int nested_data = 4;
      int released_data = 5;
      auto source = pool.Acquire(FakeTensor{&source_data});
      const int source_id = source.Native().id;
      auto destination = pool.Acquire(FakeTensor{&destination_data});
      const int destination_id = destination.Native().id;

      destination = std::move(source);
      assert(destination.Native().id == source_id);

      auto reused_destination = pool.Acquire(FakeTensor{&reused_data});
      assert(reused_destination.Native().id == destination_id);

      Lease moved{std::move(destination)};
      auto nested = pool.Acquire(FakeTensor{&nested_data});
      assert(nested.Native().id != source_id);
      moved = {};

      auto reused_source = pool.Acquire(FakeTensor{&released_data});
      assert(reused_source.Native().id == source_id);
      assert(reused_source.Native().data == &released_data);
    }

    void TestPoolVectorReallocationPreservesLiveEntries() {
      int outer_data = 1;
      int nested_data = 2;
      std::vector<Pool> pools;
      pools.emplace_back();
      auto outer = pools.front().Acquire(FakeTensor{&outer_data});
      auto* outer_state = &outer.Native();
      const int outer_id = outer.Native().id;

      for (int i = 0; i < 64; ++i) {
        pools.emplace_back();
      }

      assert(&outer.Native() == outer_state);
      assert(outer.Native().data == &outer_data);
      auto nested = pools.front().Acquire(FakeTensor{&nested_data});
      assert(nested.Native().id != outer_id);
    }

    int main() {
      assert(FakeAdapter::live_states == 0);

      TestSequentialReuseAndRebind();
      TestNestedAcquireAndStableGrowth();
      TestLeaseMovesReleaseExactlyOneEntry();
      TestPoolVectorReallocationPreservesLiveEntries();

      assert(FakeAdapter::live_states == 0);
      assert(FakeAdapter::states_created == FakeAdapter::states_destroyed);

      return 0;
    }
    """
).lstrip()
