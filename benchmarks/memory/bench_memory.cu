/** @file   bench-memory.cu
 *  @author Yuanxing Duan, PKU
 *  @brief  Compare GPUMemoryArena with GPUMemory.
 */

 /*
 * Bench content:
 * On RTX3090, use 16GB at maximum, continusly malloc and free small blocks with random
 * block size from 1MB to 64MB, the blocks are aligned to 128B and distributed exponentially.
 *
 * Results: GPUMemoryArena is good at efficiency but bad at utilization rate.
 */

#include <tiny-cuda-nn/common_device.h>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/random.h>


#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

#include <_Time.h>

using namespace tcnn;
using precision_t = network_precision_t;

constexpr size_t max_mem_size = 1ll << 33;
constexpr int times = 20000;

struct Action
{
	size_t size;
	unsigned int id;
	bool isMalloc;
};

std::vector<Action> bench_GPUMemory()
{
	constexpr int min_log = 10;
	constexpr int max_log = 30;
	constexpr size_t min_size = 1ll << min_log;
	constexpr size_t max_size = 1ll << max_log;

	std::mt19937 mt(1337);
	std::uniform_int_distribution<int> rd_log(min_log, max_log - 1);
	auto get_random_size = [&rd_log, &mt]()
	{
		int lg = rd_log(mt);
		std::uniform_int_distribution<size_t> rd(1 << lg, 1 << (lg + 1));
		return next_multiple<size_t>(rd(mt), 128);
	};

	unsigned int current_id(0);
	size_t total_size(0);
	std::vector<Action> actions;
	std::unordered_map<unsigned int, GPUMemory<char>> blocks;


	Timer timer;
	timer.begin();
	while (actions.size() < times)
	{
		if (total_size < max_mem_size)
		{
			size_t sz = get_random_size();
			blocks[current_id] = GPUMemory<char>(sz);
			blocks[current_id].memset(0);
			actions.push_back(Action{ sz, current_id, true });
			total_size += blocks[current_id].get_bytes();
			current_id += 1;
		}
		else
		{
			actions.push_back(Action{ blocks.begin()->second.get_bytes(),blocks.begin()->first,false });
			total_size -= blocks.begin()->second.get_bytes();
			// blocks.begin()->second.free_memory();
			blocks.erase(blocks.begin());
		}
		if (actions.size() % 1000 == 0)
		{
			fmt::print("{}: {} B {} B utilization {:.2f}%\n",
				actions.size(),
				total_size,
				cuda_memory_info().used,
				100 * float(total_size) / cuda_memory_info().used
			);
		}
	}
	timer.end();
	timer.print(fmt::format("Free and malloc {} times:", actions.size()).c_str());
	return actions;
}

void bench_GPUMemoryArena(std::vector<Action>const& actions)
{
	std::unordered_map<unsigned int, GPUMemoryArena::Allocation> blocks;
	size_t total_size(0);

	Timer timer;
	timer.begin();

	unsigned int c0(0);
	for (Action const& action : actions)
	{
		if (action.isMalloc)
		{
			blocks[action.id] = allocate_workspace(nullptr, action.size);
			total_size += action.size;
		}
		else
		{
			//blocks[action.id].~Allocation();
			blocks.erase(action.id);
			total_size -= action.size;
		}
		c0 += 1;
		if (c0 % 1000 == 0)
		{
			fmt::print("{}: {} B {} B utilization {:.2f}% Occupied {} free {}\n",
				c0,
				total_size,
				// total_n_bytes_allocated(),
				cuda_memory_info().used,
				100 * float(total_size) / cuda_memory_info().used,
				global_gpu_memory_arenas()[cuda_device()]->block_num(),
				global_gpu_memory_arenas()[cuda_device()]->free_block_num());
		}
	}
	timer.end();
	timer.print(fmt::format("Free and malloc {} times:", actions.size()).c_str());
}

int main(int argc, char* argv[])
{
	try {
		uint32_t compute_capability = cuda_compute_capability();
		if (compute_capability < MIN_GPU_ARCH) {
			std::cerr
				<< "Warning: Insufficient compute capability " << compute_capability << " detected. "
				<< "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly." << std::endl;
		}

		// init
		GPUMemory<float> init(4096);
		init.free_memory();
		//bench
		bench_GPUMemoryArena(bench_GPUMemory());
	}
	catch (std::exception& e) {
		std::cout << "Uncaught exception: " << e.what() << std::endl;
	}

	return EXIT_SUCCESS;
}

