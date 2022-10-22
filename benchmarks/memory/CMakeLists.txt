add_executable(bench_memory bench_memory.cu)
target_include_directories(bench_memory PUBLIC
	"../../dependencies/tiny-stl/include"
)
target_link_libraries(bench_memory PUBLIC ${CUDA_LIBRARIES} tiny-cuda-nn cublas)