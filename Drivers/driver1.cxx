#include <CL/sycl.hpp>
#include "driver.h"
#include "kernel1.h"

//#define ERT_KERNEL "kernel1.cl"
//just for now - command line defining is giving warnings

// Maybe in the futur sycl will note be in the 'cl' namespace
namespace sycl = cl::sycl;

inline std::string loadProgram(std::string input)
{
  std::ifstream stream(input.c_str());
  if (!stream.is_open()) {
    std::cout << "Cannot open file: " << input << std::endl;
    exit(1);
  }

  return std::string(std::istreambuf_iterator<char>(stream), (std::istreambuf_iterator<char>()));
}

template <typename T>
T *alloc(uint64_t psize)
{
#ifdef ERT_INTEL
  return (T *)_mm_malloc(psize, ERT_ALIGN);
#else
  return (T *)malloc(psize);
#endif
}

template <typename T>
inline void checkBuffer(T *buffer)
{
  if (buffer == nullptr) {
    fprintf(stderr, "Out of memory!\n");
    exit(1);
  }
}


//do I need to get a name for the kernel?
//should I pass in the kernel or the program? is it sycl (I think) or cl?
template<typename Kernel>
inline void launchKernel(Kernel&& sycl_kernel, uint64_t n, uint64_t t, sycl::queue queue,
                         sycl::buffer<double, 1> d_buf, sycl::buffer<int, 1> d_params, sycl::event *event)
{
  if ((global_size != 0) && (local_size != 0))
  {
    //*event = ocl_kernel(cl::EnqueueArgs(queue, cl::NDRange(global_size), cl::NDRange(local_size)), 
    //           n, t, d_buf, d_params);

      const uint64_t const_n = n;
      const uint64_t const_t = t;
      *event = queue.submit([&] (sycl::handler& cgh) {
        auto d_buf_accessor = d_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto d_params_accessor = d_params.get_access<sycl::access::mode::read_write>(cgh);
        cgh.set_arg(0, const_n);
        cgh.set_arg(1, const_t);
        cgh.set_arg(2, d_buf_accessor);
        cgh.set_arg(3, d_params_accessor);
        //cgh.set_args(const_n, const_t, d_buf_accessor, d_params_accessor);
        cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>(global_size),
                                           sycl::range<1>(local_size)},
                         sycl_kernel);
      });
   }
  /*
  else if ((global_size == 0) && (local_size != 0))
    *event = ocl_kernel(cl::EnqueueArgs(queue, cl::NDRange(n), cl::NDRange(local_size)), 
               n, t, d_buf, d_params);
  else
    *event = ocl_kernel(cl::EnqueueArgs(queue, cl::NDRange(n)), 
               n, t, d_buf, d_params);
  */
}




int main(int argc, char *argv[]) {
  // Selectors determine which device kernels will be dispatched to.
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  //sycl::default_selector selector;

  //grab arguments
  if (argc == 2) {                   // Usage: driver local_size
    global_size = 0;
    local_size = atoi(argv[1]);
  }
  else if ( argc == 3 ) {            // Usage: driver global_size local_size
    global_size = atoi(argv[1]);
    local_size  = atoi(argv[2]);
  }
  else {                             // No args, let the OpenCL runtime decide
    global_size = 0;
    local_size = 0;
  }


  int rank     = 0;
  int nprocs   = 1;
  int nthreads = 1;

  uint64_t TSIZE = ERT_MEMORY_MAX;
  uint64_t PSIZE = TSIZE / nprocs;


  double *__restrict__ dblbuf = alloc<double>(PSIZE);
  checkBuffer(dblbuf);
  //run<double>(PSIZE, dblbuf, rank, nprocs, &nthreads);

  if (rank == 0) {
    if (std::is_floating_point<double>::value) {
      if (sizeof(double) == 4) {
        printf("fp32\n");
      } else if (sizeof(double) == 8) {
        printf("fp64\n");
      }
    } else if (std::is_same<double, half2>::value) {
      printf("fp16\n");
    } else {
      fprintf(stderr, "Data type not supported.\n");
      exit(1);
    }
  }

  int id = 0;

  {
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    //printf("Platform: %s %s %s\n", platforms[0].get_info<sycl::info::platform::name>(),
    //                               platforms[0].get_info<sycl::info::platform::vendor>(),
    //                               platforms[0].get_info<sycl::info::platform::version>());
    std::vector<sycl::device> devices = platforms[0].get_devices();
    //check for if device is gpu
    int num_devices = devices.size();
    sycl::device device = devices[id % num_devices];
    sycl::event event;
    const sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    //sycl::queue queue(device, prop_list); //enables profiling for command groups on this queue
    auto  async_handler = [] (sycl::exception_list exceptions) {
      for (std::exception_ptr const &e : exceptions) {
        try {
          std::rethrow_exception(e);
        }
        catch (sycl::exception const &e) {
          printf("Async Exception: %s\n", e.what());
          std::terminate();
        }
      }
    };
    sycl::queue queue(device, async_handler, prop_list); //debugging profiling error
    sycl::context context = queue.get_context();
    size_t max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    if (local_size > max_wg_size) {
      fprintf(stderr, "ERROR: Work group size > device maximum %ld\n", max_wg_size);
      exit(1);
    }
    //building cl program
    char build_args[80];
    sprintf(build_args, "Kernels/%s", ERT_KERNEL);
    sycl::program program(context);
    char build_params[80];
    #define PRECISION "FP64"
    sprintf(build_params, "-cl-std=CL2.0 -DERT_FLOP=%d -D%s", ERT_FLOP, PRECISION);
    program.build_with_source(loadProgram(build_args), build_params); //select 2.0, I think that's what was used before, maybe 2.2
    auto ocl_kernel = program.get_kernel("ocl_kernel");
    //nthreads   = *nthreads_ptr; //in original, declared as 1 in main, declared to this in run function

  
    uint64_t nsize = PSIZE / nthreads;
    nsize          = nsize & (~(ERT_ALIGN - 1));
    nsize          = nsize / sizeof(double);
    uint64_t nid   = nsize * id;

    uint64_t n, nNew;
    uint64_t t;
    int bytes_per_elem;
    int mem_accesses_per_elem;

    n = ERT_WORKING_SET_MIN;
    while (n <= nsize) { // working set - nsize

      uint64_t ntrials = nsize / n;
      if (ntrials < ERT_TRIALS_MIN)
        ntrials = ERT_TRIALS_MIN;

      // initialize small chunk of buffer within each thread
      double value = -1.0;
      initialize<double>(nsize, &dblbuf[nid], value);
      for (t = 1; t <= ntrials; t = t * 2) { // working set - ntrials
        int params[2];
        {
          //buffers have to be in scope of the for loop or they won't update properly
          sycl::buffer<int, 1> d_params(params, 2); //gets filled by kernel?
          sycl::buffer<double, 1> d_buf(dblbuf, nsize); //mirrors dblbuf which is filled by alloc function 
          launchKernel(ocl_kernel, n, t, queue, d_buf, d_params, &event); //this most definitely needs to be tweaked
          event.wait();
        }
        bytes_per_elem = params[0];
        mem_accesses_per_elem = params[1];
        if ((id == 0) && (rank == 0)) {
          uint64_t working_set_size = n * nthreads * nprocs;
          uint64_t total_bytes      = t * working_set_size * bytes_per_elem * mem_accesses_per_elem;
          uint64_t total_flops      = t * working_set_size * ERT_FLOP;
          double seconds;
          seconds = (double)(event.get_profiling_info<sycl::info::event_profiling::command_end>() 
                       - event.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9;
          printf("%12" PRIu64 " %12" PRIu64 " %15.3lf %12" PRIu64 " %12" PRIu64 "\n", working_set_size * bytes_per_elem,
                 t, seconds * 1000000, total_bytes, total_flops);
        } //if -- print
      } //working set - ntrials

      nNew = ERT_WSS_MULT * n;
      if (nNew == n) {
        nNew = n + 1;
      }

      n = nNew;
      
    } //working set - nsize

  } //"parallel scope"
  free(dblbuf);
  if (rank == 0) {
    printf("\n");
    printf("META_DATA\n");
    printf("FLOPS          %d\n", ERT_FLOP);
    printf("GLOBAL_SIZE    %ld\n", global_size);
    printf("LOCAL_SIZE     %ld\n", local_size);
  }
  return 0;
} //main
