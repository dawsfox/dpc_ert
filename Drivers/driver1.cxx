#include <CL/sycl.hpp>
#include "driver.h"
#include "kernel1.h"


namespace sycl = cl::sycl;

#ifdef ERT_DPCPP

//#define KERNEL1(a,b,c)   ((a) = (b) + (c))
//#define KERNEL2(a,b,c)   ((a) = (a)*(b) + (c))

double getTime()
{
  double time;
  struct timeval tm;
  gettimeofday(&tm, NULL);
  time = tm.tv_sec + (tm.tv_usec / 1000000.0);
  return time;
}

template <typename TAccessorRW, typename TAccessorW> class sycl_kernel {
public:
  sycl_kernel(const ulong nsize_, const ulong trials_, TAccessorRW accessor_buf_, TAccessorW accessor_params_)
    :  nsize(nsize_), trials(trials_), accessor_buf(accessor_buf_), accessor_params(accessor_params_) {}
  void operator()(sycl::nd_item<1> idx) {

    //auto A = accessor_buf;
    //auto params = accessor_params;
    for (ulong k=0; k<trials; k++) {
      size_t total_thr = idx.get_group_range(0) * idx.get_local_range(0);
      size_t elem_per_thr = (nsize + (total_thr-1)) / total_thr;
      size_t blockOffset = idx.get_group(0)  * idx.get_local_range(0) ;

      size_t start_idx  = blockOffset + idx.get_local_id(0) ;
      size_t end_idx    = start_idx + elem_per_thr * total_thr;
      size_t stride_idx = total_thr;

      if (start_idx > nsize) {
        start_idx = nsize;
      }

      if (end_idx > nsize) {
        end_idx = nsize;
      }
      
      double alpha = 2.0;
      double beta = 1.0;

      size_t i, j;
      for (i=start_idx; i<end_idx; i+=stride_idx) {
        beta = 1.0;
#if (ERT_FLOP & 1) == 1       
        KERNEL1(beta,accessor_buf[i],alpha);
#endif
#if (ERT_FLOP & 2) == 2       
        KERNEL2(beta,accessor_buf[i],alpha);
#endif
#if (ERT_FLOP & 4) == 4       
        REP2(KERNEL2(beta,accessor_buf[i],alpha));
#endif
#if (ERT_FLOP & 8) == 8       
        REP4(KERNEL2(beta,accessor_buf[i],alpha));
#endif
#if (ERT_FLOP & 16) == 16     
        REP8(KERNEL2(beta,accessor_buf[i],alpha));
#endif
#if (ERT_FLOP & 32) == 32     
        REP16(KERNEL2(beta,accessor_buf[i],alpha));
#endif
#if (ERT_FLOP & 64) == 64     
        REP32(KERNEL2(beta,accessor_buf[i],alpha));
#endif
#if (ERT_FLOP & 128) == 128   
        REP64(KERNEL2(beta,accessor_buf[i],alpha));
#endif
#if (ERT_FLOP & 256) == 256   
        REP128(KERNEL2(beta,accessor_buf[i],alpha));
#endif
#if (ERT_FLOP & 512) == 512   
        REP256(KERNEL2(beta,accessor_buf[i],alpha));
#endif
#if (ERT_FLOP & 1024) == 1024 
        REP512(KERNEL2(beta,accessor_buf[i],alpha));
#endif

        accessor_buf[i] = -beta;
        
      } //inner for
    } //outer for
    //accessor_params[0] = accessor_buf.get_count();
    accessor_params[0] = accessor_buf.get_size() / accessor_buf.get_count();
    accessor_params[1] = 2;
  
  } //operator
private:
  const ulong nsize;
  const ulong trials;
  TAccessorRW accessor_buf;
  TAccessorW accessor_params;
};

#endif

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

#ifndef ERT_DPCPP
template<typename Kernel>
inline void launchKernel(Kernel&& sycl_kernel, uint64_t n, uint64_t t, sycl::queue queue,
                         sycl::buffer<double, 1> d_buf, sycl::buffer<int, 1> d_params, sycl::event *event)
{
  if ((global_size != 0) && (local_size != 0))
  {
      const uint64_t const_n = n;
      const uint64_t const_t = t;
      *event = queue.submit([&] (sycl::handler& cgh) {
        auto d_buf_accessor = d_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto d_params_accessor = d_params.get_access<sycl::access::mode::read_write>(cgh);
        cgh.set_arg(0, const_n);
        cgh.set_arg(1, const_t);
        cgh.set_arg(2, d_buf_accessor);
        cgh.set_arg(3, d_params_accessor);
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
#else
inline void launchKernel(uint64_t n, uint64_t t, sycl::queue queue,
                         sycl::buffer<double, 1> d_buf, sycl::buffer<int, 1> d_params, sycl::event *event)
{
  if ((global_size != 0) && (local_size != 0))
  {
      const uint64_t const_n = n;
      const uint64_t const_t = t;
      *event = queue.submit([&] (sycl::handler& cgh) {
        auto d_buf_accessor = d_buf.get_access<sycl::access::mode::read_write>(cgh);
        auto d_params_accessor = d_params.get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>(global_size),
                                           sycl::range<1>(local_size)},
                         sycl_kernel(n, t, d_buf_accessor, d_params_accessor));
      });
  }
}
#endif




int main(int argc, char *argv[]) {
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
    std::vector<sycl::device> devices = platforms[0].get_devices();
    int num_devices = devices.size();
    sycl::device device = devices[id % num_devices];
    if (!device.is_gpu()) {
      for (const auto &dev: devices) {
        if (dev.is_gpu()) {
          device = dev;
          break;
        }
      }
    }
    sycl::event event;
    sycl::event mem_event;
    const sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
    /*
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
    */
    //sycl::queue queue(device, async_handler);
    //sycl::queue queue(device, async_handler, prop_list); //debugging profiling error
    sycl::queue queue(device, prop_list);
    sycl::context context = queue.get_context();
    size_t max_wg_size = device.get_info<sycl::info::device::max_work_group_size>();
    if (local_size > max_wg_size) {
      fprintf(stderr, "ERROR: Work group size > device maximum %ld\n", max_wg_size);
      exit(1);
    }
#ifndef ERT_DPCPP
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
#endif

  
    uint64_t nsize = PSIZE / nthreads;
    nsize          = nsize & (~(ERT_ALIGN - 1));
    nsize          = nsize / sizeof(double);
    uint64_t nid   = nsize * id;

    uint64_t n, nNew;
    uint64_t t;
    int bytes_per_elem;
    int mem_accesses_per_elem;

    n = ERT_WORKING_SET_MIN;
    int params[2];
    sycl::buffer<int, 1> d_params(params, 2); //gets filled by kernel
    sycl::buffer<double, 1> d_buf(nsize); //mirrors dblbuf which is filled by alloc function  //changed from nsize -> n
    while (n <= nsize) { // working set - nsize

      uint64_t ntrials = nsize / n;
      if (ntrials < ERT_TRIALS_MIN)
        ntrials = ERT_TRIALS_MIN;

      // initialize small chunk of buffer within each thread
      double value = -1.0;
      initialize<double>(nsize, &dblbuf[nid], value);
      sycl::id<1> access_point = nid;
      sycl::range<1> reach = n;
      sycl::buffer sub_d_buf(d_buf, access_point, reach); //create sub buffer, base index = nid, size from there is n based on OCL code
      queue.submit([&](sycl::handler &cgh) {
        auto sub_buf_accessor = sub_d_buf.get_access<sycl::access::mode::read_write>(cgh);
        cgh.copy(dblbuf, sub_buf_accessor);
      });
      
// #ifndef ERT_DPCPP
      for (t = 1; t <= ntrials; t = t * 2) { // working set - ntrials
        //double startTime, endTime; //for if profiling is unavailable
        //{
#ifndef ERT_DPCPP
          launchKernel(ocl_kernel, n, t, queue, d_buf, d_params, &event);
#else
          //startTime = getTime(); //for if profiling isn't available
          launchKernel(n, t, queue, d_buf, d_params, &event);
          //event.wait();
          mem_event = queue.submit([&](sycl::handler &cgh) {
            auto accessorA = d_params.get_access<sycl::access::mode::read_write>(cgh);
            cgh.copy(accessorA, params);
          });
          mem_event.wait();
         
          
#endif
        //}
        bytes_per_elem = params[0];
        mem_accesses_per_elem = params[1];
        if ((id == 0) && (rank == 0)) {
          uint64_t working_set_size = n * nthreads * nprocs;
          uint64_t total_bytes      = t * working_set_size * bytes_per_elem * mem_accesses_per_elem;
          uint64_t total_flops      = t * working_set_size * ERT_FLOP;
          double seconds;
          //endTime = getTime(); //if profiling is unavailable
          //seconds = endTime - startTime; //if profiling is unavailable
          event.wait();     
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
