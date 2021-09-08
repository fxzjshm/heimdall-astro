
include $(top_srcdir)/cuda.mk

lib_LTLIBRARIES = libhdpipeline.la

libhdpipeline_la_SOURCES = default_params.C error.C parse_command_line.C clean_filterbank_rfi.dp.cpp get_rms.dp.cpp matched_filter.dp.cpp remove_baseline.dp.cpp find_giants.dp.cpp label_candidate_clusters.dp.cpp merge_candidates.dp.cpp pipeline.dp.cpp measure_bandpass.dp.cpp median_filter.dp.cpp matched_filter.dp.cpp 

nobase_include_HEADERS = hd/median_filter.h hd/error.h hd/types.h

# To find the dedisp lib from bbarsdell
AM_CPPFLAGS = -I/opt/boost.compute/include \
          -I$(top_srcdir) \
	      -I$(top_srcdir)/Network \
	      -I$(top_srcdir)/Formats \
	      @DEDISP_CFLAGS@ \
          -DBOOST_COMPUTE_DEBUG_KERNEL_COMPILATION \
          -DBOOST_COMPUTE_THREAD_SAFE \
          -DBOOST_COMPUTE_HAVE_THREAD_LOCAL \
          -DBOOST_COMPUTE_USE_OFFLINE_CACHE \
          -std=c++17 -fPIE
AM_LDFLAGS = -lOpenCL -pthread -lboost_system -lboost_filesystem

include $(top_srcdir)/config/Makefile.targets

