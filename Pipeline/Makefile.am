
include $(top_srcdir)/cuda.mk

lib_LTLIBRARIES = libhdpipeline.la

libhdpipeline_la_SOURCES = default_params.C error.C parse_command_line.C clean_filterbank_rfi.cu.cpp get_rms.cu.cpp matched_filter.cu.cpp remove_baseline.cu.cpp find_giants.cu.cpp label_candidate_clusters.cu.cpp merge_candidates.cu.cpp pipeline.cu.cpp measure_bandpass.cu.cpp median_filter.cu.cpp matched_filter.cu.cpp 

nobase_include_HEADERS = hd/median_filter.h hd/error.h hd/types.h

# To find the dedisp lib from bbarsdell
AM_CPPFLAGS = -I$(top_srcdir) \
	      -I$(top_srcdir)/Network \
	      -I$(top_srcdir)/Formats \
	      @DEDISP_CFLAGS@ @THRUST_CFLAGS@ @CUDA_CFLAGS@ $(BOOST_CPPFLAGS)

include $(top_srcdir)/config/Makefile.targets

