include $(top_srcdir)/cuda.mk

include_HEADERS = 

bin_PROGRAMS = heimdall coincidencer coincidencer_client candidate_profiler fil2pgm generate_dmlist

AM_CXXFLAGS = \
  -I$(top_srcdir) \
	-I$(top_srcdir)/Formats \
	-I$(top_srcdir)/Network \
	-I$(top_srcdir)/Pipeline \
	@PSRDADA_CFLAGS@ \
	@DEDISP_CFLAGS@

heimdall_SOURCES = heimdall.C
coincidencer_SOURCES = coincidencer.C Candidates.C
coincidencer_client_SOURCES = coincidencer_client.C

LDADD = \
  $(top_builddir)/Formats/libhdformats.la \
  $(top_builddir)/Pipeline/libhdpipeline.la \
  $(top_builddir)/Network/libhdnetwork.la \
  @DEDISP_LIBS@ @PSRDADA_LIBS@

AM_LDFLAGS = -lOpenCL -pthread

heimdall_CFLAGS   =
generate_dmlist_CXXFLAGS = @DEDISP_CFLAGS@


include $(top_srcdir)/config/Makefile.targets

