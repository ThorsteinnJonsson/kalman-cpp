set(sources
  ""
)

set(exe_sources
		src/main.cpp
		${sources}
)

set(headers
  src/extended_kalman_filter.h
  src/util/filter_utils.h
  src/util/type_traits.h
  src/components/base_predictor.h
  src/components/base_updater.h
)
