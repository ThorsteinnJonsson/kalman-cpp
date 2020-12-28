set(sources
  src/temp.cpp
)

set(exe_sources
		src/main.cpp
		${sources}
)

set(headers
  src/extended_kalman_filter.h
  src/filter_utils.h
  src/components/base_predictor.h
  src/components/base_updater.h
  src/internal/type_traits.h
)
