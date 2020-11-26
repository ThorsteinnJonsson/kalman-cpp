set(sources
  src/temp.cpp
)

set(exe_sources
		src/main.cpp
		${sources}
)

set(headers
  src/kalman_filter.h
  src/extended_kalman_filter.h
  src/filter_utils.h
  src/prediction/base_predictor.h
)
