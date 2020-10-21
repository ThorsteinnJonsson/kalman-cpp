set(sources
    src/kalman_filter.cpp
    src/extended_kalman_filter.cpp
)

set(exe_sources
		src/main.cpp
		${sources}
)

set(headers
  src/kalman_filter.h
  src/extended_kalman_filter.h
)
