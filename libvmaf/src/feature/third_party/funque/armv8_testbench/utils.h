#ifndef _UTILS_H_
#define _UTILS_H_

/* System include files */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/time.h>

#ifndef PROFILE_ENABLE
#define PROFILE_ENABLE
#endif

#define PROFILE_START get_curr_time(&s_tv);
#define PROFILE_END get_avg_elapsed_time(&s_tv, multiplier);

void get_curr_time(struct timeval *tv);
double get_avg_elapsed_time(struct timeval *s_tv, int multiplier);

#endif /* _UTILS_H_ */
