#include "utils.h"

void get_curr_time(struct timeval *tv)
{
    gettimeofday(tv, NULL);
}

double get_avg_elapsed_time(struct timeval *s_tv, int multiplier)
{
    double time_useconds;
    struct timeval e_tv;

    get_curr_time(&e_tv);
    time_useconds = ((e_tv.tv_sec - s_tv->tv_sec) * 1000000) +
                    (e_tv.tv_usec - s_tv->tv_usec);
	time_useconds /=  multiplier;
    return time_useconds;
}
