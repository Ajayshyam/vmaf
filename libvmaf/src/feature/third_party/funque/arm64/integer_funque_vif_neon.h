

#if USE_DYNAMIC_SIGMA_NSQ
int integer_compute_vif_funque_neon(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, 
                                    double* score, double* score_num, double* score_den, int k, int stride, 
                                    double sigma_nsq_arg, int64_t shift_val, uint32_t* log_18, int vif_level);
#else
int integer_compute_vif_funque_neon(const dwt2_dtype* x_t, const dwt2_dtype* y_t, size_t width, size_t height, 
                                    double* score, double* score_num, double* score_den, int k, int stride, 
                                    double sigma_nsq_arg, int64_t shift_val, uint32_t* log_18);
#endif
