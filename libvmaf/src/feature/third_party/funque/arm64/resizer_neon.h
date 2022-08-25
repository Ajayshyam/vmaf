
#if OPTIMISED_COEFF
void step_neon(const unsigned char *_src, unsigned char *_dst,
               const short *_alpha, const short *_beta, 
               int iwidth, int iheight, int dwidth, int channels, 
               int ksize, int start, int end, int xmin, int xmax);
#else
void step_neon(const unsigned char *_src, unsigned char *_dst, 
               const int *xofs, const int *yofs, 
               const short *_alpha, const short *_beta, 
               int iwidth, int iheight, int dwidth, int dheight, int channels, 
               int ksize, int start, int end, int xmin, int xmax);
#endif
