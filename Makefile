all:
	meson setup libvmaf/build libvmaf --buildtype debug -Denable_float=true -Denable_float_funque=true && \
	ninja -vC libvmaf/build
	cd python && python3 setup.py build_ext --build-lib .

clean:
	rm -rf libvmaf/build
	rm -f python/vmaf/core/adm_dwt2_cy.c*

install:
	meson setup libvmaf/build libvmaf --buildtype debug && \
	ninja -vC libvmaf/build install
