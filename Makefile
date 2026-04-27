run: all

all:
	cmake -B build
	make -C build -j 32

clean:
	cd build && make clean && cd ..

# bash "source ./.bashrc
upload:
	cmake -B build_test_2
	cmake --build build_test_2 --target fsi_lid_driven_sphere -j16
    # python3 features/test_tools/upload_to_notion.py

