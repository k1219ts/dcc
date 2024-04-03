## Katana 3.5 Build Information
```
GCC 4.8.2
C++11
```

---
## USDSkelSkinning(not using) change to USDSkelBinding
```
BUILDDIR = ./build/Ops

# KATANA-USD 19.11 fn2
USD_ROOT = /backstage/libs/USD/katana-usd/3.5v2/19.11/third_party/katana
make clean && make -j32
```
#### INSTALL
```
cp ./build/Ops/USDSkelSkinning.so $USD_ROOT/plugin/Ops
```
## USDVolPrman
빌드방법은 USDSkelSkinning과 같다.


### CMAKE rez-env
```
cmake를 실행하기전에 필요한 라이브러리를 rez-env로 환경 셋팅해야 한다. 그리고 아래 리스트에
있는 라이브러리를 추가하면, 해당 라이브러리 패키지만 찾는다.
    - boost
    - OpenEXR
    - TBB
    - HDF5
    - Alembic
    - ZLIB (등록하지 않으면, 로컬에서 찾는다)

# Example   
DCC rez-env openexr boost

이렇게 하면, OpenEXR과 boost 라이브러리를 가져온다. 그리고 rez-env 환경에서 아래 cmake를
실행하면 된다.
```

### CMAKE Linux Example
```
# Create a directory in which the build artefacts will be created.
cd source_directory
mkdir build
cd build

# Configure the project
cmake ..                                

# Build and install the project
cmake --build .
cmake --build . --target install

# Build only a single plug-in
cmake --build . --target Prune
cmake --build . --target install

# Re-build all
rm CMakeCache.txt; cmake ..; cmake --build . --target install

# Re-build specific project
rm CMakeCache.txt; cmake ..; cmake --build . --target DxUsdFeather; cmake --build . --target install
```
