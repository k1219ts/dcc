## Katana 3.2 Build Information
```
GCC 4.8.2
C++11
```

---
## USDSkelSkinning
```
BUILDDIR = ./build/Ops

# USD 19.07
USD_ROOT = /backstage/libs/USD/pixar-usd/19.07
make

# USD 19.11
USD_ROOT = /backstage/libs/USD/pixar-usd/19.11
scl enable devtoolset-6 "make -j32"
```
#### INSTALL
```
cp ./build/Ops/USDSkelSkinning.so $USD_ROOT/katana3.2/third_party/katana/plugin/Ops
```
## USDVolPrman
빌드방법은 USDSkelSkinning과 같다.
