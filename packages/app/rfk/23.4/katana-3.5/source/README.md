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
