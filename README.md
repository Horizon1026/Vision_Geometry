# Vision Geometry
Intergration of some vision geometry model algorithm.

# Components
- [x] Perspective n Points
- [x] Triangulation
    - [x] Analytic.
    - [x] Iterative.
    - [x] Line triangulation.
- [ ] Epipolar Constraint
    - [x] Essential
    - [ ] Fundamental
    - [ ] Homography
- [x] Relative Rotation
- [x] Iterative Closest Point.
    - [x] Point to Point (SVD method).
    - [x] Point to Line.
    - [x] Point to Plane.
- [x] Normal Distribution Transform.

# Dependence

### Project repositories
- Slam_Utility
- Visualizor2D（only for test）
- Visualizor3D（only for test）

### Third-party repositories
- Eigen3 (>= 3.3.7)（`sudo apt install libeigen3-dev`）
- glfw3（`sudo apt install libglfw3-dev`）（only for test）
- nanoflann（bundled in Slam_Utility/src/data_struct/kd_tree）

# Compile and Run
- 第三方仓库的话需要自行 apt-get install 安装
- 拉取 Dependence 中的源码，在当前 repo 中创建 build 文件夹，执行标准 cmake 过程即可
```bash
mkdir build
cmake ..
make -j
```
- 编译成功的可执行文件就在 build 中，具体有哪些可执行文件可参考 run.sh 中的列举。可以直接运行 run.sh 来依次执行所有可执行文件

```bash
sh run.sh
```

# Tips
- 欢迎一起交流学习，不同意商用；
- NanoFlann真好用，感谢开源的作者~
