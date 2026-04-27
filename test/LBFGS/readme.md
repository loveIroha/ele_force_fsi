### [LBFGS](https://github.com/yixuan/LBFGSpp)



最好提供三种接口：
```
double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad);
double operator()(const std::vector<double>& x, std::vector<double>& grad);
double operator()(const std::vector<double3>& x, std::vector<double3>& grad);
```