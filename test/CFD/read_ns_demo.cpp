/// @date 2023-09-28
/// @file NavierStokesDemo.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 从Json文件读入三维NS方程解析解，并生成可计算对象。
///
///

#include <AlgebraSolver/MultiArray.h>
#include <PhysicsSolver/StokesFlow3D/Pressure3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityU3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityV3D.h>
#include <PhysicsSolver/StokesFlow3D/VelocityW3D.h>
#include <fmt/core.h>
#include <muParser.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <variant>

using MultiArrayInt    = typename algebra::MultiArray<3, int>::type;
using MultiArrayDouble = typename algebra::MultiArray<3, double>::type;

class JsonFile {
  public:
    using json = nlohmann::json;
    JsonFile(const std::string& filename) {
        // 从文件中读取JSON数据
        std::ifstream file(filename);
        if (file.is_open()) {
            try {
                file >> jsonData; // 将文件中的JSON数据解析为 nlohmann::json 对象
            } catch (const std::exception& e) { std::cerr << "Error parsing JSON: " << e.what() << std::endl; }
            file.close();
        } else {
            std::cerr << "Error opening file: " << filename << std::endl;
        }
    }

    // void printData() {
    //     // 输出JSON数据
    //     std::cout << "JSON Data:" << std::endl;
    //     std::cout << jsonData.dump(4) << std::endl; //
    //     使用四个空格缩进进行漂亮的打印
    // }

    // 重载 [] 运算符，使得可以通过键获取值。
    // TODO: 此函数亟待优化！
    std::string operator[](const std::string& key) {
        // 根据键获取对应的表达式
        if (jsonData.find(key) != jsonData.end()) {
            CHECK_F(jsonData[key].is_string(), "The value of key %s is not string.", key.c_str());
            auto result = jsonData[key].get<std::string>();
            return result;
        } else {
            // 未找到键，返回空的 JSON 对象，因为函数的返回值类型为
            // std::string，返回一个空的 JSON 对象会报错
            std::cerr << "Key not found: " << key << std::endl;
            return json();
        }
    }

  private:
    json jsonData;
};

class ExpressionEvaluator {
  public:
    double        _x;
    double        _y;
    double        _z;
    double        _t;
    static double pow_double(double __x, double __y) { return std::pow(__x, __y); }

    static double exp_double(double __x) { return std::exp(__x); }
    ExpressionEvaluator(const std::string& expression) {
        parser = std::make_unique<mu::Parser>();
        parser->SetExpr(expression);

        parser->DefineVar("x", &_x);
        parser->DefineVar("y", &_y);
        parser->DefineVar("z", &_z);
        parser->DefineVar("t", &_t);

        parser->DefineFun("pow", pow_double);
        parser->DefineFun("exp", exp_double);
    }
    double operator()(double x, double y, double z, double t) {
        try {
            _x = x;
            _y = y;
            _z = z;
            _t = t;
            return parser->Eval();
        } catch (mu::Parser::exception_type& e) {
            std::cerr << "Error: " << e.GetMsg() << std::endl;
            return 0.0;
        }
    }

  private:
    std::unique_ptr<mu::Parser> parser;
};

class NavierStokesDemo {
  private:
    ExpressionEvaluator _u, _v, _w, _p;
    ExpressionEvaluator _dudx, _dudy, _dudz;
    ExpressionEvaluator _dvdx, _dvdy, _dvdz;
    ExpressionEvaluator _dwdx, _dwdy, _dwdz;
    ExpressionEvaluator _dpdx, _dpdy, _dpdz;
    ExpressionEvaluator _f1, _f2, _f3, _bp;

    int    Nx, Ny, Nz, Nt;
    double width, height, depth, dt, dx, dy, dz, T, rho, mu_f;
    int    DIRICHLET = 1;
    int    NEUMANN   = 2;

  public:
    NavierStokesDemo(JsonFile json_file)
        : _u(json_file["u"]), _v(json_file["v"]), _w(json_file["w"]), _p(json_file["p"]), _dudx(json_file["dudx"]),
          _dudy(json_file["dudy"]), _dudz(json_file["dudz"]), _dvdx(json_file["dvdx"]), _dvdy(json_file["dvdy"]),
          _dvdz(json_file["dvdz"]), _dwdx(json_file["dwdx"]), _dwdy(json_file["dwdy"]), _dwdz(json_file["dwdz"]),
          _dpdx(json_file["dpdx"]), _dpdy(json_file["dpdy"]), _dpdz(json_file["dpdz"]), _f1(json_file["f1"]),
          _f2(json_file["f2"]), _f3(json_file["f3"]), _bp(json_file["bp"]) {}
    void set(int Nx, int Ny, int Nz, int Nt, double width, double height, double depth, double T, double rho,
             double mu_f) {
        this->Nx     = Nx;
        this->Ny     = Ny;
        this->Nz     = Nz;
        this->Nt     = Nt;
        this->width  = width;
        this->height = height;
        this->depth  = depth;
        this->T      = T;
        this->rho    = rho;
        this->mu_f   = mu_f;
        dx           = width / Nx;
        dy           = height / Ny;
        dz           = depth / Nz;
        dt           = T / Nt;
    }

    MultiArrayDouble get_array_v(double t) {
        MultiArrayDouble data = algebra::create_multi_array<3, double>({Nx + 2, Ny + 1, Nz + 2});
        for (int i = -1; i <= Nx; ++i) {
            for (int j = 0; j <= Ny; ++j) {
                for (int k = -1; k <= Nz; ++k) {
                    data[i + 1][j][k + 1] = _v((i + 0.5) * dx, j * dy, (k + 0.5) * dz, t);
                }
            }
        }

        return data;
    }

    MultiArrayDouble get_array_u(double t) {
        auto data = algebra::create_multi_array<3, double>({Nx + 1, Ny + 2, Nz + 2});

        for (int i = 0; i < Nx + 1; ++i) {
            for (int j = -1; j < Ny + 1; ++j) {
                for (int k = -1; k < Nz + 1; ++k) {
                    data[i][j + 1][k + 1] = _u(i * dx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }

        return data;
    }

    MultiArrayDouble get_array_w(double t) {
        auto data = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 1});

        for (int i = -1; i < Nx + 1; ++i) {
            for (int j = -1; j < Ny + 1; ++j) {
                for (int k = 0; k < Nz + 1; ++k) {
                    data[i + 1][j + 1][k] = _w((i + 0.5) * dx, (j + 0.5) * dy, k * dz, t);
                }
            }
        }

        return data;
    }

    MultiArrayDouble get_array_f1(double t) {
        auto data = algebra::create_multi_array<3, double>({Nx + 1, Ny + 2, Nz + 2});
        for (int i = 0; i < Nx + 1; ++i) {
            for (int j = -1; j < Ny + 1; ++j) {
                for (int k = -1; k < Nz + 1; ++k) {
                    data[i][j + 1][k + 1] = _f1(i * dx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }

        return data;
    }

    MultiArrayDouble get_array_f2(double t) {
        MultiArrayDouble data = algebra::create_multi_array<3, double>({Nx + 2, Ny + 1, Nz + 2});
        for (int i = -1; i <= Nx; ++i) {
            for (int j = 0; j <= Ny; ++j) {
                for (int k = -1; k <= Nz; ++k) {
                    data[i + 1][j][k + 1] = _f2((i + 0.5) * dx, j * dy, (k + 0.5) * dz, t);
                }
            }
        }

        return data;
    }

    MultiArrayDouble get_array_f3(double t) {
        MultiArrayDouble data = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 1});

        for (int i = -1; i < Nx + 1; ++i) {
            for (int j = -1; j < Ny + 1; ++j) {
                for (int k = 0; k < Nz + 1; ++k) {
                    data[i + 1][j + 1][k] = _f3((i + 0.5) * dx, (j + 0.5) * dy, k * dz, t);
                }
            }
        }

        return data;
    }

    MultiArrayInt get_boundary_type_u(const std::array<int, 6>& all_boundary_type) {
        auto boundary_types = algebra::create_multi_array<3, int>({Nx + 1, Ny + 2, Nz + 2});
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                boundary_types[0][j + 1][k + 1]  = all_boundary_type[0];
                boundary_types[Nx][j + 1][k + 1] = all_boundary_type[1];
            }
        }
        for (int i = 0; i <= Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                boundary_types[i][0][k + 1]      = all_boundary_type[2];
                boundary_types[i][Ny + 1][k + 1] = all_boundary_type[3];
            }
        }
        for (int i = 0; i <= Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                boundary_types[i][j + 1][0]      = all_boundary_type[4];
                boundary_types[i][j + 1][Nz + 1] = all_boundary_type[5];
            }
        }
        return boundary_types;
    }

    MultiArrayInt get_boundary_type_v(const std::array<int, 6>& all_boundary_type) {
        // std::vector<std::vector<std::vector<int>>> getBoundaryTypeV(int Nx, int
        // Ny, int Nz, std::vector<int> all_boundary_type) {
        MultiArrayInt boundary_types = algebra::create_multi_array<3, int>({Nx + 2, Ny + 1, Nz + 2});

        for (int j = 0; j <= Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                boundary_types[0][j][k + 1]      = all_boundary_type[0];
                boundary_types[Nx + 1][j][k + 1] = all_boundary_type[1];
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                boundary_types[i + 1][0][k + 1]  = all_boundary_type[2];
                boundary_types[i + 1][Ny][k + 1] = all_boundary_type[3];
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j <= Ny; ++j) {
                boundary_types[i + 1][j][0]      = all_boundary_type[4];
                boundary_types[i + 1][j][Nz + 1] = all_boundary_type[5];
            }
        }

        return boundary_types;
    }

    MultiArrayInt get_boundary_type_w(const std::array<int, 6>& all_boundary_type) {
        MultiArrayInt boundary_types(Nx + 2, std::vector<std::vector<int>>(Ny + 2, std::vector<int>(Nz + 1)));

        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k <= Nz; ++k) {
                boundary_types[0][j + 1][k]      = all_boundary_type[0];
                boundary_types[Nx + 1][j + 1][k] = all_boundary_type[1];
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k <= Nz; ++k) {
                boundary_types[i + 1][0][k]      = all_boundary_type[2];
                boundary_types[i + 1][Ny + 1][k] = all_boundary_type[3];
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                boundary_types[i + 1][j + 1][0]  = all_boundary_type[4];
                boundary_types[i + 1][j + 1][Nz] = all_boundary_type[5];
            }
        }

        return boundary_types;
    }

    auto make_tentitive_ub(const MultiArrayDouble& un, const MultiArrayDouble& f) {
        MultiArrayDouble ub = algebra::create_multi_array<3, double>({Nx + 1, Ny + 2, Nz + 2});
        for (int i = 0; i <= Nx; ++i) {
            for (int j = 0; j < Ny + 2; ++j) {
                for (int k = 0; k < Nz + 2; ++k) {
                    ub[i][j][k] = rho * un[i][j][k] / dt + f[i][j][k];
                }
            }
        }
        return ub;
    }

    auto make_tentitive_vb(const MultiArrayDouble& vn, const MultiArrayDouble& f) {
        MultiArrayDouble vb = algebra::create_multi_array<3, double>({Nx + 2, Ny + 1, Nz + 2});

        for (int i = 0; i < Nx + 2; ++i) {
            for (int j = 0; j < Ny + 1; ++j) {
                for (int k = 0; k < Nz + 2; ++k) {
                    vb[i][j][k] = rho * vn[i][j][k] / dt + f[i][j][k];
                }
            }
        }

        return vb;
    }

    auto make_tentitive_wb(const MultiArrayDouble& wn, const MultiArrayDouble& f) {
        std::vector<std::vector<std::vector<double>>> wb(
            Nx + 2, std::vector<std::vector<double>>(Ny + 2, std::vector<double>(Nz + 1)));

        for (int i = 0; i < Nx + 2; ++i) {
            for (int j = 0; j < Ny + 2; ++j) {
                for (int k = 0; k <= Nz; ++k) {
                    wb[i][j][k] = rho * wn[i][j][k] / dt + f[i][j][k];
                }
            }
        }

        return wb;
    }

    MultiArrayDouble get_boundary_values_u(const MultiArrayInt& boundary_type, double t) {
        // MultiArrayDouble data(Nx + 1, std::vector<std::vector<double>>(Ny + 2,
        // std::vector<double>(Nz + 2, 0.0)));
        auto data = algebra::create_multi_array<3, double>({Nx + 1, Ny + 2, Nz + 2});

        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[0][j + 1][k + 1] == DIRICHLET) {
                    data[0][j + 1][k + 1] = _u(0.0, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[0][j + 1][k + 1] == NEUMANN) {
                    data[0][j + 1][k + 1] = _dudx(0.0, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx][j + 1][k + 1] == DIRICHLET) {
                    data[Nx][j + 1][k + 1] = _u(width, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx][j + 1][k + 1] == NEUMANN) {
                    data[Nx][j + 1][k + 1] = _dudx(width, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }

        for (int i = 0; i <= Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i][0][k + 1] == DIRICHLET) { data[i][0][k + 1] = _u(i * dx, 0.0, (k + 0.5) * dz, t); }
                if (boundary_type[i][0][k + 1] == NEUMANN) {
                    data[i][0][k + 1] = _dudy(i * dx, 0.0, (k + 0.5) * dz, t);
                }
                if (boundary_type[i][Ny + 1][k + 1] == DIRICHLET) {
                    data[i][Ny + 1][k + 1] = _u(i * dx, height, (k + 0.5) * dz, t);
                }
                if (boundary_type[i][Ny + 1][k + 1] == NEUMANN) {
                    data[i][Ny + 1][k + 1] = _dudy(i * dx, height, (k + 0.5) * dz, t);
                }
            }
        }

        for (int i = 0; i <= Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                if (boundary_type[i][j + 1][0] == DIRICHLET) { data[i][j + 1][0] = _u(i * dx, (j + 0.5) * dy, 0, t); }
                if (boundary_type[i][j + 1][0] == NEUMANN) { data[i][j + 1][0] = _dudz(i * dx, (j + 0.5) * dy, 0, t); }
                if (boundary_type[i][j + 1][Nz + 1] == DIRICHLET) {
                    data[i][j + 1][Nz + 1] = _u(i * dx, (j + 0.5) * dy, depth, t);
                }
                if (boundary_type[i][j + 1][Nz + 1] == NEUMANN) {
                    data[i][j + 1][Nz + 1] = _dudz(i * dx, (j + 0.5) * dy, depth, t);
                }
            }
        }

        return data;
    }

    MultiArrayDouble get_boundary_values_v(const MultiArrayInt& boundary_type, double t) {
        auto data = algebra::create_multi_array<3, double>({Nx + 2, Ny + 1, Nz + 2});

        for (int j = 0; j <= Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[0][j][k + 1] == DIRICHLET) { data[0][j][k + 1] = _v(0.0, j * dy, (k + 0.5) * dz, t); }
                if (boundary_type[0][j][k + 1] == NEUMANN) {
                    data[0][j][k + 1] = _dvdx(0.0, j * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx + 1][j][k + 1] == DIRICHLET) {
                    data[Nx + 1][j][k + 1] = _v(width, j * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx + 1][j][k + 1] == NEUMANN) {
                    data[Nx + 1][j][k + 1] = _dvdx(width, j * dy, (k + 0.5) * dz, t);
                }
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i + 1][0][k + 1] == DIRICHLET) {
                    data[i + 1][0][k + 1] = _v((i + 0.5) * dx, 0.0, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][0][k + 1] == NEUMANN) {
                    data[i + 1][0][k + 1] = _dvdy((i + 0.5) * dx, 0.0, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][Ny][k + 1] == DIRICHLET) {
                    data[i + 1][Ny][k + 1] = _v((i + 0.5) * dx, height, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][Ny][k + 1] == NEUMANN) {
                    data[i + 1][Ny][k + 1] = _dvdy((i + 0.5) * dx, height, (k + 0.5) * dz, t);
                }
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j <= Ny; ++j) {
                if (boundary_type[i + 1][j][0] == DIRICHLET) { data[i + 1][j][0] = _v((i + 0.5) * dx, j * dy, 0, t); }
                if (boundary_type[i + 1][j][0] == NEUMANN) { data[i + 1][j][0] = _dvdz((i + 0.5) * dx, j * dy, 0, t); }
                if (boundary_type[i + 1][j][Nz + 1] == DIRICHLET) {
                    data[i + 1][j][Nz + 1] = _v((i + 0.5) * dx, j * dy, depth, t);
                }
                if (boundary_type[i + 1][j][Nz + 1] == NEUMANN) {
                    data[i + 1][j][Nz + 1] = _dvdz((i + 0.5) * dx, j * dy, depth, t);
                }
            }
        }

        return data;
    }

    MultiArrayDouble get_boundary_values_w(const MultiArrayInt& boundary_type, double t) {
        auto data = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 1});

        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k <= Nz; ++k) {
                if (boundary_type[0][j + 1][k] == DIRICHLET) { data[0][j + 1][k] = _w(0.0, (j + 0.5) * dy, k * dz, t); }
                if (boundary_type[0][j + 1][k] == NEUMANN) {
                    data[0][j + 1][k] = _dwdx(0.0, (j + 0.5) * dy, k * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k] == DIRICHLET) {
                    data[Nx + 1][j + 1][k] = _w(width, (j + 0.5) * dy, k * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k] == NEUMANN) {
                    data[Nx + 1][j + 1][k] = _dwdx(width, (j + 0.5) * dy, k * dz, t);
                }
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k <= Nz; ++k) {
                if (boundary_type[i + 1][0][k] == DIRICHLET) { data[i + 1][0][k] = _w((i + 0.5) * dx, 0.0, k * dz, t); }
                if (boundary_type[i + 1][0][k] == NEUMANN) {
                    data[i + 1][0][k] = _dwdy((i + 0.5) * dx, 0.0, k * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k] == DIRICHLET) {
                    data[i + 1][Ny + 1][k] = _w((i + 0.5) * dx, height, k * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k] == NEUMANN) {
                    data[i + 1][Ny + 1][k] = _dwdy((i + 0.5) * dx, height, k * dz, t);
                }
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                if (boundary_type[i + 1][j + 1][0] == DIRICHLET) {
                    data[i + 1][j + 1][0] = _w((i + 0.5) * dx, (j + 0.5) * dy, 0, t);
                }
                if (boundary_type[i + 1][j + 1][0] == NEUMANN) {
                    data[i + 1][j + 1][0] = _dwdz((i + 0.5) * dx, (j + 0.5) * dy, 0, t);
                }
                if (boundary_type[i + 1][j + 1][Nz] == DIRICHLET) {
                    data[i + 1][j + 1][Nz] = _w((i + 0.5) * dx, (j + 0.5) * dy, depth, t);
                }
                if (boundary_type[i + 1][j + 1][Nz] == NEUMANN) {
                    data[i + 1][j + 1][Nz] = _dwdz((i + 0.5) * dx, (j + 0.5) * dy, depth, t);
                }
            }
        }

        return data;
    }

    MultiArrayDouble get_array_bp(double t) {
        auto data = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    data[i + 1][j + 1][k + 1] = _bp((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }

        return data;
    }

    MultiArrayDouble get_array_p(double t) {
        auto data = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});
        for (int i = -1; i < Nx + 1; ++i) {
            for (int j = -1; j < Ny + 1; ++j) {
                for (int k = -1; k < Nz + 1; ++k) {
                    data[i + 1][j + 1][k + 1] = _p((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }

        return data;
    }

    MultiArrayDouble get_boundary_values_p(MultiArrayInt boundary_type, double t) {
        auto data = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});

        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[0][j + 1][k + 1] == DIRICHLET) {
                    data[0][j + 1][k + 1] = _p(0.0, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[0][j + 1][k + 1] == NEUMANN) {
                    data[0][j + 1][k + 1] = _dpdx(0.0, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) {
                    data[Nx + 1][j + 1][k + 1] = _p(width, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) {
                    data[Nx + 1][j + 1][k + 1] = _dpdx(width, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i + 1][0][k + 1] == DIRICHLET) {
                    data[i + 1][0][k + 1] = _p((i + 0.5) * dx, 0.0, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][0][k + 1] == NEUMANN) {
                    data[i + 1][0][k + 1] = _dpdy((i + 0.5) * dx, 0.0, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) {
                    data[i + 1][Ny + 1][k + 1] = _p((i + 0.5) * dx, height, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) {
                    data[i + 1][Ny + 1][k + 1] = _dpdy((i + 0.5) * dx, height, (k + 0.5) * dz, t);
                }
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                if (boundary_type[i + 1][j + 1][0] == DIRICHLET) {
                    data[i + 1][j + 1][0] = _p((i + 0.5) * dx, (j + 0.5) * dy, 0, t);
                }
                if (boundary_type[i + 1][j + 1][0] == NEUMANN) {
                    data[i + 1][j + 1][0] = _dpdz((i + 0.5) * dx, (j + 0.5) * dy, 0, t);
                }
                if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) {
                    data[i + 1][j + 1][Nz + 1] = _p((i + 0.5) * dx, (j + 0.5) * dy, depth, t);
                }
                if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) {
                    data[i + 1][j + 1][Nz + 1] = _dpdz((i + 0.5) * dx, (j + 0.5) * dy, depth, t);
                }
            }
        }

        return data;
    }

    MultiArrayInt get_boundary_type_p(const std::array<int, 6>& all_boundary_type) {
        auto boundary_types = algebra::create_multi_array<3, int>({Nx + 2, Ny + 2, Nz + 2});

        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                boundary_types[0][j + 1][k + 1]      = all_boundary_type[0];
                boundary_types[Nx + 1][j + 1][k + 1] = all_boundary_type[1];
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                boundary_types[i + 1][0][k + 1]      = all_boundary_type[2];
                boundary_types[i + 1][Ny + 1][k + 1] = all_boundary_type[3];
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                boundary_types[i + 1][j + 1][0]      = all_boundary_type[4];
                boundary_types[i + 1][j + 1][Nz + 1] = all_boundary_type[5];
            }
        }

        return boundary_types;
    }

    ~NavierStokesDemo() {}
};

int max_iters = 200000;
// std::array<int, 6> all_boundary_type = {2, 2, 2, 2, 2, 2};
std::array<int, 6> all_boundary_type = {1, 1, 1, 1, 1, 1};
double             tolerance         = 1e-6;

auto create_ns_demo(int Nx, int Ny, int Nz, int Nt, double width, double height, double depth, double T, double rho,
                    double mu_f) {
    JsonFile json_file("/home/kokkos/npuheart/features/finite_difference/3D/stokes_demo.json");
    auto     ns_demo = std::make_shared<NavierStokesDemo>(json_file);
    ns_demo->set(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);
    return ns_demo;
}

void solve_p(MultiArrayDouble& ph, MultiArrayDouble& r, const MultiArrayInt& pbt, const MultiArrayDouble& pbv,
             const MultiArrayDouble& bp, int Nx, int Ny, int Nz, double width, double height, double depth,
             int max_iters, double tolerance) {
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;

    auto p_prime = pressure_3D::generate_p_prime(pbt, pbv, Nx, Ny, Nz, width, height, depth);
    auto r_prime = pressure_3D::generate_r_prime(p_prime, bp, Nx, Ny, Nz, width, height, depth);

    std::ofstream out_p_prime("p_prime.txt");
    std::ofstream out_r_prime("r_prime.txt");
    algebra::print_multi_array<3, double>(p_prime, out_p_prime);
    algebra::print_multi_array<3, double>(r_prime, out_r_prime);

    for (int iter = 0; iter < max_iters; ++iter) {
        pressure_3D::smooth(ph, r_prime, pbt, Nx, Ny, Nz, width, height, depth);
        pressure_3D::residual(r, ph, r_prime, pbt, Nx, Ny, Nz, width, height, depth);

        // Check convergence condition
        double sum_r_squared = algebra::norm(r) * std::sqrt(dx * dy * dz);
        // printf("iter = %d, sum_r_squared = %.20e\n", iter, sum_r_squared);
        if (sum_r_squared < tolerance) { break; }
    }
}

int main_p(int Nx, int Ny, int Nz, int Nt, double width, double height, double depth, double T, double rho,
           double mu_f) {
    auto ns_demo = create_ns_demo(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);

    double t  = 0.0;
    double dx = width / Nx;
    double dy = height / Ny;
    double dz = depth / Nz;
    // double dt = T / Nt;

    // 获取边界条件和右端项
    MultiArrayInt    pbt = ns_demo->get_boundary_type_p(all_boundary_type);
    MultiArrayDouble bp  = ns_demo->get_array_bp(t);
    MultiArrayDouble pbv = ns_demo->get_boundary_values_p(pbt, t);

    // 求解
    auto ph = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});
    auto r  = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});
    solve_p(ph, r, pbt, pbv, bp, Nx, Ny, Nz, width, height, depth, max_iters, tolerance);

    // 计算误差
    MultiArrayDouble exact = ns_demo->get_array_p(t);
    auto             eh    = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 2});
    algebra::axpy(-1.0, exact, ph, eh);
    algebra::zero_boundary(eh);
    double sum_eh_squared = algebra::norm(eh) * std::sqrt(dx * dy * dz);
    printf("sum_eh_squared = %.20e\n", sum_eh_squared);

    // std::ofstream out_pbt("pbt.txt");
    // std::ofstream out_bp("bp.txt");
    // std::ofstream out_p("p_exact.txt");
    // std::ofstream out_pbv("pbv.txt");
    // algebra::print_multi_array<3, int>(pbt, out_pbt);
    // algebra::print_multi_array<3, double>(bp, out_bp);
    // algebra::print_multi_array<3, double>(p, out_p);
    // algebra::print_multi_array<3, double>(pbv, out_pbv);
    return 0;
}

int main_u(int Nx, int Ny, int Nz, int Nt, double width, double height, double depth, double T, double rho,
           double mu_f) {
    double t       = 0.0;
    double dx      = width / Nx;
    double dy      = height / Ny;
    double dz      = depth / Nz;
    double dt      = T / Nt;
    auto   ns_demo = create_ns_demo(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);

    auto ubt = ns_demo->get_boundary_type_u(all_boundary_type);
    auto un  = ns_demo->get_array_u(0.0);

    std::ofstream out_ubt("ubt.txt");
    std::ofstream out_u0("u0.txt");
    algebra::print_multi_array<3, double>(un, out_u0);
    algebra::print_multi_array<3, int>(ubt, out_ubt);

    for (int i = 1; i <= Nt; i++) {
        t = i * dt;
        // printf("t = %f\n", t);

        // 获取边界条件和右端项
        auto f1  = ns_demo->get_array_f1(t);
        auto bu  = ns_demo->make_tentitive_ub(un, f1);
        auto ubv = ns_demo->get_boundary_values_u(ubt, t);

        // 求解
        auto uh = algebra::create_multi_array<3, double>({Nx + 1, Ny + 2, Nz + 2});
        auto rh = algebra::create_multi_array<3, double>({Nx + 1, Ny + 2, Nz + 2});
        velocity_u_3D::solve_u(rh, uh, ubt, ubv, bu, Nx, Ny, Nz, width, height, depth, mu_f, rho, dt, max_iters,
                               tolerance);

        // 更新un
        un = uh;

        // 计算误差
        auto exact = ns_demo->get_array_u(t);
        auto eh    = algebra::create_multi_array<3, double>({Nx + 1, Ny + 2, Nz + 2});
        algebra::axpy(-1.0, exact, un, eh);
        algebra::zero_boundary(eh);
        double sum_eh_squared = algebra::norm(eh) * std::sqrt(dx * dy * dz);
        if (i == Nt) printf("sum_eh_squared = %.20e\n", sum_eh_squared);

        // std::ofstream out_exact("u_exact.txt");
        // std::ofstream out_un("un.txt");
        // std::ofstream out_f1("f1.txt");
        // std::ofstream out_ubv("ubv.txt");
        // std::ofstream out_bu("bu.txt");
        // std::ofstream out_uh("uh.txt");
        // std::ofstream out_rh("rh.txt");
        // std::ofstream out_e("e.txt");
        // algebra::print_multi_array<3, double>(eh, out_e);
        // algebra::print_multi_array<3, double>(uh, out_uh);
        // algebra::print_multi_array<3, double>(rh, out_rh);
        // algebra::print_multi_array<3, double>(exact, out_exact);
        // algebra::print_multi_array<3, double>(un, out_un);
        // algebra::print_multi_array<3, double>(f1, out_f1);
        // algebra::print_multi_array<3, double>(ubv, out_ubv);
        // algebra::print_multi_array<3, double>(bu, out_bu);
    }

    return 0;
}

int main_v(int Nx, int Ny, int Nz, int Nt, double width, double height, double depth, double T, double rho,
           double mu_f) {
    double t       = 0.0;
    double dx      = width / Nx;
    double dy      = height / Ny;
    double dz      = depth / Nz;
    double dt      = T / Nt;
    auto   ns_demo = create_ns_demo(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);

    auto vbt = ns_demo->get_boundary_type_v(all_boundary_type);
    auto vn  = ns_demo->get_array_v(0.0);

    std::ofstream out_vbt("vbt.txt");
    std::ofstream out_v0("v0.txt");
    algebra::print_multi_array<3, double>(vn, out_v0);
    algebra::print_multi_array<3, int>(vbt, out_vbt);

    for (int i = 1; i <= Nt; i++) {
        t = i * dt;
        // printf("t = %f\n", t);

        // 获取边界条件和右端项
        auto f2  = ns_demo->get_array_f2(t);
        auto bv  = ns_demo->make_tentitive_vb(vn, f2);
        auto vbv = ns_demo->get_boundary_values_v(vbt, t);

        // 求解
        auto vh = algebra::create_multi_array<3, double>({Nx + 2, Ny + 1, Nz + 2});
        auto rh = algebra::create_multi_array<3, double>({Nx + 2, Ny + 1, Nz + 2});
        velocity_v_3D::solve_v(rh, vh, vbt, vbv, bv, Nx, Ny, Nz, width, height, depth, mu_f, rho, dt, max_iters,
                               tolerance);

        // 更新vn
        vn = vh;
        {
            // 计算误差
            auto exact = ns_demo->get_array_v(t);
            auto eh    = algebra::create_multi_array<3, double>({Nx + 2, Ny + 1, Nz + 2});
            algebra::axpy(-1.0, exact, vn, eh);
            algebra::zero_boundary(eh);
            double sum_eh_squared = algebra::norm(eh) * std::sqrt(dx * dy * dz);
            if (i == Nt) { printf("sum_eh_squared = %.20e\n", sum_eh_squared); }
            // std::ofstream out_e("e.txt");
            // std::ofstream out_r("r.txt");
            // std::ofstream out_vh("vh.txt");
            // std::ofstream out_vprime("v_prime.txt");
            // std::ofstream out_exact("v_exact.txt");
            // std::ofstream out_vn("vn.txt");
            // std::ofstream out_f2("f2.txt");
            // std::ofstream out_vbv("vbv.txt");
            // std::ofstream out_b("b.txt");
            // std::ofstream out_b_hat("b_hat.txt");
            // algebra::print_multi_array<3, double>(vh, out_vh);
            // algebra::print_multi_array<3, double>(exact, out_exact);
            // algebra::print_multi_array<3, double>(vn, out_vn);
            // algebra::print_multi_array<3, double>(f2, out_f2);
            // algebra::print_multi_array<3, double>(vbv, out_vbv);
            // algebra::print_multi_array<3, double>(v_prime, out_vprime);
            // algebra::print_multi_array<3, double>(bv, out_b);
            // algebra::print_multi_array<3, double>(bv_hat, out_b_hat);
            // algebra::print_multi_array<3, double>(eh, out_e);
            // algebra::print_multi_array<3, double>(r, out_r);
        }
    }

    return 0;
}

int main_w(int Nx, int Ny, int Nz, int Nt, double width, double height, double depth, double T, double rho,
           double mu_f) {
    double t       = 0.0;
    double dx      = width / Nx;
    double dy      = height / Ny;
    double dz      = depth / Nz;
    double dt      = T / Nt;
    auto   ns_demo = create_ns_demo(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);

    auto wbt = ns_demo->get_boundary_type_w(all_boundary_type);
    auto wn  = ns_demo->get_array_w(0.0);

    // std::ofstream out_wbt("wbt.txt");
    // std::ofstream out_w0("w0.txt");
    // algebra::print_multi_array<3, double>(wn, out_w0);
    // algebra::print_multi_array<3, int>(wbt, out_wbt);

    for (int i = 1; i <= Nt; i++) {
        t = i * dt;
        // printf("t = %f\n", t);

        // 获取边界条件和右端项
        auto f3  = ns_demo->get_array_f3(t);
        auto bw  = ns_demo->make_tentitive_wb(wn, f3);
        auto wbv = ns_demo->get_boundary_values_w(wbt, t);

        // 求解
        auto wh = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 1});
        auto rh = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 1});
        velocity_w_3D::solve_w(rh, wh, wbt, wbv, bw, Nx, Ny, Nz, width, height, depth, mu_f, rho, dt, max_iters,
                               tolerance);

        // 更新
        wn = wh;

        // 计算误差
        auto exact = ns_demo->get_array_w(t);

        auto eh = algebra::create_multi_array<3, double>({Nx + 2, Ny + 2, Nz + 1});
        algebra::axpy(-1.0, exact, wn, eh);
        algebra::zero_boundary(eh);

        double sum_eh_squared = algebra::norm(eh) * std::sqrt(dx * dy * dz);

        if (i == Nt) printf("sum_eh_squared = %.20e\n", sum_eh_squared);

        // std::ofstream out_wh("wh.txt");
        // algebra::print_multi_array<3, double>(wh, out_wh);
        // std::ofstream out_e("e.txt");
        // algebra::print_multi_array<3, double>(eh, out_e);
        // std::ofstream out_r("r.txt");
        // algebra::print_multi_array<3, double>(r, out_r);
        // std::ofstream out_exact("w_exact.txt");
        // std::ofstream out_wn("wn.txt");
        // std::ofstream out_f3("f3.txt");
        // std::ofstream out_wbv("wbv.txt");
        // std::ofstream out_b("b.txt");
        // std::ofstream out_b_hat("b_hat.txt");
        // std::ofstream out_wprime("w_prime.txt");
        // algebra::print_multi_array<3, double>(w_prime, out_wprime);
        // algebra::print_multi_array<3, double>(exact, out_exact);
        // algebra::print_multi_array<3, double>(wn, out_wn);
        // algebra::print_multi_array<3, double>(f3, out_f3);
        // algebra::print_multi_array<3, double>(wbv, out_wbv);
        // algebra::print_multi_array<3, double>(bw, out_b);
        // algebra::print_multi_array<3, double>(bw_hat, out_b_hat);
    }

    return 0;
}

void main_test(int N, int Nt) {
    int    Nx     = N;
    int    Ny     = N;
    int    Nz     = N;
    double width  = 1.0;
    double height = 1.0;
    double depth  = 1.0;
    double T      = 1.0;
    double rho    = 1.0;
    double mu_f   = 1.0;
    // main_p(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);
    main_u(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);
    // main_v(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);
    // main_w(Nx, Ny, Nz, Nt, width, height, depth, T, rho, mu_f);
}

int main() {
    std::vector<int> Ns{4, 8, 16, 32};
    std::vector<int> Nt{128, 256, 512, 1024, 2048};

    for (auto& ns : Ns) {
        std::string msg = fmt::format("Result for ns = {}:", ns);
        std::cout << msg << std::endl;
        for (auto& nt : Nt) {
            std::string msg = fmt::format("           nt = {}:", nt);
            std::cout << msg << std::endl;
            main_test(ns, nt);
        }
    }
    return 0;
}