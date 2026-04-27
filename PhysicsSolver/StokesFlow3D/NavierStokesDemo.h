#pragma once

#include <muParser.h>
#include <nlohmann/json.hpp>

#include "some_headers.h"

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
    template <typename T>
    T get_value(const std::string& key) {
        if (jsonData.find(key) != jsonData.end()) {
            if constexpr (std::is_same_v<T, double>) {
                if (jsonData[key].is_number()) {
                    return jsonData[key].get<double>();
                } else {
                    CHECK_F(false, "键存在但值不是数字：%s", key.c_str());
                }
            } else if constexpr (std::is_same_v<T, std::string>) {
                if (jsonData[key].is_string()) {
                    return jsonData[key].get<std::string>();
                } else {
                    CHECK_F(false, "键存在但值不是字符串：%s", key.c_str());
                }
            } else {
                CHECK_F(false, "仅支持 double 和 std::string 类型");
            }
        } else {
            CHECK_F(false, "键不存在：%s", key.c_str());
        }
        // 函数不会运行到这里，以下语句只为注明返回类型。
        return T{};
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

  public:
    int    Nx, Ny, Nz, Nt;
    double Lx, Ly, Lz, dt, dx, dy, dz, T, rho, mu;
    int    DIRICHLET = 1;
    int    NEUMANN   = 2;

    NavierStokesDemo(JsonFile json_file)
        : _u(json_file["u"]), _v(json_file["v"]), _w(json_file["w"]), _p(json_file["p"]), _dudx(json_file["dudx"]),
          _dudy(json_file["dudy"]), _dudz(json_file["dudz"]), _dvdx(json_file["dvdx"]), _dvdy(json_file["dvdy"]),
          _dvdz(json_file["dvdz"]), _dwdx(json_file["dwdx"]), _dwdy(json_file["dwdy"]), _dwdz(json_file["dwdz"]),
          _dpdx(json_file["dpdx"]), _dpdy(json_file["dpdy"]), _dpdz(json_file["dpdz"]), _f1(json_file["f1"]),
          _f2(json_file["f2"]), _f3(json_file["f3"]), _bp(json_file["bp"]) {}

    NavierStokesDemo()
        : _u("0"), _v("0"), _w("0"), _p("0"), _dudx("0"), _dudy("0"), _dudz("0"), _dvdx("0"), _dvdy("0"), _dvdz("0"),
          _dwdx("0"), _dwdy("0"), _dwdz("0"), _dpdx("0"), _dpdy("0"), _dpdz("0"), _f1("0"), _f2("0"), _f3("0"),
          _bp("0") {}

    void set(int Nx, int Ny, int Nz, int Nt, double Lx, double Ly, double Lz, double T, double rho, double mu) {
        this->Nx  = Nx;
        this->Ny  = Ny;
        this->Nz  = Nz;
        this->Nt  = Nt;
        this->Lx  = Lx;
        this->Ly  = Ly;
        this->Lz  = Lz;
        this->T   = T;
        this->rho = rho;
        this->mu  = mu;
        dx        = Lx / Nx;
        dy        = Ly / Ny;
        dz        = Lz / Nz;
        dt        = T / Nt;
    }

    void get_array_v(MultiArrayDouble& data, double t) {
        for (int i = -1; i <= Nx; ++i) {
            for (int j = 0; j <= Ny; ++j) {
                for (int k = -1; k <= Nz; ++k) {
                    data[i + 1][j + 1][k + 1] = _v((i + 0.5) * dx, j * dy, (k + 0.5) * dz, t);
                }
            }
        }
    }

    void get_array_u(MultiArrayDouble& data, double t) {
        for (int i = 0; i < Nx + 1; ++i) {
            for (int j = -1; j < Ny + 1; ++j) {
                for (int k = -1; k < Nz + 1; ++k) {
                    data[i + 1][j + 1][k + 1] = _u(i * dx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }
    }

    void get_array_w(MultiArrayDouble& data, double t) {
        for (int i = -1; i < Nx + 1; ++i) {
            for (int j = -1; j < Ny + 1; ++j) {
                for (int k = 0; k < Nz + 1; ++k) {
                    data[i + 1][j + 1][k + 1] = _w((i + 0.5) * dx, (j + 0.5) * dy, k * dz, t);
                }
            }
        }
    }

    void get_array_f1(MultiArrayDouble& data, double t) {
        for (int i = 0; i < Nx + 1; ++i) {
            for (int j = -1; j < Ny + 1; ++j) {
                for (int k = -1; k < Nz + 1; ++k) {
                    data[i + 1][j + 1][k + 1] = _f1(i * dx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }
    }

    void get_array_f2(MultiArrayDouble& data, double t) {
        for (int i = -1; i <= Nx; ++i) {
            for (int j = 0; j <= Ny; ++j) {
                for (int k = -1; k <= Nz; ++k) {
                    data[i + 1][j + 1][k + 1] = _f2((i + 0.5) * dx, j * dy, (k + 0.5) * dz, t);
                }
            }
        }
    }

    void get_array_f3(MultiArrayDouble& data, double t) {
        for (int i = -1; i < Nx + 1; ++i) {
            for (int j = -1; j < Ny + 1; ++j) {
                for (int k = 0; k < Nz + 1; ++k) {
                    data[i + 1][j + 1][k + 1] = _f3((i + 0.5) * dx, (j + 0.5) * dy, k * dz, t);
                }
            }
        }
    }

    void get_boundary_type_u(MultiArrayInt& boundary_types, const std::array<int, 6>& all_boundary_type) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                boundary_types[1][j + 1][k + 1]      = all_boundary_type[0];
                boundary_types[Nx + 1][j + 1][k + 1] = all_boundary_type[1];
            }
        }
        for (int i = 0; i <= Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                boundary_types[i + 1][0][k + 1]      = all_boundary_type[2];
                boundary_types[i + 1][Ny + 1][k + 1] = all_boundary_type[3];
            }
        }
        for (int i = 0; i <= Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                boundary_types[i + 1][j + 1][0]      = all_boundary_type[4];
                boundary_types[i + 1][j + 1][Nz + 1] = all_boundary_type[5];
            }
        }
    }

    void get_boundary_type_v(MultiArrayInt& boundary_types, const std::array<int, 6>& all_boundary_type) {
        for (int j = 0; j <= Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                boundary_types[0][j + 1][k + 1]      = all_boundary_type[0];
                boundary_types[Nx + 1][j + 1][k + 1] = all_boundary_type[1];
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                boundary_types[i + 1][1][k + 1]      = all_boundary_type[2];
                boundary_types[i + 1][Ny + 1][k + 1] = all_boundary_type[3];
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j <= Ny; ++j) {
                boundary_types[i + 1][j + 1][0]      = all_boundary_type[4];
                boundary_types[i + 1][j + 1][Nz + 1] = all_boundary_type[5];
            }
        }
    }

    void get_boundary_type_w(MultiArrayInt& boundary_types, const std::array<int, 6>& all_boundary_type) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k <= Nz; ++k) {
                boundary_types[0][j + 1][k + 1]      = all_boundary_type[0];
                boundary_types[Nx + 1][j + 1][k + 1] = all_boundary_type[1];
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k <= Nz; ++k) {
                boundary_types[i + 1][0][k + 1]      = all_boundary_type[2];
                boundary_types[i + 1][Ny + 1][k + 1] = all_boundary_type[3];
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                boundary_types[i + 1][j + 1][1]      = all_boundary_type[4];
                boundary_types[i + 1][j + 1][Nz + 1] = all_boundary_type[5];
            }
        }
    }

    void make_tentitive_ub(MultiArrayDouble& ub, const MultiArrayDouble& un, const MultiArrayDouble& f) {
        for (int i = 0; i < Nx + 3; ++i) {
            for (int j = 0; j < Ny + 2; ++j) {
                for (int k = 0; k < Nz + 2; ++k) {
                    ub[i][j][k] = rho * un[i][j][k] / dt + f[i][j][k];
                }
            }
        }
    }

    void make_tentitive_vb(MultiArrayDouble& vb, const MultiArrayDouble& vn, const MultiArrayDouble& f) {
        for (int i = 0; i < Nx + 2; ++i) {
            for (int j = 0; j < Ny + 3; ++j) {
                for (int k = 0; k < Nz + 2; ++k) {
                    vb[i][j][k] = rho * vn[i][j][k] / dt + f[i][j][k];
                }
            }
        }
    }

    void make_tentitive_wb(MultiArrayDouble& wb, const MultiArrayDouble& wn, const MultiArrayDouble& f) {
        for (int i = 0; i < Nx + 2; ++i) {
            for (int j = 0; j < Ny + 2; ++j) {
                for (int k = 0; k < Nz + 3; ++k) {
                    wb[i][j][k] = rho * wn[i][j][k] / dt + f[i][j][k];
                }
            }
        }
    }

    void get_boundary_values_u(MultiArrayDouble& data, const MultiArrayInt& boundary_type, double t) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[1][j + 1][k + 1] == DIRICHLET) {
                    data[1][j + 1][k + 1] = _u(0.0, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[1][j + 1][k + 1] == NEUMANN) {
                    data[1][j + 1][k + 1] = _dudx(0.0, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) {
                    data[Nx + 1][j + 1][k + 1] = _u(Lx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) {
                    data[Nx + 1][j + 1][k + 1] = _dudx(Lx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }
        for (int i = 0; i <= Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i + 1][0][k + 1] == DIRICHLET) {
                    data[i + 1][0][k + 1] = _u(i * dx, 0.0, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][0][k + 1] == NEUMANN) {
                    data[i + 1][0][k + 1] = _dudy(i * dx, 0.0, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) {
                    data[i + 1][Ny + 1][k + 1] = _u(i * dx, Ly, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) {
                    data[i + 1][Ny + 1][k + 1] = _dudy(i * dx, Ly, (k + 0.5) * dz, t);
                }
            }
        }
        for (int i = 0; i <= Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                if (boundary_type[i + 1][j + 1][0] == DIRICHLET) {
                    data[i + 1][j + 1][0] = _u(i * dx, (j + 0.5) * dy, 0, t);
                }
                if (boundary_type[i + 1][j + 1][0] == NEUMANN) {
                    data[i + 1][j + 1][0] = _dudz(i * dx, (j + 0.5) * dy, 0, t);
                }
                if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) {
                    data[i + 1][j + 1][Nz + 1] = _u(i * dx, (j + 0.5) * dy, Lz, t);
                }
                if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) {
                    data[i + 1][j + 1][Nz + 1] = _dudz(i * dx, (j + 0.5) * dy, Lz, t);
                }
            }
        }
    }

    void get_boundary_values_v(MultiArrayDouble& data, const MultiArrayInt& boundary_type, double t) {
        for (int j = 0; j <= Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[0][j + 1][k + 1] == DIRICHLET) {
                    data[0][j + 1][k + 1] = _v(0.0, j * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[0][j + 1][k + 1] == NEUMANN) {
                    data[0][j + 1][k + 1] = _dvdx(0.0, j * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) {
                    data[Nx + 1][j + 1][k + 1] = _v(Lx, j * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) {
                    data[Nx + 1][j + 1][k + 1] = _dvdx(Lx, j * dy, (k + 0.5) * dz, t);
                }
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[i + 1][1][k + 1] == DIRICHLET) {
                    data[i + 1][1][k + 1] = _v((i + 0.5) * dx, 0.0, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][1][k + 1] == NEUMANN) {
                    data[i + 1][1][k + 1] = _dvdy((i + 0.5) * dx, 0.0, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) {
                    data[i + 1][Ny + 1][k + 1] = _v((i + 0.5) * dx, Ly, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) {
                    data[i + 1][Ny + 1][k + 1] = _dvdy((i + 0.5) * dx, Ly, (k + 0.5) * dz, t);
                }
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j <= Ny; ++j) {
                if (boundary_type[i + 1][j + 1][0] == DIRICHLET) {
                    data[i + 1][j + 1][0] = _v((i + 0.5) * dx, j * dy, 0, t);
                }
                if (boundary_type[i + 1][j + 1][0] == NEUMANN) {
                    data[i + 1][j + 1][0] = _dvdz((i + 0.5) * dx, j * dy, 0, t);
                }
                if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) {
                    data[i + 1][j + 1][Nz + 1] = _v((i + 0.5) * dx, j * dy, Lz, t);
                }
                if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) {
                    data[i + 1][j + 1][Nz + 1] = _dvdz((i + 0.5) * dx, j * dy, Lz, t);
                }
            }
        }
    }

    void get_boundary_values_w(MultiArrayDouble& data, const MultiArrayInt& boundary_type, double t) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k <= Nz; ++k) {
                if (boundary_type[0][j + 1][k + 1] == DIRICHLET) {
                    data[0][j + 1][k + 1] = _w(0.0, (j + 0.5) * dy, k * dz, t);
                }
                if (boundary_type[0][j + 1][k + 1] == NEUMANN) {
                    data[0][j + 1][k + 1] = _dwdx(0.0, (j + 0.5) * dy, k * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) {
                    data[Nx + 1][j + 1][k + 1] = _w(Lx, (j + 0.5) * dy, k * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) {
                    data[Nx + 1][j + 1][k + 1] = _dwdx(Lx, (j + 0.5) * dy, k * dz, t);
                }
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int k = 0; k <= Nz; ++k) {
                if (boundary_type[i + 1][0][k + 1] == DIRICHLET) {
                    data[i + 1][0][k + 1] = _w((i + 0.5) * dx, 0.0, k * dz, t);
                }
                if (boundary_type[i + 1][0][k + 1] == NEUMANN) {
                    data[i + 1][0][k + 1] = _dwdy((i + 0.5) * dx, 0.0, k * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k + 1] == DIRICHLET) {
                    data[i + 1][Ny + 1][k + 1] = _w((i + 0.5) * dx, Ly, k * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) {
                    data[i + 1][Ny + 1][k + 1] = _dwdy((i + 0.5) * dx, Ly, k * dz, t);
                }
            }
        }

        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                if (boundary_type[i + 1][j + 1][1] == DIRICHLET) {
                    data[i + 1][j + 1][1] = _w((i + 0.5) * dx, (j + 0.5) * dy, 0, t);
                }
                if (boundary_type[i + 1][j + 1][1] == NEUMANN) {
                    data[i + 1][j + 1][1] = _dwdz((i + 0.5) * dx, (j + 0.5) * dy, 0, t);
                }
                if (boundary_type[i + 1][j + 1][Nz + 1] == DIRICHLET) {
                    data[i + 1][j + 1][Nz + 1] = _w((i + 0.5) * dx, (j + 0.5) * dy, Lz, t);
                }
                if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) {
                    data[i + 1][j + 1][Nz + 1] = _dwdz((i + 0.5) * dx, (j + 0.5) * dy, Lz, t);
                }
            }
        }
    }

    void get_array_bp(MultiArrayDouble& data, double t) {
        for (int i = 0; i < Nx; ++i) {
            for (int j = 0; j < Ny; ++j) {
                for (int k = 0; k < Nz; ++k) {
                    data[i + 1][j + 1][k + 1] = _bp((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }
    }

    void get_array_p(MultiArrayDouble& data, double t) {
        for (int i = -1; i < Nx + 1; ++i) {
            for (int j = -1; j < Ny + 1; ++j) {
                for (int k = -1; k < Nz + 1; ++k) {
                    data[i + 1][j + 1][k + 1] = _p((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
            }
        }
    }

    void get_boundary_values_p(MultiArrayDouble& data, const MultiArrayInt& boundary_type, double t) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                if (boundary_type[0][j + 1][k + 1] == DIRICHLET) {
                    data[0][j + 1][k + 1] = _p(0.0, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[0][j + 1][k + 1] == NEUMANN) {
                    data[0][j + 1][k + 1] = _dpdx(0.0, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k + 1] == DIRICHLET) {
                    data[Nx + 1][j + 1][k + 1] = _p(Lx, (j + 0.5) * dy, (k + 0.5) * dz, t);
                }
                if (boundary_type[Nx + 1][j + 1][k + 1] == NEUMANN) {
                    data[Nx + 1][j + 1][k + 1] = _dpdx(Lx, (j + 0.5) * dy, (k + 0.5) * dz, t);
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
                    data[i + 1][Ny + 1][k + 1] = _p((i + 0.5) * dx, Ly, (k + 0.5) * dz, t);
                }
                if (boundary_type[i + 1][Ny + 1][k + 1] == NEUMANN) {
                    data[i + 1][Ny + 1][k + 1] = _dpdy((i + 0.5) * dx, Ly, (k + 0.5) * dz, t);
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
                    data[i + 1][j + 1][Nz + 1] = _p((i + 0.5) * dx, (j + 0.5) * dy, Lz, t);
                }
                if (boundary_type[i + 1][j + 1][Nz + 1] == NEUMANN) {
                    data[i + 1][j + 1][Nz + 1] = _dpdz((i + 0.5) * dx, (j + 0.5) * dy, Lz, t);
                }
            }
        }
    }

    void get_boundary_type_p(MultiArrayInt& boundary_types, const std::array<int, 6>& all_boundary_type) {
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
    }

    ~NavierStokesDemo() {}
};

auto create_ns_demo(int Nx, int Ny, int Nz, int Nt, double Lx, double Ly, double Lz, double T, double rho, double mu,
                    std::string json_file_ = "") {

    std::shared_ptr<NavierStokesDemo> ns_demo = nullptr;
    if (json_file_.empty()) {
        ns_demo = std::make_shared<NavierStokesDemo>();
    } else {
        JsonFile json_file(json_file_);
        ns_demo = std::make_shared<NavierStokesDemo>(json_file);
    }
    ns_demo->set(Nx, Ny, Nz, Nt, Lx, Ly, Lz, T, rho, mu);
    return ns_demo;
}
