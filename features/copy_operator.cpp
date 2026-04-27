/// @date 2023-04-06
/// @file copy_operator.cpp
/// @author Ma Pengfei (code@pengfeima.cn)
/// @version 0.1
/// @copyright Copyright (c) 2023 Ma Pengfei
///
/// @brief 基类是一个模板类，在派生类中重载基类的拷贝赋值运算符，并实现它的纯虚函数
///
///

#include <iostream>

template <typename T>
class Base {
  public:
    virtual ~Base() = default;
    virtual Base& operator=(const Base& other) { printf("调用 Base 类的 operator=\n"); };
    virtual int   fun() {
        // 返回数据。
    }
};

template <typename T>
class Derived : public Base<T> {
  public:
    Derived& operator=(const Derived& other) {
        // 检查自我赋值
        if (this == &other) { return *this; }

        // 调用基类的拷贝赋值运算符
        Base<T>::operator=(other);

        // 执行派生类自身的操作
        // ...

        return *this;
    }

    // 实现基类的纯虚函数
    // 这个参数的意思是，
    Derived& operator=(const Base<T>& other) override {
        printf("调用 Derived 类的 Derived& operator=(const Base<T>& other) \n");
        const Derived& derivedOther = dynamic_cast<const Derived&>(other);

        // 调用派生类自身的拷贝赋值运算符
        operator=(derivedOther);

        return *this;
    }
};

// virtual bool assign(IObject const *other) override {
//     auto src = dynamic_cast<Derived const *>(other);
//     if (!src)
//         return false;
//     auto dst = static_cast<Derived *>(this);
//     *dst = *src;
//     return true;
// }

int main() {
    Derived<int>  d_int;
    Derived<int>& d_int_1 = d_int;

    Base<int>  b_int;
    Base<int>& b_int_1 = b_int;

    // 错误，指针的实际类型为基类类型，不能转换为派生类类型
    // Derived<int>* pd1 = new Base<int>();
    Base<int>* pd2 = new Derived<int>();
    Base<int>& pd3 = *pd2;

    // 正确，向上转换始终是安全的。
    dynamic_cast<Base<int>&>(d_int_1);
    // 错误，对象的实际类型必须是派生类的类型。
    // dynamic_cast<Derived<int>&>(b_int_1);
    // 正确
    dynamic_cast<Derived<int>*>(pd2);
    // 正确
    dynamic_cast<Derived<int>&>(pd3);
}