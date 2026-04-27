class MyClass {
  public:
    template <typename T>
    void my_function(T arg) {
        // 函数实现
    }
};

int main() {
    MyClass* obj = new MyClass();
    obj->my_function<int>(42); // 显式调用模板函数，传递一个int类型的参数
    delete obj;
    return 0;
}
