#include <iostream>
#include <stack>
#include <stdlib.h>
#include <string>
#include <vector>

class Node: public std::enable_shared_from_this<Node> {
 public:
  Node() {
    // shared_from_this(); // bad_weak_ptr
    //
    // This is because there's no strong reference to the instance exists yet.
    // But here is one magic from:
    //
    //   https://forum.libcinder.org/topic/\
    //   solution-calling-shared-from-this-in-the-constructor
    auto wptr = std::shared_ptr<Node>(this, [](Node *) { });
    shared_from_this();
  }
};

// Unfortunately, the above solution can go wrong in certain cases.
// Thus use a factory function, or avoid this situation entirely.
class Node2: public std::enable_shared_from_this<Node2> {
 public:
  Node2(int foo) : foo_(foo) { }
  void share() { shared_from_this(); }
  int foo() { return foo_; }
 private:
  int foo_;
};

namespace factory {

std::shared_ptr<Node2> CreateNode2(int foo) {
  auto ptr = std::make_shared<Node2>(foo);
  ptr->share();
  return ptr;
}

}

int main() {
  std::cout << "constructor" << std::endl;
  std::shared_ptr<Node> x = std::make_shared<Node>();
  std::cout << "okay" << std::endl << std::endl;

  std::cout << "factory" << std::endl;
  std::shared_ptr<Node2> y = factory::CreateNode2(3);
  std::cout << "okay " << y->foo() << std::endl << std::endl;
  return 0;
}
