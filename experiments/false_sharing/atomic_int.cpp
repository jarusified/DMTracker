#include <atomic>
#include <iostream>

// Our aligned atomic
// alignas makes sure the data_types have a specified bytes away.
struct alignas(64) AlignedType {
  AlignedType() { val = 0; }
  std::atomic<int> val;
};

int main()
{
    // If we create four atomic integers like this, there's a high probability
    // they'll wind up next to each other in memory
    std::atomic<int> a;
    std::atomic<int> b;
    std::atomic<int> c;
    std::atomic<int> d;

    // Print out the addresses
    std::cout << "Address of atomic<int> a - " << &a << '\n';
    std::cout << "Address of atomic<int> b - " << &b << '\n';
    std::cout << "Address of atomic<int> c - " << &c << '\n';
    std::cout << "Address of atomic<int> d - " << &d << '\n';

    AlignedType A{};
    AlignedType B{};
    AlignedType C{};
    AlignedType D{};

    std::cout << "Address of AlignedType A - " << &A << '\n';
    std::cout << "Address of AlignedType B - " << &B << '\n';
    std::cout << "Address of AlignedType C - " << &C << '\n';
    std::cout << "Address of AlignedType D - " << &D << '\n';

    return 0;
}