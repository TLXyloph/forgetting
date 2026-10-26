// AHC014 "RectJoin": reads the input and plays no moves (a valid answer with zero operations).
#include <cstdio>

int main() {
    int n = 0, m = 0;
    if (std::scanf("%d %d", &n, &m) != 2) return 1;
    for (int i = 0; i < m; ++i) {
        int x = 0, y = 0;
        if (std::scanf("%d %d", &x, &y) != 2) return 1;
    }
    std::printf("0\n");
    return 0;
}
