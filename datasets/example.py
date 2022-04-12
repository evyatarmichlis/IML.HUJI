
def myPow(self, x: float, n: int) -> float:
    i = 0
    while i < n:
        x *= x
        i += 1
    return x

if __name__ == '__main__':
    print(pow(2.00000,10))