def kmap(x,n):
    digits = [(x // 10**(n-i-1)) % 10 for i in range(0,n)]

    digits.sort()
    min_num = 0
    for i in range(n):
        min_num = min_num + digits[i] * (10**(n-i-1))
    
    digits.reverse()
    max_num = 0
    for i in range(n):
        max_num = max_num + digits[i] * (10**(n-i-1))
    
    return max_num - min_num

n = 3
for i in range(10**(n-1),10**n):
    if kmap(i,n) == i:
        print(i)