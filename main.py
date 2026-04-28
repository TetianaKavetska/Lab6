import numpy as np

def save_matrix(filename, A):
    np.savetxt(filename, A)

def load_matrix(filename):
    return np.loadtxt(filename)

def save_vector(filename, B):
    np.savetxt(filename, B)

def load_vector(filename):
    return np.loadtxt(filename)



def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        U[i][i] = 1

    for k in range(n):
        for i in range(k, n):
            L[i][k] = A[i][k] - sum(L[i][j] * U[j][k] for j in range(k))

        for j in range(k + 1, n):
            U[k][j] = (A[k][j] - sum(L[k][m] * U[m][j] for m in range(k))) / L[k][k]

    return L, U


def forward_substitution(L, B):
    n = len(B)
    Z = np.zeros(n)

    for i in range(n):
        Z[i] = (B[i] - sum(L[i][j] * Z[j] for j in range(i))) / L[i][i]

    return Z


def backward_substitution(U, Z):
    n = len(Z)
    X = np.zeros(n)

    for i in range(n - 1, -1, -1):
        X[i] = Z[i] - sum(U[i][j] * X[j] for j in range(i + 1, n))

    return X



def mat_vec(A, x):
    return np.dot(A, x)

def vector_norm(v):
    return np.max(np.abs(v))


def main():
    n = 100


    np.set_printoptions(threshold=np.inf)

    #матриця A
    A = np.random.rand(n, n) * 10
    save_matrix("A.txt", A)


    X_true = np.full(n, 2.5)


    B = mat_vec(A, X_true)
    save_vector("B.txt", B)

    print("===== ЗГЕНЕРОВАНА МАТРИЦЯ A =====")
    print(A)

    print("\n===== ВЕКТОР B =====")
    print(B)


    A = load_matrix("A.txt")
    B = load_vector("B.txt")


    L, U = lu_decomposition(A)

    print("\n===== МАТРИЦЯ L =====")
    print(L)

    print("\n===== МАТРИЦЯ U =====")
    print(U)


    Z = forward_substitution(L, B)
    X = backward_substitution(U, Z)

    print("\n===== РОЗВ’ЯЗОК X =====")
    print(X)


    eps = vector_norm(mat_vec(A, X) - B)
    print("\nПочаткова похибка eps =", eps)


    eps0 = 1e-14
    iterations = 0

    while eps > eps0:
        r = B - mat_vec(A, X)

        Z = forward_substitution(L, r)
        dX = backward_substitution(U, Z)

        X = X + dX
        eps = vector_norm(mat_vec(A, X) - B)
        iterations += 1

    print("\n===== УТОЧНЕНИЙ РОЗВ’ЯЗОК =====")
    print(X)

    print("\nКількість ітерацій:", iterations)
    print("Фінальна похибка:", eps)


if __name__ == "__main__":
    main()
