import jax.numpy as jnp
from jax import grad

# 선형회귀 모델 함수
def linear_regression(X, w):
    return jnp.dot(X, w.T)

# 평균제곱오차 손실 함수
def mse_loss(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

# 경사하강법 함수
def gradient_descent(w, X, y, learning_rate):
    grad_func = grad(mse_loss)
    grad_w = grad_func(y, linear_regression(X, w))
    return w - learning_rate * grad_w

# 입력과 출력 정의
X = jnp.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
y = jnp.array([[2.], [4.], [6.], [8.]])

# 가중치 초기값 설정
w = jnp.array([[1., 1.]])

# 학습
for epoch in range(10):
    # 예측 계산
    y_pred = linear_regression(X, w)

    # 손실 계산
    loss = mse_loss(y, y_pred)
    print(f"에포크 {epoch}\n가중치: {w}\n손실: {loss}\n예측결과: {y_pred}")

    # 경사하강법을 사용하여 가중치 갱신
    w = gradient_descent(w, X, y, 0.1)
