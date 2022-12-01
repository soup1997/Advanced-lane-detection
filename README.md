# Advanced-lane-detection
## Sliding Window with Kalman Filter
* 곡선 차선 인식을 위한 Sliding Window 알고리즘 적용
* 단점: 다음 조사창의 위치를 이전 조사창의 평균을 기준으로 차선을 탐색하기 때문에 차선을 놓치는 경우 명백한 한계점이 존재
* 목적 및 아이디어: Kalman Filter의 학습 및 응용, 다음 조사창의 위치를 단순한 평균으로 구하지 않고 Kalman Filter의 Estimation(Prediction, Update) 과정을 통해 다음 조사창의 위치를 결정 

'State Matrix'
> x = [[Pixel's Y Postion]]
