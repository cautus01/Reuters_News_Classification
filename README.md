# Reuters_News_Classification

## 프로젝트 개요

**진행기간 : 2021.11.7 ~ 2021.12.6**
 
**주요내용**
 
- **로이터 뉴스 데이터의 특징을 파악한다. 그 후 LSTM, Transformer, CNN 모델을 설계하여 텍스트 분류를 진행한다.**

**사용한 skill : keras, tensorflow, colab 등**

**본인이 기여한 점 : 로이터 뉴스 데이터 분석을 하였다. matplotlib를 이용하여 LSTM, Transformer, CNN의 모델 결과(acc와 loss)를 시각화하였다.**

**어려웠던점: 분석한 데이터를 matplotlib를 이용하여 보기 좋게 데이터를 시각화하는 것이 어려웠다. 어떻게 하면 더 보기 좋고 쉽게 데이터를 파악할 수 있는지 생각을 많이 했다.**

**결과**
- **LSTM은 Activation과 optimizer 조합에 따라 유의미한 정확도 차이 발생하였고, Sigmoid actaviton 방식이 softmax 방식보다 전체적으로 정확도가 높았다.**

- **Transformer는 직접 만든 코드가 케라스가 제공하는 멀티 헤드 어텐션보다 약 10배 빨랐다. 생각보다 데이터 양이 적고, 여러 레이어를 거치는 특성상 과적합이 높게 나왔다. 과적합 해결을 위해 Dropout 비율을 높이거나, L2 정규화, 여러 optimizer 사용 등을 시도했다.**

- **CNN은 단순 CNN으로 진행한 결과는 정확도가 높지 않음을 확인하였다. 정확도를 올리기 위해 커널 사이즈, 데이터 크기 등을 조절하고 Dropout 비율 변경 등의 시도를 했으나 유의미한 결과를 얻지는 못했고, CNN과 LSTM 모델의 조합을 통해 더 높은 정확도를 얻을 수 있었다.**


## 코드 및 역할

[**LINK**](https://github.com/cautus01/Reuters_News_Classification/blob/main/Reuters%20News%20Classification.py) 참고
