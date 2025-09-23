# 인공지능을 이용한 지진 분석 실습

## 실습을 위한 명령어 모음

1. 구글 colab을 활용한 실습에 필요한 초기 명령어
   
```python
!pip install obspy
```
   
2. 지진파 위상도달시각 및 진원 결정 실습에 필요한 자료와 코드 다운로드 방법
- 2024년 6월 12일 규모 4.8 부안지진의 지진자료에서 인공지능을 활용하여 지진파 위상도달시각을 결정
- 단순화시킨 비선형 역산 함수를 사용해 지진의 진원을 결정하는 실습 예제
    
```python
!wget https://github.com/dhsheen/KFpicker/raw/refs/heads/main/KFpicker_20230217.h5

!wget https://github.com/dhsheen/AI_seismology/raw/refs/heads/main/PhasePicking/EQLocateDL.py

https://github.com/dhsheen/AI_seismology/raw/refs/heads/main/PhasePicking/buan2024_practice.pkl

```


