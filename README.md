# FOODING

---

![단체사진](https://user-images.githubusercontent.com/96258104/175428073-5d2d214f-929c-4631-ac37-0192f90bceb3.png)

---

## 팀원 소개

- ### 개발자
- #### 김수진(배재대학교)
- #### 심상현(숭실대학교)
- #### 이승용(동서대학교)
- #### 정은성(인제대학교)
- ### 디자이너
- #### 김세은(인제대학교)

---

## 프로젝트 배경/목표

##### 배달비와 배달음식 비용이 점점 급증하고 있어 혼자나 소수에서 시켜먹기 부담스러워지고 있습니다.
##### 채팅으로 배달 메이트를 매칭시켜 배달비와 배달음식 비용의 부담을 덜어주는 웹페이지 제작을 하게 되었습니다.
##### '뭘 먹을까?'에 대한 고민을 줄여주기 위하여 나이대, 현재 기분, 현재 날씨, 시켜먹는 시간대 등을 선택하면
##### 달달한 맛, 짭짤한 맛, 매콤한 맛을 추천해주는 기능도 포함하려고 합니다.

--- 

## 동작 방식(프로세스)
### 로그인 환경에서 firebase와의 통신을 통해 회원가입 / 로그인 / 아이디찾기 / 비밀번호 재설정 / 내 정보 변경과 같은 동작을 구현할 수 있다.
### 또한 채팅방을 만들고 접속하는 과정을 실행할 수 있다.
### 내 인터넷 접속 위치를 갖고 내 주변 음식점과 내가 원하는 위치 주변의 음식점을 확인할 수 있다.
### 여러가지 변수(나이대, 성별, 시간대)를 통해 사용자가 원하는 종류의 맛을 추천하고, 관련된 음식 또한 추천할 수 있다.

---

## 개발/배포 환경

### 개발 환경
  - ##### anaconda navigator 2.1.0
  - ##### anaconda 4.12.0
  - ##### python 3.8.12
  #### pip
  - ##### tensorflow 2.7
  - ##### keras 2.7
  - ##### tensorflowjs 3.18
  - ##### matplotlib 3.5
  - ##### numpy 1.21.2
  - ##### pandas 1.4.2
  - ##### scikit-learn 1.1.1
  #### firebase

---

## 활용/참고 정보

#### 원-핫 인코딩
##### https://wikidocs.net/22647
#### 다중 분류란?
##### https://yhyun225.tistory.com/14
#### 소프트맥스
##### https://blog.naver.com/hobbang143/221735605346
#### logistic regression
##### https://velog.io/@arittung/DeepLearningStudyDay7
##### https://blog.naver.com/PostView.naver?blogId=psycomaniac&logNo=222472359036&parentCategoryNo=&categoryNo=7&viewDate=&isShowPopularPosts=true&from=search
#### decision tree
##### https://eunsour.tistory.com/57
##### https://www.google.com/search?q=decision+tree+%EB%8B%A4%EC%A4%91%EB%B6%84%EB%A5%98&newwindow=1&source=lnms&tbm=isch&sa=X&ved=2ahUKEwjSs-2Xz8P4AhXam1YBHT0CDQcQ_AUoAXoECAEQAw&cshid=1655988991854045&biw=1112&bih=837&dpr=1#imgrc=oMdHjv-s9ZPsdM
#### random forest
##### https://hayden-archive.tistory.com/300
#### 딥러닝 다중분류 모델
##### https://incomeplus.tistory.com/243
##### https://knowledgeforengineers.tistory.com/138
#### compile 파라미터
##### https://blog.naver.com/handuelly?Redirect=Log&logNo=221822938182
#### fit 파라미터
##### https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit?hl=ko
#### 모델 json 파일 저장
##### https://www.tensorflow.org/js/tutorials/conversion/import_keras?hl=ko
#### model.summary
##### https://afsdzvcx123.tistory.com/entry/%ED%85%90%EC%84%9C%ED%94%8C%EB%A1%9C%EC%9A%B0-%EB%AA%A8%EB%8D%B8-summary-param-%EC%9D%98%EB%AF%B8
#### 카카오MAP API 생성 하고 갖고오기
##### https://apis.map.kakao.com/
---

## 테스트 환경

##### 웹페이지에서는 해당 폴더의 루트(FOODING_WEB)에서 ```python -m http.server 5500```명령어를 통해 로컬 호스트 서버를 생성한다. 그리고 루트폴더에서 /pages/login/login.html파일부터 실행하면 됩니다.
##### 카카오 맵 API의 데이터 같은 경우 ```localhost:5500```의 IP를 제외하고는 값이 불러와지지 않으니, 주의해서 구동하면 된다.

---

## 영상 유튜브 링크


---

## 참가자 코멘트
