# FOODING

---

![단체사진](https://user-images.githubusercontent.com/96258104/175428073-5d2d214f-929c-4631-ac37-0192f90bceb3.png)

---

## 팀원 소개

### 개발자
- #### 김수진(배재대학교)
- #### 심상현(숭실대학교)
- #### 이승용(동서대학교)
- #### 정은성(인제대학교)
### 디자이너
- #### 김세은(인제대학교)

---

## 프로젝트 배경/목표

#### 배달비와 배달음식 비용이 점점 급증하고 있어 혼자나 소수에서 시켜먹기 부담스러워지고 있습니다.
#### 채팅으로 배달 메이트를 매칭시켜 배달비와 배달음식 비용의 부담을 덜어주는 웹페이지 제작을 하게 되었습니다.
#### '뭘 먹을까?'에 대한 고민을 줄여주기 위하여 나이대, 현재 기분, 현재 날씨, 시켜먹는 시간대 등을 선택하면
#### 달달한 맛, 짭짤한 맛, 매콤한 맛을 추천해주는 기능도 포함하려고 합니다.

--- 

## 동작 방식(프로세스)
#### 로그인 환경에서 firebase와의 통신을 통해 회원가입 / 로그인 / 아이디찾기 / 비밀번호 재설정 / 내 정보 변경과 같은 동작을 구현할 수 있다.
#### 또한 채팅방을 만들고 접속하는 과정을 실행할 수 있다.
#### 내 인터넷 접속 위치를 갖고 내 주변 음식점과 내가 원하는 위치 주변의 음식점을 확인할 수 있다.
#### 여러가지 변수(나이대, 성별, 시간대)를 통해 사용자가 원하는 종류의 맛을 추천하고, 관련된 음식 또한 추천할 수 있다.

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

#### 웹페이지에서는 해당 폴더의 루트(FOODING_WEB)에서 ```python -m http.server 5500```명령어를 통해 로컬 호스트 서버를 생성한다. 그리고 루트폴더에서 /pages/login/login.html파일부터 실행하면 됩니다.
#### 카카오 맵 API의 데이터 같은 경우 ```localhost:5500```의 IP를 제외하고는 값이 불러와지지 않으니, 주의해서 구동하면 된다.

---

## 영상 유튜브 링크

### https://www.youtube.com/watch?v=2cHym-uvGBQ

---

## 참가자 코멘트

#### 김세은 - 처음 참가하는 해커톤이어서 걱정을 많이 했지만 팀원분들과 원활한 소통으로 잘 마무리한 것 같습니다. 하나의 프로젝트를 완성하면서 촉박한 시간 안에 자신의 분야에서 최선을 다하는 모습들에 본받을 점들도 많았고 저, 스스로도 성장해나가는 모습에 낯설지만 재미있었습니다. 보기만 했던 툴을 직접 사용하느라 여러모로 힘들었지만 최선을 다했고 팀원분들이 도와주셔서 미흡하지만 마무리를 할 수 있었습니다. 모두의 노력으로 좋은 결과가 나온 것 같아 다행이고 3일 동안 수고하셨습니다!
#### 심상현 - 이번 해커톤을 통해 빠른 시간 안에 서로가 협업해서 하나의 서비스를 만들어내는 경험을 할 수 있던 뜻깊은 시간이었고, 이렇게 전혀 모르던 사람과 단시간에 친하게 되고, 활동을 진행하는게 너무나 재밌었던거 같아요! 이런 자리가 또 있으면 좋겠습니다!
#### 김수진 - 비대면으로 진행되는 해커톤이라서 소통이 원할하게 될지 걱정을 많이 했었지만 팀원들 모두 열심히 참여하고 협업해 좋은 경험이 된 것 같습니다.
#### 이승용 - 다른 사람들과 협업한 경험 없이 큰 규모의 해커톤에 참여하게 되었습니다. 사람들과 함께 개발하여 도움이 되기 위해서는 제가 더 많은 공부를 필요로 하게 된다는 점을 느꼈습니다. 많이 부족한 저를 끝까지 낙오시키지 않고 함께 할 수 있도록 이끌어준 팀원분들께 감사드립니다.
#### 정은성 - 웹페이지 제작을 git을 이용하여 협업을 처음 해봤고, 해커톤도 처음 접했습니다. 협업하면서 부족한 점이 많았지만 같은 팀원들끼리 협동하여 좋은 결과물을 낼 수 있었습니다. 열심히 따라와준 팀원분들께 감사하고 좋은 경험이 되어 좋았습니다.
