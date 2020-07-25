### 1. 환경구성
- conda create -n link_rl_book_codes python=3.7
- conda activate link_rl_book_codes
- git clone
- cd ~/git/link_rl_book_codes/

### 2. 모든 의존성 다운 받기

- pip install -r requirements.txt --upgrade

### 3. 추가 Package 설치
- pip install gym[atari]

### 4. 한글 폰트 설치 참고 사이트
- https://programmers.co.kr/learn/courses/21/lessons/950

### 5. RL 구현 참고 자료
- https://towardsdatascience.com/deep-reinforcement-learning-build-a-deep-q-network-dqn-to-play-cartpole-with-tensorflow-2-and-gym-8e105744b998
- http://inoryy.com/post/tensorflow2-deep-reinforcement-learning/
- https://blog.naver.com/PostView.nhn?blogId=msnayana&logNo=221431225117&categoryNo=182&parentCategoryNo=0&viewDate=&currentPage=1&postListTopCurrentPage=1&from=postView
- Pong-v0: https://towardsdatascience.com/intro-to-reinforcement-learning-pong-92a94aa0f84d#9e3b

### 6. 통계 그래프 보는 방법
- python -m visdom.server 수행 
- 브라우저에서 http://localhost:8097 접속 