# MQTT 실습 (브로커 / 구독자 / 발행자 컨테이너 3개)

```
publisher ──publish(sensor/temperature)──▶ broker(mosquitto:1883) ──deliver(sensor/#)──▶ subscriber
```

| 파일 | 역할 |
|---|---|
| `mosquitto.conf` | 브로커 설정 (1883 포트, 익명 허용 — 실습용) |
| `subscriber.py` | `sensor/#` 구독, 받은 메시지 출력 |
| `publisher.py` | `sensor/temperature`에 1초 간격 10개 발행 |
| `Dockerfile` | 구독자/발행자 공용 이미지 (python + paho-mqtt) |
| `docker-compose.yml` | 컨테이너 3개 정의 |

## 실행 순서

```bash
cd code/mqtt

# 1. 이미지 빌드
docker compose build

# 2. 브로커 실행
docker compose up -d broker
docker compose logs broker          # "mosquitto version ... running" 확인

# 3. 구독자 실행 + 로그 보기 (터미널 A, 계속 켜두기)
docker compose up -d subscriber
docker compose logs -f subscriber   # "[SUB] 구독 시작: sensor/#" 확인

# 4. 발행자 실행 (터미널 B)
docker compose run --rm publisher   # 터미널 A에 [SUB] 수신 ... 이 찍힘

# 5. 정리
docker compose --profile pub down
```

## 추가로 해볼 것

```bash
# 브로커 컨테이너 안의 CLI로 직접 발행
docker exec mqtt-broker mosquitto_pub -t sensor/humidity -m '{"hum": 55}'

# retain: 구독자가 나중에 접속해도 마지막 메시지를 받음
docker exec mqtt-broker mosquitto_pub -t sensor/door -m open -r
docker compose restart subscriber   # 재시작 직후 retain=True 로 수신

# 토픽 필터 차이: '+'는 한 단계, '#'은 하위 전부
docker exec mqtt-broker mosquitto_sub -t 'sensor/+' -v
```
