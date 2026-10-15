-- 자동 생성 파일 (build_seed.ps1). 직접 고치지 말고 knowledge/nooki_prd.md 를 고친 뒤 다시 생성하세요.
truncate public.nooki_chunks, public.nooki_documents restart identity;  -- 기존 지식을 비우고 id 를 1부터 다시 매김

with d as (
  insert into public.nooki_documents (title) values ('Nooki 소개와 해결하려는 문제') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
Nooki는 작은 물건을 찾아 주는 실내 수색 로봇이다. 정식 과제명은 '비전 기반 실내 분실물 검출 무인이동체'이다.

가구나 물품이 많은 공간에서 에어팟이나 액세서리 같은 작은 물체를 떨어뜨리면 어디로 들어갔는지 확인하기 어렵다. 떨어뜨린 물품을 찾으려면 바닥에 엎드려 손전등 등으로 빛을 비추며 찾아야 하는 경우가 많고, 이 과정에서 불편한 자세를 취하고 청결하지 않은 바닥과 직접 접촉하게 된다. 물품이 가구 밑으로 들어가지 않았더라도 작은 물품을 눈으로 찾는 일은 시간이 오래 걸린다.

Nooki의 기대 효과(ROI): 바닥에 엎드려 손전등을 비추며 오래 찾던 작은 물건을 로봇이 3분 안에 찾아 주어, 사용자의 시간·불편·위생 부담을 없애고 분실로 인한 재구매 비용을 줄인다.
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('사용자 시나리오와 동작 흐름') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
사용자 시나리오는 다음 순서로 진행된다. 1) 사용자가 물품을 떨어뜨린다. 2) Nooki의 전원을 켠다. 3) 탐색하려는 위치 주위에 Nooki를 놓는다. 4) 찾을 물품을 음성이나 대시보드로 입력한 뒤 버튼이나 음성으로 탐색을 시작한다. 5) Nooki가 탐색을 시작하며, 어두운 공간으로 들어가 비전 사용이 어려우면 플래시(LED)를 켠다. 6) 지정한 물품을 검출한다.

내부 동작 흐름은 다음과 같다. 음성 또는 웹 대시보드로 찾을 물품을 입력받고 탐색을 시작한다. SLAM으로 주변 지도를 만들고, Nav2가 커버리지(지그재그) 경로를 따라 이동한다. 이동 중 카메라 노출 시간과 게인 값으로 밝기를 살피고, 어두우면 LED를 켠다. YOLO가 매 프레임에서 대상 물품을 찾고, 여러 프레임에서 연속으로 확인되면 검출을 확정한다. 검출하면 멈추고 알림을 보내며, 대시보드에 검출 사진과 지도 위 위치를 표시한다.
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('성능 목표와 성공지표(KPI)') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
Nooki의 성능 목표와 성공 기준은 네 가지 지표로 나뉜다.

성능 지표(제품이 기능을 제대로 해내는가): 물건 인식 정확도 90% 이상, 탐색 성공률 85% 이상, 방 하나 탐색 시간 3분 이내.

사용자 지표(사람들이 실제로 쓰고 만족하는가): 주간 사용 횟수, 재사용률(30일 리텐션), 만족도(NPS·별점).

비즈니스 지표(사업 목표에 기여하는가): 판매 대수, 대당 원가, 반품률, 고객 1명당 지원 비용.

품질·안전 지표(문제없이 안정적으로 동작하는가): 충돌·낙하 사고 0건, 고장률(MTBF), 배터리 1회 충전 사용 시간.
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('하드웨어 구성') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
Nooki는 모든 처리를 로봇 안(엣지)에서 한다. 집 내부 영상을 외부로 보내지 않기 위해서다.

차체: 목표 높이 5.4cm의 저상형 차동구동(2륜) 베이스. 소파·침대 밑 틈으로 들어가야 하기 때문이다.

연산 장치: Raspberry Pi 5 (8GB)와 AI HAT+ 13 TOPS(Hailo-8L), 액티브 쿨러. YOLO는 NPU에서 실시간(10fps 이상)으로 돌리고, 전처리는 CPU와 카메라 ISP에서 처리한다.

카메라: Camera Module 3 Wide(수평 화각 102°) 1대를 바닥 가까이 장착한다. 바닥의 작은 물체를 가까이에서 크게 찍기 위해서다.

거리 센서: 2D LiDAR(예: RPLIDAR 계열). SLAM 지도 작성과 장애물 회피에 쓴다.

보조 센서: 휠 엔코더, IMU, 범퍼, 낙하 방지(cliff) 센서. 주행 거리 추정과 충돌·낙하 사고 방지에 쓴다.

조명: 카메라 방향의 PWM 밝기 조절 LED(RP1 하드웨어 PWM 20kHz 이상). 가구 밑 어두운 공간에서 시야를 확보한다.

입출력: 마이크, 스피커, 물리 버튼. 음성 입력, 시작 트리거, 검출 알림에 쓴다.
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('하드웨어 두 보드 구조와 부품 연결') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
Nooki 하드웨어는 부품 19종으로 구성된다(하드웨어 부품 정리 보고서, 2026.10.14). 두뇌가 둘이다. 느려도 되는 AI 판단은 Raspberry Pi 5가, 늦으면 위험한 주행·충돌·낙하 대응은 ESP32-S3가 맡는다. PRD에는 MCU가 없어서 보고서가 ESP32-S3 추가를 제안했다.

Raspberry Pi 5는 YOLO 인식, SLAM 지도, Nav2 경로 계획, 음성 인식을 맡고 ESP32-S3에는 '이 속도로 가라'만 보낸다. 두 보드는 USB 시리얼(micro-ROS, 921600bps)로 연결한다.

Raspberry Pi 5에 연결되는 부품: AI HAT+ 13 TOPS(Hailo-8L, PCIe), 액티브 쿨러, Camera Module 3 Wide(CSI-2), 2D LiDAR RPLIDAR C1(UART를 USB 어댑터로), 조명 LED 2개와 MOSFET 드라이버(PWM), 마이크 INMP441(I2S), 스피커 앰프 MAX98357A와 3W 스피커(I2S), 시작 버튼(GPIO 5).

ESP32-S3에 연결되는 부품: 모터 내장 휠 엔코더 2개(PCNT), IMU MPU6050(I2C 주소 0x68), 범퍼 마이크로스위치 2개(GPIO 풀업), 낙하 방지 IR 센서 TCRT5000 4개(ADC1), 모터 드라이버 TB6612FNG와 N20 6V 기어모터 2개.

ESP32-S3는 범퍼나 낙하 센서가 반응하면 Pi 5의 명령을 기다리지 않고 즉시 멈춘다. Pi 5와 통신이 0.5초 끊겨도 모터 드라이버 STBY 핀을 내려 스스로 정지한다.

N20 모터는 차체 높이 5.4cm 안에 들어가는 직경 12mm급 소형 모터라서 골랐다. 모터, 배터리, LiDAR, IMU, 낙하 센서, 오디오 부품의 모델은 PRD에 없어서 보고서가 제안한 것이며 아직 확정되지 않았다.
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('전원 설계와 배터리 사용 시간') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
Nooki는 18650 배터리 2S2P(7.4V, 6000mAh)와 2S BMS를 쓴다. 배터리 뒤에 8A 퓨즈와 전원 스위치를 둔다. 평균 소비 전력은 약 18W, 피크는 약 43W다(추정치).

1회 충전 사용 시간은 약 100분이다. 7.4V × 6Ah = 44Wh에서 여유율 30%를 빼면 31Wh이고, 31Wh ÷ 18W ≈ 100분이다. 방 하나 탐색 3분 목표 기준으로 약 30회 탐색할 수 있다.

전원은 배터리에서 세 갈래로 나눈다. 5V-A(Buck 5A)는 Pi 5, AI HAT+, 쿨러, 카메라, LiDAR에 쓰고 피크 사용률은 88%다. 5V-B(Buck 3A)는 ESP32-S3, 센서, LED, 스피커 앰프, 마이크에 쓰고 피크 사용률은 57%다. 모터는 배터리 7.4V를 TB6612 드라이버로 직접 받고 피크 사용률은 19%다.

모터 전원을 따로 떼어내는 이유는 모터가 출발하는 순간 전압이 떨어져 Pi 5와 ESP32가 재부팅되는 브라운아웃을 막기 위해서다. 모든 GND를 한 점에서 묶고 TB6612 VM 단에 470µF 커패시터를 단다.

배터리는 만충 시 8.4V라 6V 모터에 그대로 가면 과전압이다. 그래서 PWM 듀티 상한을 80%로 묶는다. Buck 컨버터는 USB-C PD 협상을 하지 않으므로 Pi 5 config.txt에 usb_max_current_enable=1을 넣어야 LiDAR와 ESP32가 안정적으로 동작한다.
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('통신 구간과 핀 배정') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
Nooki는 로봇 안의 모든 데이터를 선으로 주고받고, 무선은 Pi 5와 웹 대시보드 사이의 Wi-Fi(rosbridge) 한 구간에서만 쓴다. 집 안 영상을 밖으로 내보내지 않는다는 원칙 때문이다.

통신 구간: 카메라 → Pi 5는 MIPI CSI-2(1280×720, 10fps 이상). Pi 5 ↔ AI HAT+는 PCIe Gen3 x1(약 8Gbps)로 YOLO를 NPU에서 10fps 이상 돌린다. LiDAR → Pi 5는 UART 460800bps를 USB 어댑터로 받는다. ESP32-S3 ↔ Pi 5는 USB-CDC 시리얼(micro-ROS, 921600bps)로 무선보다 끊김이 없고 지연이 일정하다. IMU → ESP32-S3는 I2C 400kHz. 마이크·스피커 ↔ Pi 5는 I2S이며, Pi 5에는 아날로그 오디오 단자가 없기 때문이다.

ESP32-S3 핀 배정: 낙하 IR 센서 GPIO 1·2·3·4(ADC1, Wi-Fi를 켜도 읽힘), 범퍼 좌우 GPIO 5·6(풀업, 20ms 디바운스), IMU SDA·SCL GPIO 8·9, 엔코더 왼쪽 GPIO 11·12와 오른쪽 GPIO 13·14(PCNT), 왼쪽 모터 PWMA·AIN1·AIN2 GPIO 15·16·17, 오른쪽 모터 PWMB·BIN1·BIN2 GPIO 38·39·40, 비상 정지 STBY GPIO 21, Pi 5 통신 USB GPIO 19·20. GPIO 0·45·46과 35~37은 쓰지 않는다.

Pi 5 핀 배정: LED PWM GPIO 12·13, I2S GPIO 18·19·20·21, 시작 버튼 GPIO 5. Pi 5와 ESP32-S3는 둘 다 3.3V 로직이라 레벨 시프터가 필요 없다. LED 1개는 약 0.3A라 GPIO가 직접 못 켜므로 MOSFET 게이트만 구동한다.
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('하드웨어 설계 리스크와 남은 결정') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
가장 큰 리스크는 높이 5.4cm 안에 Pi 5, AI HAT+, 쿨러를 쌓는 것이다. 적층 높이를 실측하고, 모자라면 Pi를 눕히거나 저상형 방열판으로 바꾼다.

다른 리스크와 대책: 모터 기동 시 재부팅은 모터 라인 분리, 공통 GND, 470µF 커패시터, 가속 램프업으로 막는다. LiDAR가 간헐적으로 끊기면 Pi 5 USB 전류 제한을 푼다. 발열로 YOLO가 10fps 아래로 떨어지면 쿨러를 필수로 달고 탐색 중 음성 인식을 멈춘다. 낙하 센서가 검은 바닥이나 유광 타일을 낭떠러지로 오인하면 바닥 재질별 ADC 기준값을 보정한다. 유광 타일에 비친 LED가 흰 이어버드처럼 보이면 LED 밝기 상한, 확산판, 각도로 줄인다. Pi와 통신이 끊겨도 바퀴가 계속 도는 폭주는 ESP32의 0.5초 정지 규칙으로 막는다.

부품을 주문하기 전에 정할 것: N20 모터의 기어비와 엔코더 분해능(목표 주행 속도 0.2m/s), RPLIDAR C1이 차체 높이에 들어가는지, 18650 4개가 안 들어가면 저상형 리포 2S로 바꿀지, ESP32-S3 추가를 PRD에 반영할지. 시제품으로 라인별 전류를 실측해 추정치를 교체한다.
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('소프트웨어 구성') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
플랫폼: ROS 2 Jazzy를 사용하며 Raspberry Pi OS에서는 Docker로 실행한다. 인식·이동·인터페이스를 각각 노드로 분리해 연결한다.

객체 인식: YOLO 경량 모델(v8n, 11n 등)을 에어팟·반지·귀걸이 등 대상 물품으로 파인튜닝한다. ONNX로 내보낸 뒤 Hailo DFC로 INT8 HEF 파일로 컴파일해 NPU에서 가속한다.

작은 물체 대응: 고해상도 입력과 타일링 추론을 쓴다. 이미지를 여러 조각으로 나눠 추론해 작은 물체가 축소 과정에서 사라지지 않게 한다.

검출 확정: 다중 프레임 검증을 한다. 연속 N프레임 이상 신뢰도가 기준을 넘으면 검출로 확정해 오검출을 줄인다.

밝기 인식: Picamera2와 OpenCV를 쓴다. 카메라 메타데이터(노출 시간 × 게인)로 어두움을 판단해 LED를 켜고, ISP 노이즈 제거와 CLAHE 대비 보정을 적용한다.

지도 작성: slam_toolbox 기반 SLAM으로 2D LiDAR를 이용해 탐색 공간 지도를 만들며 위치를 추정한다.

자율 이동: Nav2로 경로 계획과 장애물 회피를 하고, 지그재그(커버리지) 경로로 바닥 전체를 빠짐없이 훑는다.

검출 위치 표시: TF 좌표 변환으로 카메라 기준 검출 위치를 지도 좌표로 바꿔 표시한다.

음성 입력: whisper.cpp 경량 모델 같은 오프라인 STT를 쓴다. 탐색 중에는 STT를 멈춰 CPU를 확보한다. "에어팟 찾아줘" 같은 말을 대상 물품 클래스로 매핑한다.

대시보드: rosbridge와 연동한 웹 대시보드에서 물품을 선택하고 탐색을 시작하며, 탐색 진행 상황과 검출 사진·위치를 보여 준다.
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('추론 백엔드 속도 비교 결과') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
2026.10.15에 YOLO26n 모델을 CPU 추론 런타임 4가지(ONNX Runtime, NCNN, OpenVINO, TFLite(LiteRT))로 돌려 속도를 비교했다. 측정 코드는 저장소의 code/raspberrypi/benchmark.py이고, 결과는 code/raspberrypi/runs/compare_docker_amd64/에 있다.

측정 조건: 입력 640×640, FP32, 4스레드, 웜업 100회 뒤 300회 측정. 라즈베리파이 대신 x86 노트북(Ryzen 9 5900HS)의 Docker 컨테이너를 4코어로 제한해 측정했다. 전처리·추론·후처리를 따로 쟀다. 측정 전에 검출 결과를 ONNX와 비교하는 정확도 확인(parity)을 했고 4개 모두 통과했다. NCNN은 YOLO26의 NMS 없는 헤드를 지원하지 않아 일반 헤드로 바뀌고 NMS를 따로 돌린다.

x86 결과(전체 처리 시간 p50 / p95, 단위 ms): TFLite 31.9 / 36.9 (31.3 FPS), OpenVINO 33.0 / 39.6 (30.3 FPS), ONNX Runtime 39.0 / 50.5 (25.7 FPS), NCNN 70.8 / 80.8 (14.1 FPS). x86에서는 TFLite가 가장 빨랐다.

Raspberry Pi 5에서는 순위가 다르다. Ultralytics가 공개한 Pi 5 측정(YOLO26n, FP32, 640, 추론 시간만)은 NCNN 67.0ms, OpenVINO 104.6ms, TFLite 123.3ms, ONNX 126.0ms다. NCNN은 ARM용으로 최적화된 런타임이라 Pi에서 가장 빠르다. 그래서 x86 측정값으로 Pi의 순위를 정하면 안 된다.

지표 읽는 법: 300번 잰 시간을 빠른 순서로 세웠을 때 pXX는 XX% 지점의 값이다. p50(중앙값)은 보통 걸리는 시간, p95는 20번 중 1번은 이보다 느리다는 뜻으로 실시간 처리 가능 여부를 판단할 때 쓴다. max는 가장 느렸던 한 번이라 참고용이다. p95는 95번째 값이 아니라 95% 지점의 값이다(300회면 285번째).
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('추론 장치 선택: Hailo 사용 시와 Pi 5 단독 시') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
Nooki의 YOLO 목표는 10fps 이상, 즉 프레임당 100ms다. 작은 물체를 찾으려 타일링하면 조각 수만큼 시간을 나눠 써야 한다(4조각이면 1회 25ms). 같은 Pi 5 CPU(4코어)에서 SLAM, Nav2, 음성 인식도 돈다.

시나리오 A, AI HAT+(Hailo-8L)를 쓸 수 있을 때: YOLO는 Hailo-8L NPU에서 돌리는 것을 권장한다. PRD의 원래 설계와 같다. NPU가 추론을 맡아 CPU를 SLAM·Nav2에 쓸 수 있고, 타일링 여유도 생긴다. Hailo-8L은 YOLO26 검출을 지원하며 변환 경로는 .pt → ONNX → INT8 양자화 → HEF이고 INT8만 지원한다. ONNX Runtime은 Hailo 변환의 입력이자 INT8 정확도 비교 기준(FP32)으로 쓰고, NCNN은 HAT 고장이나 HAT 없이 개발할 때의 CPU 예비로 쓴다.

시나리오 B, Raspberry Pi 5만 쓸 수 있을 때: NCNN을 권장한다. 다만 추론만 67ms라 PRD 그대로 10fps에 타일링까지 하는 것은 사실상 불가능하다. 대응책은 효과가 큰 순서로 다음과 같다. 1) NCNN FP16 연산 사용(Pi 5 CPU가 지원). 2) 목표를 5fps(프레임당 200ms)로 조정하고, 여러 프레임 연속 확인 기준을 시간 기준으로 바꾼다. 3) 화면 전체를 타일링하지 않고 바닥이 보이는 아래쪽 영역만 원본 해상도로 잘라 1회 추론한다. 4) 스레드를 3개로 제한해 코어 1개를 SLAM·Nav2용으로 남긴다. 5) 최후 수단으로 입력을 640에서 416으로 줄인다(계산량 약 40%, 작은 물체를 놓칠 수 있음).

다음에 확인할 것: Pi 5에서 4개 런타임을 FP32와 FP16으로 직접 측정한다. Hailo를 같은 방식으로 측정하고 INT8 양자화 후 정확도 하락을 확인한다(목표 인식 정확도 90% 이상, 양자화 보정에는 Nooki 카메라로 찍은 실내 바닥 이미지 1,024장 이상 권장). SLAM·Nav2를 켠 상태에서 p95를 다시 잰다.
$body$, 500, 100) c;

with d as (
  insert into public.nooki_documents (title) values ('범위와 비범위(MVP)') returning id
)
insert into public.nooki_chunks (document_id, chunk_index, content)
select d.id, c.chunk_index, c.content from d, public.split_into_chunks($body$
초기 버전(MVP)의 목표는 "한 방의 바닥에서 지정한 작은 물품을 찾아 위치를 알려 주는 것"까지다.

이번에 하는 것(범위): 평평한 실내 바닥에서 한 번에 방 하나 단위로 탐색한다. 사전 학습한 물품 목록(에어팟·무선 이어폰, 반지, 귀걸이, 열쇠, 리모컨 등)을 검출한다. 소파·침대처럼 로봇 높이보다 틈이 넓은 가구 밑으로 들어간다. 어두운 공간에서 LED를 자동으로 켠다. 음성이나 대시보드로 물품을 입력하고 탐색을 시작한다. 검출하면 알림과 함께 사진과 지도 위 위치를 제공한다. 충돌을 피하고 낙하를 방지한다.

이번에 하지 않는 것(비범위): 물품을 집거나 꺼내 오는 기능(로봇팔·그리퍼)은 없다. 학습하지 않은 임의 물품 검출, 즉 텍스트로 아무 물건이나 찾는 기능은 추후 확장 과제다. 여러 방이나 층을 오가는 집 전체 탐색, 계단·문턱 넘기는 하지 않는다. 두꺼운 러그나 놀이 매트처럼 고르지 않은 바닥과 실외는 지원하지 않는다. 가방 안이나 쿠션 틈처럼 물체에 가려 보이지 않는 물품은 찾지 않는다. 청소 등 다른 기능과 영상의 클라우드 전송·저장도 하지 않는다.
$body$, 500, 100) c;

