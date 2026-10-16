// Nooki MCU · SBC 분석 보고서 (report/ 기존 데이터시트 스타일과 같은 색·글꼴)
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  BorderStyle, AlignmentType, HeadingLevel, LevelFormat, Header, Footer, PageNumber,
  TableOfContents, PageBreak, LineRuleType,
} = require("docx");

const INK = "1C2733", SLATE = "5B6B7B", RULE = "CBD3DC", FILL = "F2F5F8", POW = "D9622B", DATA = "2B6CB0", OK = "2F855A";
const FONT = "Malgun Gothic";
const W = 9638; // A4 - 20mm*2 (dxa)

const run = (t, o = {}) => new TextRun({ text: t, font: FONT, color: o.color || INK, bold: o.bold, size: o.size ? o.size * 2 : undefined });
const P = (t, o = {}) => new Paragraph({
  children: Array.isArray(t) ? t : [run(t, o)],
  spacing: { before: (o.before || 0) * 20, after: (o.after ?? 7) * 20, line: Math.round((o.line || 1.15) * 240), lineRule: LineRuleType.AUTO },
  alignment: o.align, keepNext: o.keepNext,
});
const H2 = (t, pageBreak) => new Paragraph({ heading: HeadingLevel.HEADING_2, pageBreakBefore: pageBreak, children: [run(t)] });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [run(t)] });
const B = (t) => new Paragraph({ numbering: { reference: "dot", level: 0 }, spacing: { after: 80, line: 276 },
  children: Array.isArray(t) ? t : [run(t)] });
const note = (t) => P(t, { color: SLATE, size: 8.5, after: 12 });

const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(head, rows, widths, o = {}) {
  const size = o.size || 8.5;
  const cell = (txt, ci, ri) => {
    const isHead = ri < 0;
    const hl = !isHead && o.highlight && o.highlight(ri, ci);
    const lines = String(txt).split("\n");
    return new TableCell({
      width: { size: widths[ci], type: WidthType.DXA },
      shading: isHead ? { type: ShadingType.CLEAR, fill: INK, color: "auto" }
        : hl ? { type: ShadingType.CLEAR, fill: "E6F4EA", color: "auto" }
        : ri % 2 === 1 ? { type: ShadingType.CLEAR, fill: FILL, color: "auto" } : undefined,
      margins: { top: 55, bottom: 55, left: 85, right: 85 },
      borders: { top: none, left: none, right: none, bottom: { style: BorderStyle.SINGLE, size: isHead ? 8 : 4, color: isHead ? INK : RULE } },
      children: lines.map((l) => new Paragraph({
        alignment: (o.align || {})[ci],
        spacing: { before: 0, after: 0, line: 252 },
        children: [run(l, { size, bold: isHead || (hl && ci === 0), color: isHead ? "FFFFFF" : INK })],
      })),
    });
  };
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    borders: { top: none, left: none, right: none, insideVertical: none, bottom: { style: BorderStyle.SINGLE, size: 8, color: INK }, insideHorizontal: { style: BorderStyle.SINGLE, size: 4, color: RULE } },
    rows: [
      new TableRow({ tableHeader: true, cantSplit: true, children: head.map((h, ci) => cell(h, ci, -1)) }),
      ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((c, ci) => cell(c, ci, ri)) })),
    ],
  });
}
const gap = () => P("", { after: 4 });
const C = AlignmentType.CENTER, R = AlignmentType.RIGHT;

// ---------- 점수 계산 (가중치 × 1~5점 → 100점 만점) ----------
const score = (w, s) => Math.round(s.reduce((a, v, i) => a + v * w[i], 0) / 5);

const SBC_W = [25, 15, 15, 10, 10, 10, 10, 5];
const SBC_CRIT = ["AI 추론", "CPU 여유", "카메라 스택", "ROS 2·Docker", "높이·크기", "전원", "가격", "자료·경험"];
const SBC = [
  ["A. Pi 5 8GB + AI HAT+ 13T", [4, 4, 5, 4, 3, 4, 3, 5]],
  ["B. Pi 5 8GB + AI HAT+ 26T", [5, 4, 5, 4, 3, 4, 2, 5]],
  ["C. Pi 5 8GB 단독 (CPU)", [2, 2, 5, 4, 4, 5, 4, 5]],
  ["D. Jetson Orin Nano Super", [5, 4, 2, 4, 2, 2, 1, 4]],
  ["E. RK3588 보드", [3, 5, 2, 3, 3, 3, 3, 2]],
];
const MCU_W = [20, 15, 15, 10, 15, 10, 5, 10];
const MCU_CRIT = ["micro-ROS", "엔코더 HW", "ADC·GPIO", "USB CDC", "실시간·안전", "크기", "가격", "개발 난이도"];
const MCU = [
  ["ESP32-S3 DevKitC-1", [5, 5, 4, 5, 4, 3, 5, 5]],
  ["Teensy 4.1", [5, 5, 5, 5, 5, 4, 2, 4]],
  ["STM32F411 (Black Pill)", [3, 5, 5, 4, 5, 4, 5, 2]],
  ["RP2350 (Pico 2)", [2, 5, 2, 5, 4, 5, 5, 4]],
];
const sbcScores = SBC.map(([n, s]) => [n, ...s, score(SBC_W, s)]);
const mcuScores = MCU.map(([n, s]) => [n, ...s, score(MCU_W, s)]);
const best = (rows) => Math.max(...rows.map((r) => r[r.length - 1]));

// ---------- 본문 ----------
const cover = [
  P("비전 기반 실내 분실물 검출 무인이동체", { color: SLATE, size: 10.5, before: 40, after: 8 }),
  P("Nooki MCU · SBC", { size: 30, bold: true, after: 0, line: 1.25 }),
  P("분석 보고서", { size: 30, bold: true, after: 14, line: 1.25 }),
  P("Nooki의 요구사항(작은 물체 실시간 인식, 저상형 차체, 배터리 구동, 충돌·낙하 안전)으로 SBC 5개 안과 MCU 4종을 비교했다. 결론은 Raspberry Pi 5 8GB + AI HAT+(Hailo NPU)와 ESP32-S3 조합을 유지하는 것이다.",
    { color: SLATE, size: 11, after: 24, line: 1.3 }),
  table(["권장 SBC", "권장 MCU", "YOLO 목표", "평균 전력", "차체 높이"],
    [["Pi 5 + Hailo", "ESP32-S3", "≥10fps", "≈18W", "5.4cm"]], [1928, 1928, 1928, 1927, 1927], { size: 14, align: { 0: C, 1: C, 2: C, 3: C, 4: C } }),
  P("", { after: 20 }),
  P("목차", { size: 11, bold: true, after: 6 }),
  new TableOfContents("목차", { hyperlink: true, headingStyleRange: "2-2" }),
  P("작성  유정완 (2026년 10월 16일)", { color: SLATE, size: 9, before: 24, after: 2 }),
  P("근거 자료  Nooki PRD(2026.10.06), Nooki 하드웨어 부품 정리 보고서(2026.10.14), 추론 백엔드 비교(2026.10.15), code/nooki_sbc/DESIGN.md", { color: SLATE, size: 9 }),
];

const body = [
  H2("1. 요약", true),
  P("Nooki는 두 개의 두뇌를 쓴다. 느려도 되는 AI 판단(인식·지도·경로·음성)은 SBC가, 늦으면 위험한 주행·충돌·낙하 대응은 MCU가 맡는다. 이 보고서는 두 자리에 어떤 보드가 맞는지를 같은 기준으로 비교한다."),
  table(["구분", "권장", "예비안", "핵심 근거"], [
    ["SBC", "Raspberry Pi 5 8GB\n+ AI HAT+ (Hailo-8L 13 TOPS)", "AI HAT+ 26 TOPS (Hailo-8)\n→ 타일링 실측이 부족할 때", "Camera Module 3 노출·게인 메타데이터를 그대로 쓰는 유일한 카메라 스택, NPU가 추론을 맡아 CPU를 SLAM·Nav2에 남김, 자료·경험 최다"],
    ["MCU", "ESP32-S3 (DevKitC-1)", "Teensy 4.1\n→ 실시간·ADC 문제가 생길 때", "micro-ROS 공식 ESP-IDF 컴포넌트, 엔코더용 PCNT 4개, 네이티브 USB, 가장 싸고 10.14 보고서 핀 배정이 이미 이 보드 기준"],
  ], [900, 2500, 2400, 3838], { highlight: () => false }),
  gap(),
  B([run("SBC 점수: ", { bold: true }), run(sbcScores.map((r) => `${r[0].split(".")[0]} ${r[9]}`).join(" · ") + "점 (100점 만점). A와 B의 차이는 성능 여유와 가격 두 항목뿐이라, Nooki 모델로 4타일 추론을 실측한 뒤 13 TOPS로 충분한지 결정한다.")]),
  B([run("MCU 점수: ", { bold: true }), run(mcuScores.map((r) => `${r[0].split(" ")[0]} ${r[9]}`).join(" · ") + "점. Teensy 4.1이 근소하게 높지만 가격이 2배 이상이고, ESP32-S3로 요구사항을 모두 채울 수 있어 ESP32-S3를 유지한다.")]),
  B([run("주의: ", { bold: true, color: POW }), run("2025년 말부터 메모리 가격 급등으로 Pi 5와 Jetson 가격이 여러 차례 올랐다. 이 보고서의 가격은 2026년 확인값이며 구매 전에 다시 확인해야 한다.")]),

  H2("2. 왜 두 보드인가"),
  P("PRD에는 MCU가 없었다. 10.14 하드웨어 보고서가 ESP32-S3 추가를 제안했고, 이 보고서도 같은 결론이다. Linux는 실시간 운영체제가 아니라서 SBC가 바쁘거나 멈추면 모터 정지가 늦어진다."),
  table(["", "SBC (Raspberry Pi 5)", "MCU (ESP32-S3)"], [
    ["맡는 일", "YOLO 인식, SLAM 지도, Nav2 경로, 음성 인식, 대시보드", "모터 PWM, 엔코더 계수, IMU, 범퍼·낙하 감지, 비상 정지"],
    ["허용 지연", "수십~수백 ms (프레임당 100ms)", "수 ms (범퍼 → 정지 즉시)"],
    ["운영체제", "Linux + Docker + ROS 2 Jazzy", "FreeRTOS + micro-ROS"],
    ["멈추면", "검출·주행 명령이 멈춤 → MCU가 0.5초 뒤 스스로 정지", "로봇 전체 정지 (STBY 핀)"],
    ["연결", "USB-CDC 시리얼 921600bps (micro-ROS)", "←"],
  ], [1300, 4169, 4169]),
  gap(),
  P("MCU 없이 Pi 5 GPIO로 모터와 엔코더를 직접 다루는 안은 제외했다. 엔코더 펄스를 Linux 사용자 공간에서 놓치기 쉽고, 안전 정지가 Pi의 부하와 컨테이너 상태에 묶이기 때문이다.", { color: SLATE }),

  H2("3. 평가 기준", true),
  H3("SBC 기준"),
  table(["기준", "가중치", "요구값", "근거"], [
    ["AI 추론", "25", "YOLO 640 입력, 4타일 × 10fps = 초당 40회 추론", "PRD 성능 목표, 작은 물체 타일링"],
    ["CPU 여유", "15", "추론 중에도 SLAM·Nav2·STT용 코어 확보", "같은 CPU에서 동시 실행"],
    ["카메라 스택", "15", "광각 CSI 카메라, 노출 시간·게인 메타데이터, ISP 노이즈 제거", "밝기 판단 → LED 제어"],
    ["ROS 2·Docker", "10", "ROS 2 Jazzy를 Docker로 실행", "소프트웨어 구성, 컨테이너 설계"],
    ["높이·크기", "10", "차체 높이 5.4cm 안에 보드·가속기·쿨러 적층", "가구 밑 진입"],
    ["전원", "10", "평균 18W 예산, 2S 배터리(7.4V)에서 Buck 5V", "100분 사용 목표"],
    ["가격", "10", "시제품 1~2대", "프로젝트 예산"],
    ["자료·경험", "5", "예제·커뮤니티·팀 경험", "개발 속도"],
  ], [1400, 800, 4238, 3200], { align: { 1: C } }),
  H3("MCU 기준"),
  table(["기준", "가중치", "요구값", "근거"], [
    ["micro-ROS", "20", "공식 또는 검증된 포팅", "Pi 5와 ROS 2 토픽으로 연결"],
    ["엔코더 HW", "15", "직교 엔코더 2채널 하드웨어 계수", "주행 거리 추정, 0.2m/s 주행"],
    ["ADC·GPIO", "15", "ADC 4채널(낙하 IR), GPIO 19개 이상", "10.14 핀 배정"],
    ["USB CDC", "10", "네이티브 USB 시리얼", "USB-UART 칩 없이 921600bps"],
    ["실시간·안전", "15", "범퍼·낙하 즉시 정지, 0.5초 통신 감시", "충돌·낙하 사고 0건"],
    ["크기", "10", "저상 차체에 눕혀 장착", "5.4cm 높이"],
    ["가격", "5", "", ""],
    ["개발 난이도", "10", "Arduino/ESP-IDF 수준의 자료", "펌웨어 담당 1명"],
  ], [1400, 800, 4238, 3200], { align: { 1: C } }),

  H2("4. SBC 후보 비교", true),
  H3("4.1 후보 사양"),
  table(["후보", "CPU", "AI 가속", "카메라", "전원 입력", "가격 (2026 확인)"], [
    ["A. Pi 5 8GB\n+ AI HAT+ 13T", "Cortex-A76 4코어 2.4GHz", "Hailo-8L 13 TOPS (PCIe)", "CSI 2개, Camera Module 3 공식", "5V 5A", "Pi 5 $125+ / HAT $70 (출시가)"],
    ["B. Pi 5 8GB\n+ AI HAT+ 26T", "같음", "Hailo-8 26 TOPS (PCIe)", "같음", "5V 5A", "Pi 5 $125+ / HAT $110 (출시가)"],
    ["C. Pi 5 8GB 단독", "같음", "없음 (NCNN CPU 추론)", "같음", "5V 5A", "$125+"],
    ["D. Jetson Orin Nano\nSuper 개발 키트", "Cortex-A78AE 6코어 1.7GHz", "Ampere GPU 67 TOPS", "CSI, IMX219·477 위주 (IMX708 비공식)", "DC 잭 (19V 어댑터)", "$399 (2026.07 인상)"],
    ["E. RK3588 보드\n(Orange Pi 5 등)", "A76 4 + A55 4코어", "NPU 6 TOPS (RKNN)", "CSI, 보드별 드라이버", "5V (USB-C)", "시세 변동"],
  ], [1650, 1600, 1650, 1900, 1200, 1638], { size: 8 }),
  note("Pi 5 8GB는 $80 → $95(2025.12) → $125(2026.02)로 올랐고, 2026.04에 추가 인상 보도가 있다. AI HAT+ 2(Hailo-10H, 40 TOPS INT4, 온보드 8GB, $130)도 있지만 LLM용이라 MVP에는 필요 없다."),
  H3("4.2 평가 점수 (1~5점, 가중 합계 100점)"),
  table(["후보", ...SBC_CRIT, "합계"],
    sbcScores.map((r) => r.map(String)),
    [1838, 900, 900, 900, 900, 900, 900, 900, 900, 600],
    { size: 8, align: { 1: C, 2: C, 3: C, 4: C, 5: C, 6: C, 7: C, 8: C, 9: C }, highlight: (ri) => sbcScores[ri][9] >= best(sbcScores) - 3 }),
  note("가중치(%): " + SBC_CRIT.map((c, i) => `${c} ${SBC_W[i]}`).join(" · ") + ". 녹색 = 최고점과 3점 이내."),
  H3("4.3 후보별 판단"),
  B([run("A·B. Pi 5 + AI HAT+ (권장). ", { bold: true }), run("YOLO는 NPU가, SLAM·Nav2·STT는 CPU 4코어가 나눠 맡는다. Camera Module 3의 노출·게인 메타데이터를 Picamera2로 바로 읽어 LED를 제어할 수 있는 것은 이 조합뿐이다. 13 TOPS와 26 TOPS의 차이는 타일링 여유다. Nooki 모델을 HEF로 컴파일해 4타일 × 10fps의 p95를 재고, 100ms를 넘으면 26 TOPS로 바꾼다. 같은 PCIe 슬롯·같은 소프트웨어라 교체 비용이 작다.")]),
  B([run("C. Pi 5 단독. ", { bold: true }), run("NCNN으로 YOLO26n 추론만 약 67ms(Ultralytics 공개 Pi 5 측정)라 10fps에 타일링까지는 불가능하다. 목표를 5fps로 낮추고 바닥 영역만 1회 추론하는 예비 모드로만 쓴다(추론 백엔드 비교 보고서 시나리오 B).")]),
  B([run("D. Jetson Orin Nano Super. ", { bold: true }), run("추론 성능은 가장 좋지만 Nooki에는 맞지 않는다. Camera Module 3(IMX708)이 공식 지원되지 않아 카메라와 밝기 판단 코드를 바꿔야 하고, 개발 키트는 19V 입력이라 2S 배터리에서 승압이 필요하며, 방열판 포함 높이가 5.4cm 차체에 불리하다. 2026.07 가격 인상으로 $399가 되어 가격 점수도 가장 낮다.")]),
  B([run("E. RK3588 보드. ", { bold: true }), run("8코어 CPU는 SLAM·Nav2에 유리하지만, NPU 변환(RKNN)과 카메라 드라이버가 보드마다 달라 자료가 적다. Picamera2를 쓸 수 없어 밝기 판단을 V4L2로 다시 만들어야 한다.")]),

  H2("5. MCU 후보 비교"),
  H3("5.1 후보 사양"),
  table(["후보", "코어", "엔코더 계수", "ADC", "USB", "micro-ROS", "가격"], [
    ["ESP32-S3 DevKitC-1", "Xtensa LX7 2코어 240MHz", "PCNT 4유닛", "ADC1 10채널\n(Wi-Fi와 무관)", "네이티브 USB", "micro_ros_espidf_component (micro-ROS 공식)", "$10~15"],
    ["Teensy 4.1", "Cortex-M7 600MHz", "HW 직교 디코더 4개", "12bit 다채널", "네이티브 USB", "micro_ros_arduino 지원 목록", "$30대"],
    ["STM32F411 (Black Pill)", "Cortex-M4F 100MHz", "타이머 엔코더 모드", "12bit 10채널", "USB FS (설정 필요)", "STM32CubeMX 유틸 (커뮤니티)", "$5~10"],
    ["RP2350 (Pico 2)", "Cortex-M33 2코어 150MHz", "PIO로 구현", "핀으로 3채널만 노출", "네이티브 USB", "RP2040용만 확인, RP2350 미확인", "$5"],
    ["ATmega328P (Nano)", "8bit 16MHz", "인터럽트 2개", "8채널", "USB-UART 칩", "불가 (RAM 2KB)", "제외"],
  ], [1650, 1500, 1350, 1300, 1150, 1800, 888], { size: 8 }),
  H3("5.2 평가 점수 (1~5점, 가중 합계 100점)"),
  table(["후보", ...MCU_CRIT, "합계"],
    mcuScores.map((r) => r.map(String)),
    [1838, 900, 900, 900, 900, 900, 900, 900, 900, 600],
    { size: 8, align: { 1: C, 2: C, 3: C, 4: C, 5: C, 6: C, 7: C, 8: C, 9: C }, highlight: (ri) => mcuScores[ri][9] >= best(mcuScores) - 3 }),
  note("가중치(%): " + MCU_CRIT.map((c, i) => `${c} ${MCU_W[i]}`).join(" · ") + ". 녹색 = 최고점과 3점 이내."),
  H3("5.3 후보별 판단"),
  B([run("ESP32-S3 (권장). ", { bold: true }), run("필요한 자원을 모두 하드웨어로 가지고 있고(6장), micro-ROS 공식 ESP-IDF 컴포넌트가 있다. 2코어라 micro-ROS 통신과 모터 제어 루프를 코어별로 나눌 수 있다. 약점은 ADC 비선형성인데, 낙하 센서는 바닥 재질별 기준값 비교만 하므로 문제가 작다.")]),
  B([run("Teensy 4.1 (예비). ", { bold: true }), run("실시간 성능과 ADC 품질이 가장 좋고 micro_ros_arduino가 지원한다. 가격이 2배 이상이라 ESP32-S3에서 제어 주기·엔코더 누락 문제가 실측될 때 바꾼다. Arduino 코드라 이식이 쉽다.")]),
  B([run("STM32F411. ", { bold: true }), run("하드웨어 자원은 충분하지만 CubeMX + FreeRTOS + micro-ROS 빌드 환경을 처음 갖추는 비용이 크다. 펌웨어 담당이 STM32 경험이 있으면 다시 검토한다.")]),
  B([run("RP2350. ", { bold: true }), run("PIO로 엔코더를 잘 세지만 Pico 2 보드에 ADC가 3개만 나와 낙하 센서 4개를 받으려면 외부 ADC가 필요하고, micro-ROS 지원이 확인되지 않았다.")]),

  H2("6. 권장 조합의 자원 점검"),
  H3("ESP32-S3: 필요한 것 vs 가진 것"),
  table(["자원", "필요", "ESP32-S3", "판정"], [
    ["GPIO", "19개 (10.14 핀 배정)", "DevKitC-1 노출 GPIO 다수\n(0·45·46·35~37, USB 19·20 제외)", "충분"],
    ["엔코더 계수", "2채널 직교", "PCNT 4유닛", "충분"],
    ["ADC", "4채널 (낙하 IR, GPIO 1~4)", "ADC1 10채널", "충분"],
    ["모터 PWM", "2채널 + 방향 4핀", "LEDC 8채널 / MCPWM 2유닛", "충분"],
    ["I2C", "1개 (IMU 400kHz)", "2개", "충분"],
    ["Pi 5 연결", "USB-CDC 921600bps", "네이티브 USB (GPIO 19·20)", "충분"],
    ["안전 정지", "범퍼·낙하 즉시, 통신 0.5초 감시", "GPIO 인터럽트 + FreeRTOS 타이머 → STBY(GPIO 21)", "충분"],
  ], [1600, 2600, 3838, 1600], { highlight: () => false }),
  H3("Pi 5 + AI HAT+: 실측으로 확인할 것"),
  table(["항목", "왜", "합격 기준"], [
    ["타일링 추론 속도", "4타일 × 10fps = 초당 40회", "Hailo에서 프레임당 p95 < 100ms (SLAM·Nav2 실행 중)"],
    ["INT8 정확도", "Hailo는 INT8만 지원", "ONNX FP32 대비 정확도 하락 2%p 이내, 인식 정확도 90% 이상"],
    ["적층 높이", "가장 큰 하드웨어 리스크", "Pi 5 + HAT + 쿨러가 차체 허용 높이 이내"],
    ["발열", "스로틀링 시 fps 하락", "10분 연속 탐색 후 vcgencmd get_throttled = 0x0"],
    ["전류", "5V-A 라인 피크 88% 추정", "라인 실측 피크 < 5A"],
  ], [2000, 3000, 4638]),

  H2("7. 결론과 다음 단계"),
  B([run("SBC는 Raspberry Pi 5 8GB + AI HAT+를 유지한다. ", { bold: true }), run("13 TOPS로 시작하고, 6장의 타일링 실측이 기준을 못 넘으면 26 TOPS로 바꾼다.")]),
  B([run("MCU는 ESP32-S3를 유지한다. ", { bold: true }), run("시제품에서 제어 주기와 엔코더 누락을 재고, 문제가 있으면 Teensy 4.1로 바꾼다.")]),
  B("PRD에 ESP32-S3 추가와 두 보드 역할 분담을 반영한다."),
  B("구매 전 Pi 5·AI HAT+ 가격과 재고를 다시 확인한다 (2026년 메모리 가격 변동)."),

  H2("참고 자료"),
  ...[
    "Raspberry Pi 가격 인상: omgubuntu.co.uk/2026/02/raspberry-pi-annouce-more-price-rises, bgr.com/2150106/raspberry-pi-cost-ram-crisis",
    "AI HAT+ 2 (Hailo-10H, $130): tecnoblog.net/noticias/raspberry-pi-anuncia-placa-ai-hat-2-com-40-tops-para-ia-preco-e-alto",
    "Jetson 가격 인상 ($249 → $399): cnx-software.com/2026/07/22/nvidia-increases-the-price-of-jetson-modules-and-devkits-by-up-to-101",
    "micro-ROS 지원 하드웨어: micro.ros.org/docs/overview/hardware, github.com/micro-ROS/micro_ros_arduino",
    "Pi 5 NCNN 67ms: Ultralytics 공개 Raspberry Pi 5 YOLO26n 벤치마크 (추론 백엔드 비교 보고서 인용)",
    "Nooki 내부: PRD(2026.10.06), 하드웨어 부품 정리 보고서(2026.10.14), code/raspberrypi/benchmark.py, code/nooki_sbc/DESIGN.md",
  ].map((t) => B([run(t, { size: 8.5, color: SLATE })])),
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 20, color: INK } } },
    paragraphStyles: [
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: INK }, paragraph: { spacing: { before: 480, after: 160 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: FONT, color: INK }, paragraph: { spacing: { before: 280, after: 100 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "dot", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] }] },
  features: { updateFields: true },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1247, bottom: 1134, left: 1134, right: 1134, header: 567, footer: 567 } }, titlePage: true },
    headers: { default: new Header({ children: [new Paragraph({ alignment: R, children: [run("Nooki MCU · SBC 분석 보고서", { size: 8, color: SLATE })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: R, children: [
      new TextRun({ children: [PageNumber.CURRENT, " / ", PageNumber.TOTAL_PAGES], font: FONT, size: 16, color: SLATE })] })] }) },
    children: [...cover, ...body],
  }],
});

const out = process.argv[2];
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("saved", out); });
