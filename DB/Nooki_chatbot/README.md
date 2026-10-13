# Nooki_chatbot

Nooki 홈페이지(`index.html`) 오른쪽 아래에 뜨는 RAG 챗봇입니다. 방문자 질문을 Nooki PRD에서 찾아 Gemini가 답합니다. 로그인 없이 누구나 쓸 수 있습니다.

```
[홈페이지 위젯 nooki-chat.js]
   │ 질문 + 최근 대화 6턴
   ▼
[Edge Function nooki-chat]  ── 입력 길이 제한
   │ ① gemini-embedding-2 로 질문 임베딩 (768차원)
   │ ② nooki_match_chunks 로 유사 청크 상위 4개 검색 (유사도 0.6 이상)
   │ ③ gemini-3.5-flash-lite 가 검색된 PRD 내용만 근거로 답변
   ▼
{ answer, sources }
```

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `nooki-chat.js` | 홈페이지 위젯. `index.html`에 `<script src="DB/Nooki_chatbot/nooki-chat.js" defer>` 한 줄로 연결 |
| `knowledge/nooki_prd.md` | 챗봇 지식 원본. PRD docx를 주제별 6개 섹션으로 정리 |
| `supabase/migrations/001_nooki_chatbot.sql` | 테이블·인덱스·RLS·검색 함수 |
| `supabase/migrations/002_drop_chat_requests.sql` | 사용량 제한용 `nooki_chat_requests` 테이블 제거 (프로토타입 단계에서 불필요) |
| `supabase/build_seed.ps1` | `nooki_prd.md` → `seed.sql` 생성 |
| `supabase/seed.sql` | 지식 데이터 입력용 SQL (자동 생성 파일) |
| `supabase/functions/nooki-chat/index.ts` | Edge Function 소스 (배포된 버전과 동일) |

## Supabase 구성

실습용 `documents` / `chunks`와 같은 프로젝트(`somiyetvungpiupzktav`)를 쓰지만, 테이블은 완전히 분리되어 있습니다.

| 이름 | 설명 |
|---|---|
| `nooki_documents` | 지식 문서 (제목, 출처) |
| `nooki_chunks` | 잘린 본문 + `embedding vector(768)` |
| `nooki_match_chunks()` | 코사인 유사도 검색 함수 (anon/authenticated 실행 권한 없음) |
| Edge Function `nooki-chat` | 챗봇 API |
| Secret `GEMINI_API_KEY` | 실습 때 등록한 키를 같이 사용 |

두 테이블 모두 RLS가 켜져 있고 정책이 없습니다. 브라우저(anon key)에서는 직접 읽거나 쓸 수 없고, Edge Function이 service_role로만 접근합니다.

## 지식 내용 수정하기

1. `knowledge/nooki_prd.md`를 고칩니다. `## 제목` 하나가 문서 하나입니다.
2. `seed.sql`을 다시 만듭니다.
   ```powershell
   powershell -ExecutionPolicy Bypass -File DB/Nooki_chatbot/supabase/build_seed.ps1
   ```
3. Supabase SQL Editor에서 `seed.sql`을 실행합니다. 기존 Nooki 지식을 지우고 다시 넣습니다.
4. 임베딩을 만듭니다. 비어 있는 청크만 처리합니다.
   ```bash
   curl -X POST https://somiyetvungpiupzktav.supabase.co/functions/v1/nooki-chat \
     -H "Authorization: Bearer <anon key>" -H "Content-Type: application/json" \
     -d '{"action":"reindex"}'
   ```

## 설정값 (`supabase/functions/nooki-chat/index.ts`)

| 상수 | 값 | 의미 |
|---|---|---|
| `MATCH_COUNT` / `MATCH_THRESHOLD` | 4 / 0.6 | 검색할 청크 수 / 최소 유사도 |
| `MAX_QUESTION_CHARS` | 500 | 질문 최대 길이 |
| `CHAT_MODEL` | `gemini-3.5-flash-lite` | 답변 모델 (품질을 높이려면 `gemini-3.8-flash`) |

함수를 고친 뒤에는 Supabase CLI(`supabase functions deploy nooki-chat`)나 Supabase MCP로 다시 배포해야 반영됩니다.

## 참고

- 위젯은 Shadow DOM 안에 그려지기 때문에 홈페이지의 Tailwind 스타일과 서로 영향을 주지 않습니다. 화면 폭이 480px 이하면 전체 화면으로 열립니다.
- 대화 기록은 브라우저 탭의 sessionStorage에만 저장되고, 탭을 닫으면 사라집니다.
- 요청 횟수 제한은 없습니다. 공개 서비스로 전환할 때는 Google Cloud 콘솔에서 Gemini API 키의 일일 할당량을 걸거나 사용량 제한을 다시 추가하세요.
- anon key는 공개를 전제로 한 키라 위젯 코드에 들어 있어도 됩니다. service_role key는 절대 넣으면 안 됩니다.
