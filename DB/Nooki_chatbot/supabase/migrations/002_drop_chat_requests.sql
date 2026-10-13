-- 프로토타입 단계에서는 IP별 사용량 제한이 불필요해 요청 기록 테이블 제거
drop table if exists public.nooki_chat_requests;
