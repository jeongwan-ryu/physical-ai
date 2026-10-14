-- 모든 문서가 같은 PRD 하나에서 나오고 어디서도 읽지 않아 출처 컬럼 제거
alter table public.nooki_documents drop column if exists source;
