-- nooki_chunks 를 주제별로 나눈 테이블 3개 (구조는 nooki_chunks 와 동일)
-- overview: 소개·시나리오·KPI·MVP / hardware: 구성·보드·전원·통신·리스크 / software: 구성·추론 비교·추론 장치
create table public.nooki_overview_chunks (like public.nooki_chunks including defaults including identity);
create table public.nooki_hardware_chunks (like public.nooki_chunks including defaults including identity);
create table public.nooki_software_chunks (like public.nooki_chunks including defaults including identity);

alter table public.nooki_overview_chunks add primary key (id), add foreign key (document_id) references public.nooki_documents(id) on update cascade on delete cascade;
alter table public.nooki_hardware_chunks add primary key (id), add foreign key (document_id) references public.nooki_documents(id) on update cascade on delete cascade;
alter table public.nooki_software_chunks add primary key (id), add foreign key (document_id) references public.nooki_documents(id) on update cascade on delete cascade;

create index on public.nooki_overview_chunks using hnsw (embedding extensions.vector_cosine_ops);
create index on public.nooki_hardware_chunks using hnsw (embedding extensions.vector_cosine_ops);
create index on public.nooki_software_chunks using hnsw (embedding extensions.vector_cosine_ops);

-- 정책 없음 = Edge Function(service_role)만 접근
alter table public.nooki_overview_chunks enable row level security;
alter table public.nooki_hardware_chunks enable row level security;
alter table public.nooki_software_chunks enable row level security;
