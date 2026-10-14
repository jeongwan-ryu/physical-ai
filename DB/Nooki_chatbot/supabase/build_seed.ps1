# knowledge/nooki_prd.md 의 각 "## 제목" 섹션을 nooki_documents 1건 + 잘린 nooki_chunks 로 넣는 seed.sql 생성
# 사용: powershell -File DB/Nooki_chatbot/supabase/build_seed.ps1  →  seed.sql 을 SQL Editor 에서 실행 → 임베딩 백필 호출
$root = Split-Path -Parent $PSScriptRoot
$md = Get-Content -Raw -Encoding utf8 (Join-Path $root "knowledge\nooki_prd.md")

$sections = [regex]::Split($md, '(?m)^## ') | Select-Object -Skip 1
$sql = New-Object System.Text.StringBuilder
[void]$sql.AppendLine("-- 자동 생성 파일 (build_seed.ps1). 직접 고치지 말고 knowledge/nooki_prd.md 를 고친 뒤 다시 생성하세요.")
[void]$sql.AppendLine("truncate public.nooki_chunks, public.nooki_documents restart identity;  -- 기존 지식을 비우고 id 를 1부터 다시 매김")
[void]$sql.AppendLine("")

foreach ($s in $sections) {
  $lines = $s -split "`r?`n", 2
  $title = $lines[0].Trim().Replace("'", "''")
  $body = $lines[1].Trim()
  [void]$sql.AppendLine("with d as (")
  [void]$sql.AppendLine("  insert into public.nooki_documents (title) values ('$title') returning id")
  [void]$sql.AppendLine(")")
  [void]$sql.AppendLine("insert into public.nooki_chunks (document_id, chunk_index, content)")
  [void]$sql.AppendLine("select d.id, c.chunk_index, c.content from d, public.split_into_chunks(`$body`$")
  [void]$sql.AppendLine($body)
  [void]$sql.AppendLine("`$body`$, 500, 100) c;")
  [void]$sql.AppendLine("")
}

$out = Join-Path $PSScriptRoot "seed.sql"
[IO.File]::WriteAllText($out, $sql.ToString(), (New-Object Text.UTF8Encoding($false)))
"wrote $out ($($sections.Count) documents)"
