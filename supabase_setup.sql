-- Swasthya AI — Supabase setup (run entire file in SQL Editor)
-- Dashboard: supabase.com → your project → SQL Editor → New query → paste → Run

-- ─── Step 1: Tables ─────────────────────────────────────────────
create table if not exists query_logs (
  id uuid primary key default gen_random_uuid(),
  query text not null,
  length int not null,
  timestamp timestamptz not null default now(),
  was_emergency boolean not null default false,
  latency_ms int not null default 0,
  input_tokens int not null default 0,
  output_tokens int not null default 0,
  cost_usd float not null default 0,
  cached boolean not null default false
);

create table if not exists response_cache (
  id uuid primary key default gen_random_uuid(),
  query text unique not null,
  response text not null,
  timestamp timestamptz not null default now()
);

-- ─── Step 2: Row Level Security ─────────────────────────────────
alter table query_logs enable row level security;
alter table response_cache enable row level security;

-- Re-run safe: drop old policies if they exist
drop policy if exists "insert_only_query_logs" on query_logs;
drop policy if exists "insert_select_response_cache" on response_cache;
drop policy if exists "insert_only_response_cache" on response_cache;
drop policy if exists "update_only_response_cache" on response_cache;

-- API role must be explicit (fixes empty query_logs on some projects)
grant usage on schema public to anon, authenticated;
grant insert on table public.query_logs to anon, authenticated;
grant select, insert, update on table public.response_cache to anon, authenticated;

-- query_logs: anon key can INSERT only (app logging). No public read.
create policy "insert_only_query_logs"
on public.query_logs
as permissive
for insert
to anon, authenticated
with check (true);

-- response_cache: app needs SELECT (cache hit) + INSERT + UPDATE (upsert)
create policy "insert_select_response_cache"
on public.response_cache
as permissive
for select
to anon, authenticated
using (true);

create policy "insert_only_response_cache"
on public.response_cache
as permissive
for insert
to anon, authenticated
with check (true);

create policy "update_only_response_cache"
on public.response_cache
as permissive
for update
to anon, authenticated
using (true);

-- No DELETE policy on either table (anon cannot delete)

-- ─── Step 3: Verify (optional — run separately) ─────────────────
-- select tablename, rowsecurity from pg_tables
--   where schemaname = 'public' and tablename in ('query_logs', 'response_cache');
-- Expected: rowsecurity = true for both.
--
-- Top 10 symptoms (after you have data):
-- select query, count(*) as n from query_logs group by query order by n desc limit 10;
