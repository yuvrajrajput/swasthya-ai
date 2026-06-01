-- Run this if query_logs stays EMPTY or RLS error 42501 on insert
-- App uses INSERT only (no public SELECT on query_logs) — fix is in app.py returning="minimal"
-- Supabase → SQL Editor → paste → Run

-- 1) Add any missing columns (safe if table was created earlier)
alter table query_logs add column if not exists length int not null default 0;
alter table query_logs add column if not exists was_emergency boolean not null default false;
alter table query_logs add column if not exists latency_ms int not null default 0;
alter table query_logs add column if not exists input_tokens int not null default 0;
alter table query_logs add column if not exists output_tokens int not null default 0;
alter table query_logs add column if not exists cost_usd float not null default 0;
alter table query_logs add column if not exists cached boolean not null default false;

-- 2) Ensure anon role can insert (API uses publishable/anon key)
grant usage on schema public to anon, authenticated;
grant insert on table public.query_logs to anon, authenticated;
grant select, insert, update on table public.response_cache to anon, authenticated;

-- 3) RLS + policy for anon INSERT on query_logs
alter table query_logs enable row level security;

drop policy if exists "insert_only_query_logs" on query_logs;

create policy "insert_only_query_logs"
on public.query_logs
as permissive
for insert
to anon, authenticated
with check (true);

-- 4) Test row (you should see 1 row in Table Editor after Run)
insert into public.query_logs (query, length, was_emergency, cached)
values ('supabase test row', 3, false, false);
