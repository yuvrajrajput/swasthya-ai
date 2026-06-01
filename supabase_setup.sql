-- Run this in Supabase SQL Editor AFTER creating tables
-- Go to: supabase.com → your project → SQL Editor → paste → Run

-- Enable Row Level Security on both tables
alter table query_logs enable row level security;
alter table response_cache enable row level security;

-- query_logs: allow insert only (no public read/update/delete)
create policy "insert_only_query_logs"
on query_logs
for insert
with check (true);

-- response_cache: allow insert + select + update only
-- (needed for cache reads and upserts)
create policy "insert_select_response_cache"
on response_cache
for select
using (true);

create policy "insert_only_response_cache"
on response_cache
for insert
with check (true);

create policy "update_only_response_cache"
on response_cache
for update
using (true);

-- No DELETE allowed on either table for safety
