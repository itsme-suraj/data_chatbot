create table if not exists ai_memory (
  id uuid primary key default gen_random_uuid(),
  sig text, question text, code text,
  rating int default 0,
  created_at timestamptz default now()
);
alter table ai_memory enable row level security;
create policy "open_access" on ai_memory for all using (true) with check (true);
