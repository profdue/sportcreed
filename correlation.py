-- correlation.py — FIRST VERSION (not building stage)
-- Creates the ONE table this app uses. Do NOT point it at king_matches.
-- Run this in Supabase > SQL Editor, then Reboot the Streamlit app.

create table if not exists public.correlation_matches (
  id bigint generated always as identity primary key,
  home_team text,
  away_team text,
  created_at timestamptz not null default now(),
  raw_paste text,
  decision text,
  decision_reason text,
  main_pick_key text,
  main_pick_label text,
  main_pick_decision text,
  main_pick_margin_pct double precision,
  survivors_count integer,
  home_score integer,
  away_score integer,
  settled_at timestamptz,
  is_deleted boolean not null default false,
  picks jsonb not null default '[]'::jsonb,
  sections jsonb not null default '{}'::jsonb
);

-- Allow the anon key (used by Streamlit Secrets) to read/write this table.
alter table public.correlation_matches enable row level security;

drop policy if exists correlation_all on public.correlation_matches;
create policy correlation_all on public.correlation_matches
  for all
  using (true)
  with check (true);
