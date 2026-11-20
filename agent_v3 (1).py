"""Larkspur Monday account check, built from PRD v3. Rules 1-13 in PRD order; first rule that fits decides."""
import pandas as pd, numpy as np, datetime as dt

P = '/mnt/project/'
EMAIL_DAYS = 10        # rule 9
MERGE_SHARE = 0.75     # rule 6, proposed, OPEN 7
SCHOOL_BACK = 0.50     # rule 5, OPEN 8
DROP = 0.30
FULL = 'Request full export of all work orders and vehicle records'
ADMIN = 'Change account admin contact'
V = [f'active_vehicles_w{i}' for i in range(1, 13)]
L = [f'logins_w{i}' for i in range(1, 13)]

def load():
    u = pd.read_csv(P + 'usage_export_2026-10-12.csv', skiprows=1)
    u = u[u.account_id.astype(str).str.startswith('LKS')].reset_index(drop=True)
    s = pd.read_csv(P + 'salesforce_accounts.csv', skiprows=1)
    t = pd.read_csv(P + 'support_tickets_open.csv', skiprows=1)
    for c in ['renewal_date', 'notice_date']:
        s[c] = pd.to_datetime(s[c]).dt.date
    t['opened'] = pd.to_datetime(t.opened).dt.date
    return u, s, t

U, S, T = load()
EXPORT_WK = [dt.date(2026, 7, 20) + dt.timedelta(weeks=i) for i in range(12)]

def incident_weeks(t):
    inc = t[(t.priority == 'P1') & t.resolution.fillna('').str.contains('INC-')].copy()
    inc['inc'] = inc.resolution.str.extract(r'(INC-[\d-]+)')
    out = set()
    for _, g in inc.groupby('inc'):
        if g.account_id.nunique() >= 10:
            d = g.opened.min(); out.add(d - dt.timedelta(days=d.weekday()))
    return out
INC = incident_weeks(T)

# ---------- helpers ----------
def series(row, cols, bad=()):
    x = pd.to_numeric(row[cols], errors='coerce').values.astype(float)
    for i in bad: x[i] = np.nan
    return x

def f3l3(x):
    ok = np.where(~np.isnan(x))[0]
    if len(ok) < 3: return None
    return x[ok[:3]].mean(), x[ok[-3:]].mean(), ok[0]

def chg(x, WK):
    r = f3l3(x)
    if r is None or r[0] <= 0: return None
    return (r[1] - r[0]) / r[0], WK[r[2]]

def pct(c): return f"{abs(c) * 100:.0f}%"
def line(x): return ' '.join('-' if np.isnan(v) else str(int(v)) for v in x)
def mon(d): return d.strftime('%b %-d')

def zero_from(xv, xl):
    """First week from which logins and vehicles are 0 through the end (None if not)."""
    for w in range(1, 12):
        if (np.nan_to_num(xv[w:], nan=-1) == 0).all() and (np.nan_to_num(xl[w:], nan=-1) == 0).all():
            return w
    return None

def starts_at(xv):
    ok = np.where(~np.isnan(xv))[0]
    return ok[0] if len(ok) else None

def find_twin(r, xv, xl, src, bad):
    """Rule 7: old ID goes to 0 the same week a same-name ID starts. Returns (stitched_v, stitched_l, prefix)."""
    if src is None: return None
    for _, o in src[(src.account_name == r.account_name) & (src.account_id != r.account_id)].iterrows():
        ov, ol = series(o, V, bad), series(o, L, bad)
        w = zero_from(xv, xl)
        if w is not None and starts_at(ov) == w and np.isnan(ov[:w]).all():   # this row is the old ID
            sv = np.where(np.isnan(ov), xv, ov); sl = np.where(np.isnan(ol), xl, ol)
            return sv, sl, f"re-keyed to {o.account_id}"
        w = zero_from(ov, ol)
        if w is not None and starts_at(xv) == w and np.isnan(xv[:w]).all():   # this row is the new ID
            sv = np.where(np.isnan(xv), ov, xv); sl = np.where(np.isnan(xl), ol, xl)
            return sv, sl, f"re-keyed from {o.account_id}"
    return None

# ---------- the rules ----------
def decide(r, sf, ctx):
    """r: usage row. sf: Salesforce-like Series (or None). ctx: RUN, WK, bad, tickets, src, group_src, last_email."""
    RUN, WK, bad, t = ctx['RUN'], ctx['WK'], ctx['bad'], ctx['tickets']
    xv, xl = series(r, V, bad), series(r, L, bad)
    if sf is None:                                                         # rule 2
        return 'ask Marcus', 'not in Salesforce by ID or name; show export row', f"vehicles {line(xv)} | logins {line(xl)}"
    own, ren = sf.owner, sf.renewal_date
    dl = ren - dt.timedelta(days=90)
    meta = f"owner {own}; renews {ren}; notice deadline {dl}"
    ev = f"vehicles {line(xv)} | logins {line(xl)} | {sf.fleet_type} | {meta}"
    tk = t[t.account_id == sf.sfid]
    cv = float(sf.contracted_vehicles)
    def util(x):
        r_ = f3l3(x); return None if r_ is None else r_[1] / cv
    # 1
    if sf.status == 'Notice given' and pd.notna(sf.notice_date) and sf.notice_date <= RUN:
        return 'ignore', f"gave notice {sf.notice_date}", meta
    # 3
    if (tk.subject == FULL).any():
        d = tk[tk.subject == FULL].opened.max()
        return 'call this week', f"asked for full export {mon(d)}; goes to Marcus and {own} (OPEN 3b)", meta
    ft, m = sf.fleet_type, RUN.month
    u_ = util(xv)
    # 4
    if ft == 'snowplow' and 3 <= m <= 11:
        return 'ignore', 'plow fleet, in season', meta
    if ft == 'snowplow' and m == 12 and u_ is not None and u_ < .5:
        return 'ask Marcus', 'plow fleet after Nov 1, still under 50% (OPEN 5c)', ev
    # 5
    if ft == 'school bus':
        sep7 = dt.date(RUN.year, 9, 7)
        if (m == 6 and RUN.day >= 15) or m in (7, 8) or (m == 9 and RUN < sep7):
            return 'ignore', 'school fleet, summer', meta
        if RUN >= sep7:
            if u_ is not None and u_ < SCHOOL_BACK:
                n = f3l3(xv)[1]
                return 'call this week', f"school fleet, still at {n:.0f} of {cv:.0f} in {RUN.strftime('%B')}", meta
            return 'ignore', 'school fleet, back', meta
    # 6
    if isinstance(sf.parent_account, str) and ctx['group_src'] is not None:
        P_ = sf.parent_account
        g = ctx['group_src']
        names = S[S.parent_account == P_].account_name
        c_own = chg(xv, WK)
        if c_own and c_own[0] < -DROP:
            a, b, _ = f3l3(xv); fall = a - b
            rise = 0.0
            for _, o in g[g.account_name.isin(names) & (g.account_name != r.account_name)].iterrows():
                rr = f3l3(series(o, V, bad))
                if rr: rise += max(0.0, rr[1] - rr[0])
            if rise >= MERGE_SHARE * fall:
                return 'ignore', f"merged into {P_}", meta
            return 'ask the owner', f"{own}: fell {fall:.0f}, only {rise:.0f} arrived on {P_}", ev
        grp = g[g.account_name.isin(names)]
        tot = grp[V].apply(pd.to_numeric, errors='coerce').sum(axis=0, min_count=len(grp)).values.astype(float)
        for i in bad: tot[i] = np.nan
        gc = chg(tot, WK)
        if gc is not None and not gc[0] < -DROP:
            return 'ignore', f"merged into {P_}", meta
    # 7
    prefix = ''
    tw = find_twin(r, xv, xl, ctx['src'], bad)
    if tw:
        xv, xl, prefix = tw
        u_ = util(xv)
        ev = f"stitched vehicles {line(xv)} | logins {line(xl)} | {sf.fleet_type} | {meta}"
    def out(v, why, e):
        return v, (f"{prefix}; {why}" if prefix else why), e
    # 8
    ok = np.where(~np.isnan(xv))[0]
    last3_blank = np.isnan(xv[-3:]).all()
    if (np.nan_to_num(xl[-3:], nan=-1) == 0).all() and (np.isnan(xv[-3:]) | (xv[-3:] == 0)).all():
        return out('ask Marcus', '0 logins, no vehicles, no re-keyed twin', ev)
    if last3_blank or len(ok) < 3:
        miss = [str(WK[i]) for i in range(12) if np.isnan(xv[i])]
        return out('ask the owner', f"{own}: vehicle weeks missing: {', '.join(miss)} (blank is not zero)", ev)
    c, cl = chg(xv, WK), chg(xl, WK)
    # 9
    if c and c[0] < -DROP:
        le = ctx.get('last_email')
        if le is not None and 0 <= (RUN - le).days <= EMAIL_DAYS:
            return out('ask the owner', f"{own}: down {pct(c[0])}, fleet manager wrote {mon(le)}, ask before calling", ev)
        why = f"down {pct(c[0])} since week of {c[1]}" + (f", logins {'down' if cl[0] < 0 else 'up'} {pct(cl[0])}" if cl else '')
        if RUN > dl: why += f"; deadline passed, notice takes effect at next renewal {ren + dt.timedelta(days=365)}"
        if float(sf.annual_value) > 75000: why += "; over $75,000, goes to Marcus the same week (OPEN 3b)"
        return out('call this week', why, meta)
    # 10
    if cl and cl[0] < -DROP:
        return out('ask Marcus', f"only logins down {pct(cl[0])}, vehicles not (OPEN 1b)", ev)
    # 11
    if u_ is not None and u_ < .5 and RUN <= ren <= RUN + dt.timedelta(days=120) and RUN <= dl:
        return out('ask Marcus', f"under 50% of contracted vehicles ({u_ * 100:.0f}%), renews within 120 days (OPEN 1c)", ev)
    # 12
    a = tk[(tk.subject == ADMIN) & (tk.opened >= RUN - dt.timedelta(days=60))]
    if len(a):
        return out('ask the owner', f"{own}: admin contact change ticket {a.opened.max()} (proposed)", ev)
    # 13
    if prefix:
        return 'ignore', f"{prefix}, flat at about {f3l3(xv)[1]:.0f}", meta
    return 'ignore', 'no drop against its own level', meta

# ---------- Salesforce lookup ----------
def sf_for(aid, name, override=None):
    m = S[S.account_id == aid]
    if m.empty: m = S[S.account_name == name]
    if m.empty: return None
    m = m.iloc[0].copy(); m['sfid'] = m.account_id
    for k, v in (override or {}).items(): m[k] = v
    return m

# ---------- day-one run (live Monday) ----------
def run_day_one(n=None):
    RUN = dt.date(2026, 10, 12)
    bad = [i for i, w in enumerate(EXPORT_WK) if w in INC]
    ctx = dict(RUN=RUN, WK=EXPORT_WK, bad=bad, tickets=T[T.opened <= RUN], src=U, group_src=U, last_email=None)
    rows = []
    for _, r in (U if n is None else U.head(n)).iterrows():
        v, why, ev = decide(r, sf_for(r.account_id, r.account_name), ctx)
        rows.append((r.account_id, r.account_name, v, why))
    return pd.DataFrame(rows, columns=['id', 'name', 'verdict', 'reason'])

# ---------- Door 3 run ----------
def run_door3(screen='/home/claude/d3/screen_these_30.csv', context='/home/claude/d3/account_context_30.csv'):
    d = pd.read_csv(screen); c = pd.read_csv(context).set_index('account_id')
    rows = []
    for _, r in d.iterrows():
        RUN = pd.to_datetime(r.decision_date).date()
        WK = [pd.to_datetime(r.window_w1_starts).date() + dt.timedelta(weeks=i) for i in range(12)]
        cc = c.loc[r.account_id]
        sf = sf_for(r.account_id, r.account_name, dict(
            fleet_type=cc.fleet_type,
            renewal_date=pd.to_datetime(cc.renewal_date).date(),
            annual_value=cc.annual_value,
            parent_account=cc.parent_account if isinstance(cc.parent_account, str) else np.nan,
            contracted_vehicles=r.contracted_vehicles))
        same_window = WK[0] == EXPORT_WK[0]
        bad = [i for i, w in enumerate(WK) if w in INC]
        le = pd.to_datetime(cc.last_email_from_fleet_manager).date() if pd.notna(cc.last_email_from_fleet_manager) else None
        ctx = dict(RUN=RUN, WK=WK, bad=bad, tickets=T[T.opened <= RUN],
                   src=U if same_window else None, group_src=U if same_window else None, last_email=le)
        v, why, ev = decide(r, sf, ctx)
        rows.append(dict(item_id=r.account_id, name=r.account_name, verdict=v, reason=why,
                         door3_file='screen_these_30.csv + account_context_30.csv'))
    return pd.DataFrame(rows)
