"""Draw the paper's figures from the results files.

usage: python3 make_paper_figures.py RESULTS_DIR FONT_DIR [OUT_DIR]

Numbers come from the rerun's result files where a file exists:
  eventstudy_estimates.csv  -> Figure 2
  event_estimates.csv       -> Figure 3 (upgrade estimates and SEs)
  results_all.txt           -> Figure 1 counts, Figure A1, suspension row
SWE-Bench Pro scores are Anthropic system-card values (Table A4 caption).
"""
import re, sys, os
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'results')
FONTS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, '..', 'fonts')
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, '..', 'figures'); os.makedirs(OUT, exist_ok=True)

for f in ('Regular', 'Bold', 'Italic', 'BoldItalic'):
    font_manager.fontManager.addfont(os.path.join(FONTS, f'OpenSans-{f}.ttf'))
MAIN, COMP, ACCENT = '#9A9CB0', '#2A8FBD', '#6658E8'
MAIN_T, COMP_T, ACCENT_T = '#F4F3FA', '#F0F0F4', '#ECEAFB'
INK, MUTED, GRID = '#090C27', '#454754', '#E3E2EC'
mpl.rcParams.update({'font.family': ['Open Sans', 'DejaVu Sans'], 'font.size': 7,
    'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'axes.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'savefig.dpi': 300, 'pdf.fonttype': 42})
W = 183 / 25.4
MINUS = '−'


def fmt(x, nd=1, sign=True):
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    if s in ('-0.0', '+0.0'):
        s = '0.0'
    return s.replace('-', MINUS)


def est(e, lo, hi):
    return f'{fmt(e)} ({fmt(lo)}, {fmt(hi)})'


txt = open(os.path.join(R, 'results_all.txt'), encoding='utf-8').read()

def grab(pattern):
    m = re.search(pattern, txt)
    assert m, pattern
    return [float(g) for g in m.groups()]

NUM = r'([+-]?\d+\.\d+)'
SUSP = grab(r'\[per PR\] merged_7d\s+reversal contrast \(P1\+P3\)/2 - P2: ' + NUM + r' \[' + NUM + r', ' + NUM + r'\]')
active = eval(re.search(r'treated developers active by phase: (\{[^}]*\})', txt).group(1))
share = eval(re.search(r'Fable-tagged share of treated PRs by phase: (\{[^}]*\})', txt).group(1))
POOL = {
    'raw': grab(r'pooled difference ' + NUM + r' \[' + NUM + r', ' + NUM + r'\]'),
    'day': grab(r'pooled, day fixed effects only \(clustered by developer\): ' + NUM + r' \[' + NUM + r', ' + NUM + r'\]'),
    'dev': grab(r'within developer \(\+ day FE\): ' + NUM + r' \[' + NUM + r', ' + NUM + r'\]'),
    'repo': grab(r'within developer and repository \(\+ day FE\): ' + NUM + r' \[' + NUM + r', ' + NUM + r'\]'),
}
print('suspension', SUSP, 'active', active, 'share', share)
print('pooled', POOL)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight', pad_inches=0.04)
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


# ------------------------------------------------------------------ Figure 1
def fig1():
    fig = plt.figure(figsize=(W, 3.0))
    # panel a: timeline
    ax = fig.add_axes([0.135, 0.53, 0.845, 0.36])
    d = lambda m, day: dt.date(2026, m, day).toordinal()
    x0, x1 = d(1, 25), d(10, 12)
    ax.set_xlim(x0, x1); ax.set_ylim(-0.95, 1.25)
    ax.axvspan(d(6, 12), d(7, 1), ymax=0.84, color=COMP_T, zorder=0)
    for xx in (d(6, 12), d(7, 1)):
        ax.axvline(xx, ymax=0.84, color=MAIN, lw=0.6, ls=(0, (2, 2)), zorder=0.5)
    ax.axhline(0.75, color=GRID, lw=0.8, zorder=1)
    ups = [(2, 5, 'Opus 4.5→4.6', False, 'below'), (2, 17, 'Sonnet 4.5→4.6', False, 'above'),
           (4, 16, 'Opus 4.6→4.7', True, 'below'), (5, 28, 'Opus 4.7→4.8', True, 'below'),
           (6, 30, 'Sonnet 4.6→5', False, 'above'), (7, 24, 'Opus 4.8→5', True, 'below'),
           (9, 22, 'Opus 5→5.5', True, 'below'), (9, 28, 'Sonnet 5→5.5', True, 'above')]
    for m, day, lab, main, pos in ups:
        x = d(m, day)
        ax.plot(x, 0.75, 'o', ms=6, mfc=MAIN if main else 'white', mec=MAIN, mew=1.1, zorder=3)
        ax.text(x, 0.75 + (0.27 if pos == 'above' else -0.27), lab, ha='center',
                va='bottom' if pos == 'above' else 'top', fontsize=6.5, color=INK)
    # Fable 5 track
    y = -0.45
    ax.plot([d(6, 9), d(6, 12)], [y, y], color=ACCENT, lw=2.4, solid_capstyle='butt')
    ax.plot([d(6, 12), d(7, 1)], [y, y], color=ACCENT, lw=1.0, ls=(0, (2, 1.5)))
    ax.plot([d(7, 1), x1], [y, y], color=ACCENT, lw=2.4, solid_capstyle='butt')
    ax.text(d(6, 8), y, 'released Jun 9 ', ha='right', va='center', fontsize=6.5, color=ACCENT)
    ax.text((d(6, 12) + d(7, 1)) / 2, y + 0.12, 'suspended', ha='center', va='bottom', fontsize=6.5, color=ACCENT)
    ax.text(d(7, 6), y + 0.12, 'restored Jul 1', ha='left', va='bottom', fontsize=6.5, color=ACCENT)
    ax.set_yticks([0.75, y]); ax.set_yticklabels(['Opus and Sonnet\nupgrades', 'Fable 5'], fontsize=7, fontweight='bold')
    ax.get_yticklabels()[1].set_color(ACCENT); ax.get_yticklabels()[0].set_color(INK)
    ax.tick_params(axis='y', length=0, pad=6)
    months = [(2, 'Feb'), (3, 'Mar'), (4, 'Apr'), (5, 'May'), (6, 'Jun'), (7, 'Jul'), (8, 'Aug'), (9, 'Sep'), (10, 'Oct')]
    ax.set_xticks([d(m, 1) for m, _ in months]); ax.set_xticklabels([n for _, n in months])
    ax.tick_params(axis='x', length=2)
    for s in ('left',): ax.spines[s].set_visible(False)
    fig.text(0.01, 0.975, 'a', fontweight='bold', fontsize=8, va='top')
    fig.text(0.035, 0.975, 'Model changes in Claude Code, 2026', fontsize=7, va='top', color=MUTED)
    ax.plot([], [], 'o', ms=5, mfc=MAIN, mec=MAIN, label='upgrade in the main pool')
    ax.plot([], [], 'o', ms=5, mfc='white', mec=MAIN, label='other upgrade')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, fontsize=6.5,
              handletextpad=0.3, columnspacing=1.2, borderaxespad=0)

    # panel b: same encoding as a, zoomed to the suspension design
    bx = fig.add_axes([0.135, 0.085, 0.845, 0.235])
    t = lambda m, day, h=0: dt.datetime(2026, m, day, h).timestamp() / 86400
    b0, b1 = t(5, 26), t(7, 16)
    s0, s1 = t(6, 12, 21), t(7, 1)
    bx.set_xlim(b0, b1); bx.set_ylim(-1.05, 1.0)
    bx.axvspan(s0, s1, color=COMP_T, zorder=0)
    for xx in (s0, s1):
        bx.axvline(xx, color=MAIN, lw=0.6, ls=(0, (2, 2)), zorder=0.5)
    yf = 0.5
    bx.plot([t(6, 9, 18), s0], [yf, yf], color=ACCENT, lw=2.4, solid_capstyle='butt')
    bx.plot([s0, s1], [yf, yf], color=ACCENT, lw=1.0, ls=(0, (2, 1.5)))
    bx.plot([s1, b1], [yf, yf], color=ACCENT, lw=2.4, solid_capstyle='butt')
    bx.text(t(6, 9, 6), yf, 'released Jun 9 ', ha='right', va='center', fontsize=6.5, color=ACCENT)
    bx.text((s0 + s1) / 2, yf + 0.13, 'suspended Jun 12 to Jul 1', ha='center', va='bottom', fontsize=6.5, color=ACCENT)
    bx.text(t(7, 2, 12), yf + 0.13, 'restored Jul 1', ha='left', va='bottom', fontsize=6.5, color=ACCENT)
    yp, hp = -0.4, 0.62
    phases = [('P0', 'before release', t(5, 26), t(6, 9, 18)),
              ('P1', '', t(6, 9, 18), s0),
              ('P2', 'suspended (reference)', t(6, 15), s1),
              ('P3', 'restored', t(7, 2), b1)]
    for name, sub, a0, a1 in phases:
        bx.add_patch(Rectangle((a0, yp - hp / 2), a1 - a0, hp, facecolor='none', edgecolor=MAIN, lw=0.6, zorder=2))
        lab = name + ('  ' + sub if sub else '')
        bx.text((a0 + a1) / 2, yp, lab, ha='center', va='center', fontsize=6.5, color=INK, zorder=3)
    for a0, a1 in [(s0, t(6, 15)), (s1, t(7, 2))]:
        bx.add_patch(Rectangle((a0, yp - hp / 2), a1 - a0, hp, facecolor='white', edgecolor=MAIN, hatch='////', lw=0.6, zorder=2))
    bx.set_yticks([yf, yp]); bx.set_yticklabels(['Fable 5', 'Phases'], fontsize=7, fontweight='bold')
    bx.get_yticklabels()[0].set_color(ACCENT); bx.get_yticklabels()[1].set_color(INK)
    bx.tick_params(axis='y', length=0, pad=6)
    ticks = [('May 26', t(5, 26)), ('Jun 1', t(6, 1)), ('Jun 15', t(6, 15)), ('Jul 1', s1), ('Jul 15', t(7, 15, 23))]
    bx.set_xticks([x for _, x in ticks]); bx.set_xticklabels([l for l, _ in ticks])
    bx.tick_params(axis='x', length=2)
    bx.spines['left'].set_visible(False)
    fig.text(0.01, 0.375, 'b', fontweight='bold', fontsize=8, va='top')
    fig.text(0.035, 0.375, 'The suspension, enlarged: phases by PR opening time (UTC)', fontsize=7, va='top', color=MUTED)
    from matplotlib.patches import Patch
    bx.legend(handles=[Patch(facecolor='white', edgecolor=MAIN, hatch='////', lw=0.6, label='excluded: days right after a switch')],
              loc='lower right', bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=6.5,
              handlelength=1.6, handleheight=0.9, handletextpad=0.4, borderaxespad=0)
    save(fig, 'fig1_design')


# ------------------------------------------------------------------ Figure 2
def fig2():
    es = pd.read_csv(os.path.join(R, 'eventstudy_estimates.csv'))
    labels = ['May 26\u2013Jun 1', 'Jun 2\u20139', 'Jun 9\u201312', 'Jun 15\u201321', 'Jun 22\u201328', 'Jun 29\u201330', 'Jul 2\u20138', 'Jul 9\u201315']
    groups = {0: 'Before release', 2: 'Fable 5 available', 3: 'Suspended', 6: 'Restored'}
    ypos = np.array([0, 1, 2.35, 3.7, 4.7, 5.7, 7.05, 8.05])
    fig = plt.figure(figsize=(W, 2.9))
    a2 = fig.add_axes([0.45, 0.15, 0.32, 0.74])
    a1 = fig.add_axes([0.255, 0.15, 0.135, 0.74])
    a1.set_ylim(ypos[-1] + 0.6, -0.6); a1.set_yticks([]); a1.spines['left'].set_visible(False); a1.set_facecolor('none')
    a1.set_xlim(0, 100); a1.set_xticks([0, 50, 100]); a1.set_xlabel('% of PRs')
    a2.set_ylim(ypos[-1] + 0.6, -0.6)
    a2.set_yticks([]); a2.spines['left'].set_visible(False); a2.set_facecolor('none')
    for y0, y1 in [(ypos[2] - 0.5, ypos[2] + 0.5), (ypos[6] - 0.5, ypos[7] + 0.5)]:
        f0 = fig.transFigure.inverted().transform(a2.transData.transform((0, y0)))[1]
        f1 = fig.transFigure.inverted().transform(a2.transData.transform((0, y1)))[1]
        fig.patches.append(Rectangle((0.005, f1), 0.99, f0 - f1, transform=fig.transFigure, color=ACCENT_T, lw=0, zorder=-10))
    for i, row in es.iterrows():
        y = ypos[i]
        ref = 'reference' in row.period
        if ref:
            a2.plot(0, y, 'o', ms=5, mfc='white', mec=MAIN, mew=1, zorder=3)
        else:
            a2.plot([row.lo, row.hi], [y, y], color=MAIN, lw=1.1, solid_capstyle='butt')
            a2.plot(row.est, y, 'o', ms=5, color=MAIN, zorder=3)
        sh = row.fable_share
        a1.barh(y, sh, height=0.55, color=ACCENT, lw=0)
        a1.text(sh + 3, y, f'{sh:.0f}' if sh >= 1 else ('<1' if sh > 0 else '0'), va='center', ha='left', fontsize=6, color=INK)
        yf = fig.transFigure.inverted().transform(a2.transData.transform((0, y)))[1]
        fig.text(0.905, yf, 'reference' if ref else est(row.est, row.lo, row.hi), ha='right', va='center', fontsize=6.5,
                 color=MUTED if ref else INK)
        fig.text(0.985, yf, f'{int(row.devs)}', ha='right', va='center', fontsize=6.5, color=MUTED)
        fig.text(0.235, yf, labels[i], ha='right', va='center', fontsize=6.5)
        if i in groups:
            fig.text(0.012, yf, groups[i], ha='left', va='center', fontsize=7, fontweight='bold')
    a2.set_xlim(-11, 11); a2.set_xticks([-10, -5, 0, 5, 10]); a2.set_xticklabels([f'{v:+d}'.replace('-', MINUS) if v else '0' for v in [-10, -5, 0, 5, 10]])
    a2.axvline(0, color=GRID, lw=0.8, zorder=0)
    a2.set_xlabel('Percentage points (95% CI)')
    top = 0.95
    fig.text(0.255, top, 'Named Fable 5', fontweight='bold', fontsize=7, va='center')
    fig.text(0.45, top, 'Change in the 7-day merge rate', fontweight='bold', fontsize=7, va='center')
    fig.text(0.905, top, 'vs Jun 22\u201328', ha='right', fontsize=6.5, color=MUTED, va='center')
    fig.text(0.985, top, 'Developers', ha='right', fontsize=6.5, color=MUTED, va='center')
    save(fig, 'fig2_eventstudy')


# ------------------------------------------------------------------ Figure 3
def fig3():
    ev = pd.read_csv(os.path.join(R, 'event_estimates.csv'))
    swe = {('Opus 4.6', 'Opus 4.7'): (53.4, 64.3), ('Opus 4.7', 'Opus 4.8'): (64.3, 69.2),
           ('Sonnet 4.6', 'Sonnet 5'): (58.1, 63.2), ('Opus 4.8', 'Opus 5'): (69.2, 79.2),
           ('Opus 5', 'Opus 5.5'): (79.2, 89.9), ('Sonnet 5', 'Sonnet 5.5'): (63.2, 81.3)}
    rows = []
    for _, r in ev.iterrows():
        k = (r.old, r.new)
        if k in swe:
            gain = round(swe[k][1] - swe[k][0], 1)
            lab = f"{r.old} → {r.new.split()[1]}"
            rows.append(dict(lab=lab, gain=gain, e=r.all_est, lo=r.all_est - 1.96 * r.all_se, hi=r.all_est + 1.96 * r.all_se,
                             pooled=int(r.event) in (3, 4, 6, 7, 8)))
    rows.sort(key=lambda d: d['gain'])
    # pooled estimate from results file
    pooled = grab(r'upgrades merged within 7 days \(main\): ' + NUM + r' \[' + NUM + r', ' + NUM + r'\]')
    fig = plt.figure(figsize=(W, 3.25))
    n = len(rows)
    yy = list(range(n)) + [n + 0.6, n + 1.75]
    a1 = fig.add_axes([0.22, 0.15, 0.22, 0.75]); a2 = fig.add_axes([0.50, 0.15, 0.30, 0.75])
    for ax in (a1, a2):
        ax.set_ylim(yy[-1] + 0.6, -0.6); ax.set_yticks([]); ax.spines['left'].set_visible(False); ax.set_facecolor('none')
    f = lambda y: fig.transFigure.inverted().transform(a1.transData.transform((0, y)))[1]
    fy0, fy1 = f(yy[-1] - 0.55), f(yy[-1] + 0.55)
    fig.patches.append(Rectangle((0.005, fy1), 0.99, fy0 - fy1, transform=fig.transFigure, color=ACCENT_T, lw=0, zorder=-10))
    fsep = f(n - 0.3 + 0.5)
    fig.add_artist(plt.Line2D([0.01, 0.99], [fsep, fsep], color=GRID, lw=0.6))
    for y, r in zip(yy, rows):
        alpha = 1 if r['pooled'] else 0.45
        a1.barh(y, r['gain'], height=0.55, color=MAIN, alpha=alpha, lw=0)
        a1.text(r['gain'] + 0.5, y, fmt(r['gain']), va='center', fontsize=6.5)
        a2.plot([r['lo'], r['hi']], [y, y], color=MAIN, lw=1.1, alpha=alpha)
        a2.plot(r['e'], y, 'o', ms=5, mfc=MAIN if r['pooled'] else 'white', mec=MAIN, mew=1, alpha=1 if r['pooled'] else 0.7, zorder=3)
        fig.text(0.205, f(y), r['lab'], ha='right', va='center', fontsize=6.5)
        fig.text(0.985, f(y), est(r['e'], r['lo'], r['hi']), ha='right', va='center', fontsize=6.5)
    y = yy[n]
    a2.plot([pooled[1], pooled[2]], [y, y], color=MAIN, lw=1.1); a2.plot(pooled[0], y, 'D', ms=5, color=MAIN, zorder=3)
    fig.text(0.205, f(y), 'Five upgrades pooled', ha='right', va='center', fontsize=7, fontweight='bold')
    fig.text(0.985, f(y), est(*pooled), ha='right', va='center', fontsize=6.5, fontweight='bold')
    y = yy[n + 1]
    a1.barh(y, 10.8, height=0.55, color=ACCENT, lw=0); a1.text(11.3, y, fmt(10.8), va='center', fontsize=6.5)
    a2.plot([SUSP[1], SUSP[2]], [y, y], color=ACCENT, lw=1.1); a2.plot(SUSP[0], y, 'D', ms=5, color=ACCENT, zorder=3)
    fig.text(0.205, f(y), 'Opus 4.8 → Fable 5 (suspension)', ha='right', va='center', fontsize=7, fontweight='bold')
    fig.text(0.985, f(y), est(*SUSP), ha='right', va='center', fontsize=6.5, fontweight='bold')
    a1.set_xlim(0, 21); a1.set_xticks([0, 5, 10, 15, 20]); a1.set_xlabel('SWE-Bench Pro gain (points)')
    a2.set_xlim(-10.5, 10.5); a2.set_xticks([-10, -5, 0, 5, 10]); a2.set_xticklabels(['−10', '−5', '0', '+5', '+10'])
    a2.axvline(0, color=GRID, lw=0.8, zorder=0)
    a2.set_xlabel('Change in seven-day merge share (points, 95% CI)')
    top = 0.97
    fig.text(0.22, top, 'a', fontweight='bold', fontsize=8, va='center'); fig.text(0.24, top, 'Benchmark', fontweight='bold', fontsize=7, va='center')
    fig.text(0.24, top - 0.05, 'newer model higher in every change', fontsize=6.5, va='center', color=MUTED)
    fig.text(0.50, top, 'b', fontweight='bold', fontsize=8, va='center'); fig.text(0.52, top, 'Pull requests merged', fontweight='bold', fontsize=7, va='center')
    fig.text(0.52, top - 0.05, 'every 95% interval includes 0', fontsize=6.5, va='center', color=MUTED)
    save(fig, 'fig3_upgrades')


# ------------------------------------------------------------------ Figure A1
def figA1():
    rows = [('Pooled, every tagged pull request\n(how a leaderboard counts)', POOL['raw'], MAIN, 'o', False),
            ('Pooled, calendar-day effects removed', POOL['day'], MAIN, 'o', False),
            ('Within developer', POOL['dev'], COMP, 'o', False),
            ('Within developer and repository', POOL['repo'], COMP, 'o', False),
            ('Suspension design: within developer,\navailability set by the government', SUSP, ACCENT, 'D', True)]
    fig = plt.figure(figsize=(W, 2.75))
    ax = fig.add_axes([0.40, 0.2, 0.42, 0.72])
    ax.set_ylim(len(rows) - 0.15, -0.6); ax.set_yticks([]); ax.spines['left'].set_visible(False)
    f = lambda y: fig.transFigure.inverted().transform(ax.transData.transform((0, y)))[1]
    for y, (lab, (e, lo, hi), c, mk, bold) in enumerate(rows):
        ax.plot([lo, hi], [y, y], color=c, lw=1.4); ax.plot(e, y, mk, ms=6, color=c, zorder=3)
        fig.text(0.38, f(y), lab, ha='right', va='center', fontsize=7, linespacing=1.15)
        fig.text(0.985, f(y), est(e, lo, hi), ha='right', va='center', fontsize=7, fontweight='bold' if bold else 'normal',
                 color=INK if bold else MUTED)
    fig.text(0.985, 0.97, 'Estimate (95% CI)', ha='right', va='center', fontsize=6.5, color=MUTED)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(-6.2, 4.6); ax.set_xticks([-6, -4, -2, 0, 2, 4]); ax.set_xticklabels(['−6', '−4', '−2', '0', '+2', '+4'])
    ax.text(-0.25, len(rows) - 0.4, '← Opus 4.8 ahead', ha='right', va='center', fontsize=6.5, color=MUTED)
    ax.text(0.25, len(rows) - 0.4, 'Fable 5 ahead →', ha='left', va='center', fontsize=6.5, color=MUTED)
    ax.set_xlabel('Fable 5 minus Opus 4.8, seven-day merge share (points)')
    save(fig, 'figA1_pooled')


fig1(); fig2(); fig3(); figA1()
print('done')
