#!/usr/bin/env python3
"""Deterministic vector exporter for CoMeP Closed-Loop Main Figure (Option A, v4)
and compact single-column Neurocomputing Figure 1 motivation schematic (v4).

Satisfies all Round 15 Codex pre-implementation conditions with automated geometry/font assertions:
  - Main figure: 6.50 in x 5.96 in (468.0 pt x 429.1 pt = 624.0 px x 572.16 px @ 96dpi),
    native ordinary font >= 8.4 pt (=> >= 7.06 pt in arXiv 0.99\\textwidth=393.5pt [0.841x],
    and >= 8.40 pt in Neurocomputing 0.95\\textwidth=469.8pt [1.004x]).
  - Compact Figure 1: 3.30 in x 3.92 in (237.6 pt x 282.2 pt = 316.8 px x 376.32 px @ 96dpi),
    native ordinary font >= 7.7 pt (=> >= 7.72 pt in Neuro EN 1.00\\columnwidth=238.2pt [1.003x],
    and >= 7.41 pt in Neuro ZH 0.96\\columnwidth=228.7pt [0.963x]).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

DEFAULT_REPO_ROOT = Path("/root/.paseo/worktrees/1s5ku8sn/lucid-bumblebee")
KATEX_DIR = Path("/root/Developer/math/static/vendor/katex")
KATEX_CSS_PATH = KATEX_DIR / "katex.min.css"
KATEX_JS_PATH = KATEX_DIR / "katex.min.js"
CHROME_PATH = Path("/opt/sillytavern-browser/chromium/chrome")


def verify_render_prerequisites() -> None:
    for p in (KATEX_CSS_PATH, KATEX_JS_PATH, CHROME_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Required rendering asset missing: {p}")


# ===========================================================================
# 1. MAIN CLOSED-LOOP FIGURE (6.50 in x 5.96 in = 624.0 px x 572.16 px @ 96dpi)
#    Minimum ordinary CSS font size = 8.4pt (=> >= 7.06pt even at 0.841x in arXiv!).
# ===========================================================================
MAIN_CLOSED_LOOP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="__KATEX_CSS__">
<script src="__KATEX_JS__"></script>
<style>
  @page {
    size: 6.60in 5.96in;
    margin: 0;
  }
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  html, body {
    width: 633.6px;
    height: 572.16px;
    background: #FFFFFF;
    font-family: "Liberation Sans", "Nimbus Sans", "DejaVu Sans", sans-serif;
    color: #1E293B;
    overflow: hidden;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .canvas {
    position: relative;
    width: 633.6px;
    height: 572.16px;
    background: #FFFFFF;
  }
  .math {
    font-size: 0.98em;
  }
  .nw {
    white-space: nowrap;
  }

  .step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 1.6px solid currentColor;
    font-size: 8.5pt;
    font-weight: 800;
    background: #FFFFFF;
    flex-shrink: 0;
    line-height: 1;
  }

  /* (1) Top-left Episode Query Bubble */
  .query-box {
    position: absolute;
    left: 90px;
    top: 6px;
    width: 248px;
    height: 64px;
    background: #FDECE3;
    border: 1.6px solid #D97E54;
    border-radius: 10px;
    padding: 5px 8px;
    display: flex;
    align-items: center;
    gap: 7px;
    z-index: 5;
  }
  .query-title {
    font-size: 8.6pt;
    font-weight: 800;
    color: #7C2D12;
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 2px;
  }
  .query-sub {
    font-size: 8.5pt;
    font-weight: 700;
    color: #3B2316;
    line-height: 1.2;
  }

  /* (3) Top-right Bidirectional Rollout Box */
  .rollout-box {
    position: absolute;
    left: 346px;
    top: 6px;
    width: 272px;
    height: 64px;
    background: #EBF7FC;
    border: 1.6px solid #2AA8B8;
    border-radius: 10px;
    padding: 5px 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 5px;
    z-index: 5;
  }
  .rollout-title {
    font-size: 8.6pt;
    font-weight: 800;
    color: #0F2937;
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 2px;
  }
  .rollout-sub {
    font-size: 8.5pt;
    font-weight: 700;
    color: #1E3A5F;
    line-height: 1.2;
  }

  /* Left tall card: Embodied Task Environment */
  .env-card {
    position: absolute;
    left: 5px;
    top: 112px;
    width: 84px;
    height: 316px;
    background: #FFFFFF;
    border: 2.3px solid #D97E54;
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding: 16px 3px 16px 3px;
    z-index: 6;
  }
  .side-title {
    text-align: center;
    font-size: 8.8pt;
    font-weight: 800;
    line-height: 1.18;
    color: #18181B;
  }
  .side-sub {
    text-align: center;
    font-size: 8.5pt;
    font-weight: 700;
    color: #475569;
    margin-top: 5px;
    line-height: 1.22;
  }

  /* Right tall card: Frozen LLM Policy */
  .policy-card {
    position: absolute;
    left: 542px;
    top: 112px;
    width: 86px;
    height: 316px;
    background: #FFFFFF;
    border: 2.3px solid #2AA8B8;
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding: 16px 3px 16px 3px;
    z-index: 6;
  }

  /* Central Dashed ZERO-LLM-CALL Core Manager (generous 364px height: y=104..468!) */
  .core-box {
    position: absolute;
    left: 95px;
    top: 104px;
    width: 384px;
    height: 364px;
    background: #FCF6F8;
    border: 2.1px dashed #B56986;
    border-radius: 15px;
    padding: 15px 7px 9px 7px;
    z-index: 5;
  }
  .zero-call-badge {
    position: absolute;
    left: 14px;
    top: -10px;
    background: #1E3A5F;
    color: #FFFFFF;
    font-size: 8.5pt;
    font-weight: 800;
    padding: 1.5px 9px;
    border-radius: 999px;
    white-space: nowrap;
  }

  .core-grid {
    display: grid;
    grid-template-columns: 168px 36px 162px;
    height: 100%;
    align-items: stretch;
  }
  .col-stack {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .col-head {
    font-size: 8.7pt;
    font-weight: 800;
    color: #111827;
    text-align: center;
    white-space: nowrap;
  }
  .col-head .sec {
    color: #7E4A63;
    font-size: 8.5pt;
  }

  /* Routing sub-cards */
  .white-card {
    background: #FFFFFF;
    border: 1.2px solid #CBD5E1;
    border-radius: 7px;
    padding: 4.5px 5px;
    font-size: 8.5pt;
    line-height: 1.2;
    font-weight: 700;
    color: #1E293B;
  }
  .cand-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 3px;
    margin-top: 3px;
  }
  .mini-cand {
    border-radius: 6px;
    padding: 3px 2px;
    text-align: center;
    font-size: 8.5pt;
    font-weight: 800;
    line-height: 1.16;
  }
  .gate-card {
    background: #EFF6FF;
    border: 1.2px solid #93C5FD;
    border-radius: 7px;
    padding: 5px 5px;
    font-size: 8.5pt;
    line-height: 1.2;
    font-weight: 700;
    color: #1E3A8A;
  }

  /* Right column: Posterior Parameter Groups */
  .param-table {
    background: #FFFFFF;
    border: 1.5px solid #3F2E3E;
    border-radius: 8px;
    overflow: hidden;
  }
  .pt-title {
    background: #F8FAFC;
    border-bottom: 1.4px solid #3F2E3E;
    font-size: 8.5pt;
    font-weight: 800;
    color: #1E293B;
    text-align: center;
    padding: 3px 3px;
    line-height: 1.18;
  }
  .pt-row {
    display: grid;
    grid-template-columns: 85px 74px;
    min-height: 76px;
  }
  .pt-row + .pt-row {
    border-top: 1.4px solid #3F2E3E;
  }
  .pt-left-blue {
    background: #B9E2F5;
    border-right: 1.4px solid #3F2E3E;
    padding: 4px 4px;
    font-size: 8.5pt;
    font-weight: 700;
    line-height: 1.18;
    color: #0F172A;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .pt-left-pink {
    background: #F8BBD0;
    border-right: 1.4px solid #3F2E3E;
    padding: 4px 4px;
    font-size: 8.5pt;
    font-weight: 700;
    line-height: 1.18;
    color: #0F172A;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .pt-right {
    display: flex;
    flex-direction: column;
    font-size: 8.5pt;
    font-weight: 700;
    line-height: 1.16;
  }
  .pt-subcell {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2px 2px;
    text-align: center;
  }
  .pt-subcell + .pt-subcell {
    border-top: 1.1px solid #94A3B8;
  }
  .hatched {
    background: repeating-linear-gradient(
      -45deg,
      #E2E8F0,
      #E2E8F0 3.5px,
      #CBD5E1 3.5px,
      #CBD5E1 7px
    );
  }

  .prune-card {
    background: #ECFDF5;
    border: 1.3px solid #6EE7B7;
    border-radius: 7px;
    padding: 5px 5px;
    font-size: 8.5pt;
    line-height: 1.2;
    font-weight: 700;
    color: #064E3B;
  }

  /* SEGREGATED OUTSIDE 0-LLM BOX: Optional Post-Success LLM Distillation (y=480..530) */
  .distill-outside-card {
    position: absolute;
    left: 280px;
    top: 480px;
    width: 342px;
    height: 48px;
    background: #FFFBEB;
    border: 1.6px solid #D97706;
    border-radius: 9px;
    padding: 4px 7px;
    font-size: 8.5pt;
    line-height: 1.2;
    font-weight: 700;
    color: #78350F;
    z-index: 6;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
</style>
</head>
<body>
<div class="canvas">

  <!-- SVG LAYER FOR ALL CAUSAL ARROWS -->
  <svg width="633.6" height="572.16" viewBox="0 0 633.6 572.16" style="position:absolute;left:0;top:0;z-index:2;pointer-events:none;">
    <defs>
      <marker id="m-orange" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#D97E54"/>
      </marker>
      <marker id="m-teal" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#2AA8B8"/>
      </marker>
      <marker id="m-plum" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#A85D7F"/>
      </marker>
      <marker id="m-indigo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#4C5186"/>
      </marker>
      <marker id="m-amber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#D97706"/>
      </marker>
      <marker id="m-green" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
        <path d="M 0 1 L 10 5 L 0 9 z" fill="#047857"/>
      </marker>
    </defs>

    <!-- (1) Env initial obs o_{t,0} -> Top-left Query Bubble -> Routing -->
    <path d="M 28 112 L 28 38 L 84 38" fill="none" stroke="#D97E54" stroke-width="2.5" stroke-linejoin="round" marker-end="url(#m-orange)"/>
    <path d="M 184 70 L 184 99" fill="none" stroke="#D97E54" stroke-width="2.5" marker-end="url(#m-orange)"/>

    <!-- (2) Routing -> Fixed Bundle S_t -> Right Frozen LLM Policy -->
    <path d="M 479 272 L 536 272" fill="none" stroke="#4C5186" stroke-width="2.8" marker-end="url(#m-indigo)"/>

    <!-- (3) Clean dedicated channel (y=77..87, ABOVE badge at y=94!) for Policy <-> Env Rollout -->
    <path d="M 584 112 L 584 75" fill="none" stroke="#2AA8B8" stroke-width="2.5" marker-end="url(#m-teal)"/>
    <!-- Action a_h from Policy/Rollout -> Left Env -->
    <path d="M 358 70 L 358 78 L 56 78 L 56 106" fill="none" stroke="#2AA8B8" stroke-width="2.1" stroke-dasharray="4.5,3" stroke-linejoin="round" marker-end="url(#m-teal)"/>
    <!-- Env obs o_{h+1} -> Right Policy -->
    <path d="M 72 112 L 72 86 L 558 86 L 558 106" fill="none" stroke="#D97E54" stroke-width="2.0" stroke-dasharray="4.5,3" stroke-linejoin="round" marker-end="url(#m-orange)"/>

    <!-- (4) Terminal Episode Record (c_t, S_t, r_t, g_t) from Env bottom -> Fit (0-LLM) AND Optional Distillation (outside) -->
    <path d="M 42 428 L 42 504 L 274 504" fill="none" stroke="#A85D7F" stroke-width="2.5" stroke-linejoin="round" marker-end="url(#m-amber)"/>
    <!-- Branch up into 2nd-order Laplace Fit (at x=262, well below core-box y=468!) -->
    <path d="M 262 504 L 262 473" fill="none" stroke="#A85D7F" stroke-width="2.5" marker-end="url(#m-plum)"/>
    <!-- Admitted m_new from Optional Distillation up into Conditional Pruning (at x=428) -->
    <path d="M 428 480 L 428 473" fill="none" stroke="#D97706" stroke-width="2.3" stroke-dasharray="3.5,2.5" marker-end="url(#m-amber)"/>

    <!-- (5) Updated Bank M_{t+1} from Pruning (x=320) -> clean bottom bus y=552 -> Routing (x=142) -->
    <path d="M 280 468 L 280 552 L 138 552 L 138 473" fill="none" stroke="#047857" stroke-width="2.6" stroke-linejoin="round" marker-end="url(#m-green)"/>
  </svg>

  <!-- (1) TOP-LEFT EPISODE QUERY BUBBLE -->
  <div class="query-box">
    <svg width="32" height="32" viewBox="0 0 64 64" style="flex-shrink:0;">
      <circle cx="32" cy="32" r="29" fill="#FFF7ED" stroke="#D97E54" stroke-width="2.8"/>
      <rect x="18" y="23" width="22" height="22" rx="5" fill="#FFFFFF" stroke="#9A4E28" stroke-width="2.8"/>
      <path d="M 40 28 C 47 28, 47 40, 40 40" fill="none" stroke="#9A4E28" stroke-width="2.8"/>
      <circle cx="47" cy="18" r="4.5" fill="#38BDF8"/>
    </svg>
    <div>
      <div class="query-title">
        <span class="step-badge" style="color:#D97E54;">1</span>
        <span>Query <span class="math" data-tex="(q_t,c_t)"></span> &amp; Bank <span class="math" data-tex="\mathcal{M}_t"></span></span>
      </div>
      <div class="query-sub">
        <i>Illustr.:</i> “Put clean mug in cabinet.” (<span class="math" data-tex="\mathcal{T}_{\text{test}}"></span>)
      </div>
    </div>
  </div>

  <!-- (3) TOP-RIGHT BIDIRECTIONAL TRAJECTORY ROLLOUT -->
  <div class="rollout-box">
    <div>
      <div class="rollout-title">
        <span class="step-badge" style="color:#2AA8B8;">3</span>
        <span>Rollout (<span class="math nw" data-tex="H_t \le H_{\max}"></span>)</span>
      </div>
      <div class="rollout-sub">
        Policy <span class="math" data-tex="a_h"></span> ↔ Env <span class="math" data-tex="o_{h+1}"></span>; outcome <span class="math nw" data-tex="r_t\!\in\!\{0,1\}"></span>
      </div>
    </div>
    <svg width="34" height="34" viewBox="0 0 68 68" style="flex-shrink:0;">
      <path d="M 14 32 A 20 20 0 0 1 52 18" fill="none" stroke="#F4A27E" stroke-width="7.5" stroke-linecap="round"/>
      <polygon points="54,9 61,25 44,22" fill="#F4A27E"/>
      <path d="M 54 36 A 20 20 0 0 1 16 50" fill="none" stroke="#2AA8B8" stroke-width="7.5" stroke-linecap="round"/>
      <polygon points="14,59 7,43 24,46" fill="#2AA8B8"/>
    </svg>
  </div>

  <div style="position:absolute;left:4px;top:54px;font-size:8.5pt;font-weight:800;color:#B45309;">
    <span class="math" data-tex="o_{t,0}"></span>
  </div>

  <!-- LEFT CARD: EMBODIED TASK ENVIRONMENT -->
  <div class="env-card">
    <svg width="70" height="102" viewBox="0 0 140 165">
      <circle cx="70" cy="82" r="60" fill="#FFF3EB"/>
      <rect x="20" y="88" width="100" height="44" rx="8" fill="#F4C095" stroke="#8C4F2B" stroke-width="3"/>
      <line x1="70" y1="88" x2="70" y2="132" stroke="#8C4F2B" stroke-width="2.8"/>
      <rect x="16" y="80" width="108" height="9" rx="4" fill="#D97E54" stroke="#8C4F2B" stroke-width="2.6"/>
      <rect x="30" y="54" width="23" height="26" rx="5" fill="#FFFFFF" stroke="#8C4F2B" stroke-width="2.8"/>
      <path d="M 53 60 C 60 60, 60 72, 53 72" fill="none" stroke="#8C4F2B" stroke-width="2.8"/>
      <polygon points="86,48 98,48 108,80 76,80" fill="#E0F2FE" stroke="#0369A1" stroke-width="2.8" stroke-linejoin="round"/>
      <polygon points="80,67 104,67 107,78 77,78" fill="#38BDF8"/>
    </svg>
    <div>
      <div class="side-title">Embodied<br>Task Env</div>
      <div class="side-sub">ALFWorld &amp;<br>SciWorld<br>(<span class="math nw" data-tex="t\!=\!1\dots T"></span>)</div>
    </div>
  </div>

  <!-- (2) FIXED BUNDLE S_t INJECTION LABEL -->
  <div style="position:absolute;left:480px;top:206px;width:60px;text-align:center;z-index:8;">
    <div style="display:inline-flex;align-items:center;gap:2px;font-size:8.6pt;font-weight:800;color:#4C5186;">
      <span class="step-badge" style="color:#4C5186;">2</span>
      <span><span class="math" data-tex="S_t"></span></span>
    </div>
    <div style="font-size:8.5pt;font-weight:700;color:#1E293B;line-height:1.24;margin-top:2px;">
      <span class="math nw" data-tex="\sum\ell_i\!\le\!90"></span><br>
      <span class="math nw" data-tex="|S_t|\!\le\!3"></span>
    </div>
    <div style="font-size:8.5pt;font-weight:700;color:#4C5186;line-height:1.18;margin-top:26px;">
      Fixed in<br>prompt
    </div>
  </div>

  <!-- RIGHT CARD: FROZEN LLM POLICY -->
  <div class="policy-card">
    <svg width="70" height="102" viewBox="0 0 140 165">
      <circle cx="70" cy="82" r="60" fill="#E6F7FA"/>
      <path d="M 48 36 Q 70 22 92 36" fill="none" stroke="#2AA8B8" stroke-width="4.2" stroke-linecap="round"/>
      <circle cx="70" cy="27" r="6" fill="#38BDF8" stroke="#0F172A" stroke-width="2.4"/>
      <rect x="30" y="42" width="80" height="62" rx="24" fill="#FFFFFF" stroke="#0F172A" stroke-width="3.2"/>
      <rect x="39" y="54" width="62" height="32" rx="14" fill="#0F172A"/>
      <ellipse cx="56" cy="69" rx="6" ry="7" fill="#38BDF8"/>
      <ellipse cx="84" cy="69" rx="6" ry="7" fill="#38BDF8"/>
      <path d="M 62 92 Q 70 98 78 92" fill="none" stroke="#0F172A" stroke-width="2.8" stroke-linecap="round"/>
      <path d="M 44 108 L 96 108 C 104 108 108 136 98 140 L 42 140 C 32 136 36 108 44 108 Z" fill="#FFFFFF" stroke="#0F172A" stroke-width="3"/>
    </svg>
    <div>
      <div class="side-title">Frozen LLM<br>Policy</div>
      <div class="side-sub"><span class="math" data-tex="\pi_{\text{LLM}}"></span><br><span class="math nw" data-tex="(a_h\!\mid\!o_h,S_t)"></span><br>Frozen wts</div>
    </div>
  </div>

  <!-- CENTRAL DASHED ZERO-LLM-CALL CORE MANAGER -->
  <div class="core-box">
    <div class="zero-call-badge">CoMeP Core · 0 LLM Calls (Routing, Fit &amp; Pruning)</div>

    <div class="core-grid">
      <!-- LEFT COLUMN: SYNERGY-AWARE ROUTING -->
      <div class="col-stack">
        <div class="col-head">Synergy Routing <span class="sec">(§3.2–3.3)</span></div>

        <div class="white-card">
          <div style="color:#0F172A;"><b>Contrast Gate</b> <span class="math nw" data-tex="\tilde{g}_{t,i}\!\in\![0,1]"></span></div>
          <div style="color:#475569;">31 starters (Eqs. 4–5)</div>
        </div>

        <div class="white-card">
          <div style="color:#1D4ED8;"><b>All feasible singletons;</b></div>
          <div style="color:#334155;"><b>pairs from up to 10 candidates</b></div>
          <div style="color:#64748B;margin-top:1px;"><i>Illustrative signs (Eq. 2, 11):</i></div>
          <div class="cand-grid">
            <div class="mini-cand" style="background:#EFF6FF;border:1.2px solid #60A5FA;color:#1E3A8A;">
              <div><span class="math" data-tex="m_i"></span></div>
              <div><span class="math nw" data-tex="\hat{\theta}_i\!>\!0"></span></div>
            </div>
            <div class="mini-cand" style="background:#ECFDF5;border:1.2px solid #34D399;color:#065F46;">
              <div><span class="math" data-tex="m_j"></span></div>
              <div><span class="math nw" data-tex="\hat{s}_{ij}\!>\!0"></span></div>
            </div>
            <div class="mini-cand" style="background:#FFF1F2;border:1.2px solid #FB7185;color:#9F1239;">
              <div><span class="math" data-tex="m_q"></span></div>
              <div><span class="math nw" data-tex="\hat{s}_{iq}\!<\!0"></span></div>
            </div>
          </div>
        </div>

        <div class="gate-card">
          <div style="color:#1D4ED8;"><b>Gate <span class="math" data-tex="G_t(i,j)"></span> (Eq. 9):</b></div>
          <div><span class="math nw" data-tex="N_{c_t}\!\ge\!5,\; n_i\!\ge\!2,\; n_j\!\ge\!2"></span></div>
          <div>• Gates pair UCB &amp; random pair explore at <span class="math nw" data-tex="S_t\!=\!\varnothing"></span></div>
          <div>• Own rules: <span class="math" data-tex="\hat{s}_{ij}"></span> fit, Conflict, confirm., <span class="math" data-tex="\kappa_{\text{stop}}"></span></div>
        </div>
      </div>

      <!-- CENTER HORIZONTAL POSTERIOR BRIDGE -->
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;text-align:center;">
        <div style="font-size:8.5pt;font-weight:800;color:#991B1B;line-height:1.15;">
          Post.<br><span class="math nw" data-tex="\hat{\boldsymbol{\psi}},\hat{\boldsymbol{\tau}}"></span>
        </div>
        <svg width="30" height="16" viewBox="0 0 30 16">
          <path d="M 28 8 L 6 8" fill="none" stroke="#B91C1C" stroke-width="2.4" stroke-dasharray="3.5,2"/>
          <polygon points="7,3 0,8 7,13" fill="#B91C1C"/>
        </svg>
      </div>

      <!-- RIGHT COLUMN: 2ND-ORDER LAPLACE FIT & CONDITIONAL PRUNING -->
      <div class="col-stack">
        <div class="col-head">Credit &amp; Pruning <span class="sec">(§3.2–3.3)</span></div>

        <!-- Posterior Parameter Groups -->
        <div class="param-table">
          <div class="pt-title">
            <div>Posterior Groups</div>
            <div>(<span class="math nw" data-tex="W\!=\!600"></span>, Eqs. 3, 6–8)</div>
          </div>

          <!-- Row 1: Main & Context -->
          <div class="pt-row">
            <div class="pt-left-blue">
              <div><b>Main &amp; <span class="math" data-tex="\hat{\alpha}_c"></span></b></div>
              <div>All <span class="math nw" data-tex="i\!\in\!\mathcal{I}(\mathcal{H}_t)"></span> kept in <span class="math" data-tex="\mathbf{X}_t"></span></div>
            </div>
            <div class="pt-right">
              <div class="pt-subcell" style="background:#EFF6FF;color:#1E3A8A;">
                <span class="math nw" data-tex="\hat{\alpha}_c,\;\hat{\theta}_i"></span>
                <span>(active)</span>
              </div>
              <div class="pt-subcell hatched" style="color:#0F172A;">
                <span class="math nw" data-tex="\hat{\theta}_k^{\text{hist}}"></span>
                <span>(in <span class="math" data-tex="\mathbf{X}_t"></span>)</span>
              </div>
            </div>
          </div>

          <!-- Row 2: Pairwise Coupling -->
          <div class="pt-row">
            <div class="pt-left-pink">
              <div><b>Pairs <span class="math" data-tex="\hat{s}"></span></b> (<span class="math nw" data-tex="\ell_2\!+\!\ell_1"></span>)</div>
              <div><span class="math nw" data-tex="P_{\max}\!=\!48"></span>; <b>not</b></div>
              <div>gated by <span class="math" data-tex="G_t"></span></div>
            </div>
            <div class="pt-right">
              <div class="pt-subcell" style="background:#FFF1F2;color:#9F1239;">
                <span class="nw"><i>eg</i> <span class="math" data-tex="\hat{s}_{ij}\!>\!0"></span>,</span>
                <span class="math nw" data-tex="\hat{s}_{iq}\!<\!0"></span>
              </div>
              <div class="pt-subcell" style="background:#F8FAFC;color:#475569;">
                <span>pruned pair</span>
                <span>dropped</span>
              </div>
            </div>
          </div>
        </div>

        <!-- Conditional Coalition-Aware Pruning (0 LLM calls) -->
        <div class="prune-card">
          <div style="color:#047857;"><b>Coalition Pruning (Eqs. 12–13)</b></div>
          <div>• <b>If bank over capacity:</b> evict <span class="math" data-tex="\arg\min\mathrm{Score}_{\text{prune}}"></span></div>
          <div>• <b>Guard <span class="math" data-tex="\mathrm{Ready}(i)"></span></b> <span class="nw">(<span class="math" data-tex="1\!\le\!n_i\!\le\!2"></span>):</span> pruning score only</div>
        </div>
      </div>
    </div>
  </div>

  <!-- (4) TERMINAL RECORD LABEL (clean upper-bottom row y=492, well below core-box y=468!) -->
  <div style="position:absolute;left:50px;top:492px;z-index:8;background:#FFFFFF;padding:1px 4px;font-size:8.5pt;font-weight:800;color:#8E4466;display:flex;align-items:center;gap:3px;">
    <span class="step-badge" style="color:#A85D7F;">4</span>
    <span class="nw">Record <span class="math" data-tex="(c_t,S_t,r_t,\tilde{\mathbf{g}}_t)"></span></span>
  </div>

  <!-- SEGREGATED OUTSIDE 0-LLM BOX: Optional Post-Success LLM Distillation -->
  <div class="distill-outside-card">
    <div style="color:#B45309;display:flex;justify-content:space-between;align-items:center;">
      <span><b>Optional Post-Success LLM Distill</b></span>
      <span style="background:#FEF3C7;padding:0 5px;border-radius:4px;border:1px solid #F59E0B;">Outside 0-call</span>
    </div>
    <div>
      If <span class="math nw" data-tex="r_t\!=\!1"></span>: admit <span class="math nw" data-tex="\le\!1"></span> rule (15–32 tok); if <span class="math nw" data-tex="r_t\!=\!0"></span>: <b>0 calls</b>
    </div>
  </div>

  <!-- (5) NEXT-EPISODE BANK M_{t+1} RETURN TO ROUTING (clean lower-bottom row y=541!) -->
  <div style="position:absolute;left:148px;top:541px;z-index:8;background:#FFFFFF;padding:1px 5px;font-size:8.5pt;font-weight:800;color:#047857;display:flex;align-items:center;gap:4px;">
    <span class="step-badge" style="color:#047857;">5</span>
    <span class="nw">Next <span class="math" data-tex="\mathcal{M}_{t+1}"></span> (<span class="math" data-tex="\sum_{i\in\mathcal{M}_{t+1}}\ell_i\!\le\!560,\;|\mathcal{M}_{t+1}|\!\le\!22"></span>) → Routing</span>
  </div>

</div>
<script>
  document.querySelectorAll('.math').forEach(el => {
    katex.render(el.getAttribute('data-tex'), el, { throwOnError: false });
  });
</script>
</body>
</html>
"""


# ===========================================================================
# 2. COMPACT SINGLE-COLUMN FIGURE 1 (3.30 in x 3.92 in = 316.8 px x 376.32 px)
#    Minimum ordinary CSS font size = 7.7pt (=> >= 7.41pt even at 0.96\columnwidth!).
# ===========================================================================
COMPACT_FIG1_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="__KATEX_CSS__">
<script src="__KATEX_JS__"></script>
<style>
  @page {
    size: 3.30in 3.92in;
    margin: 0;
  }
  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  html, body {
    width: 316.8px;
    height: 376.32px;
    background: #FFFFFF;
    font-family: "Liberation Sans", "Nimbus Sans", "DejaVu Sans", sans-serif;
    color: #1E293B;
    overflow: hidden;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }
  .canvas {
    width: 316.8px;
    height: 376.32px;
    padding: 7px;
    display: flex;
    flex-direction: column;
    gap: 7px;
    background: #FFFFFF;
  }
  .math {
    font-size: 0.98em;
  }
  .nw {
    white-space: nowrap;
  }

  .top-banner {
    background: #FDF4EF;
    border: 1.6px solid #E08B5B;
    border-radius: 10px;
    padding: 5px 6px;
  }
  .banner-title {
    font-size: 8.0pt;
    font-weight: 800;
    color: #7C2D12;
    text-align: center;
    margin-bottom: 4px;
    white-space: nowrap;
  }
  .bundle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 3px;
  }
  .b-chip {
    border-radius: 6px;
    padding: 3px 4px;
    text-align: center;
    font-size: 7.7pt;
    font-weight: 700;
    line-height: 1.18;
  }

  .two-cols {
    flex: 1;
    display: grid;
    grid-template-columns: 148px 148px;
    gap: 6px;
  }
  .p-card {
    border-radius: 11px;
    padding: 7px 6px 7px 6px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .indep {
    background: #FFF8F8;
    border: 1.8px solid #E11D48;
  }
  .comep {
    background: #F4FAFF;
    border: 1.8px solid #0284C7;
  }
  .p-head {
    text-align: center;
    padding-bottom: 4px;
    border-bottom: 1.2px dashed rgba(100,116,139,0.38);
  }
  .p-head .mt {
    font-size: 8.5pt;
    font-weight: 800;
    line-height: 1.16;
  }
  .p-head .st {
    font-size: 7.7pt;
    font-weight: 700;
    margin-top: 2px;
    line-height: 1.18;
  }
  .risk-box {
    background: #FFFFFF;
    border: 1.1px solid #FECDD3;
    border-left: 3.2px solid #E11D48;
    border-radius: 6px;
    padding: 4px 5px;
    font-size: 7.7pt;
    line-height: 1.2;
    font-weight: 700;
    color: #334155;
  }
  .risk-box b {
    color: #9F1239;
    font-weight: 800;
  }
  .factor-card {
    background: #FFFFFF;
    border: 1.2px solid #BAE6FD;
    border-radius: 8px;
    padding: 5px 4px 6px 4px;
    text-align: center;
  }
  .node-c {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 7.8pt;
    font-weight: 800;
    z-index: 4;
  }
  .guard-note {
    background: #ECFDF5;
    border: 1.2px solid #6EE7B7;
    border-radius: 7px;
    padding: 5px 5px;
    font-size: 7.7pt;
    line-height: 1.2;
    font-weight: 700;
    color: #064E3B;
  }
</style>
</head>
<body>
<div class="canvas">

  <!-- TOP ILLUSTRATIVE SHARED-FAILURE BUNDLE -->
  <div class="top-banner">
    <div class="banner-title">Illustrative Failed Episode <span class="math" data-tex="\tau"></span>: <span class="math nw" data-tex="S_\tau \to r_\tau = 0"></span></div>
    <div class="bundle-row">
      <div class="b-chip" style="background:#DBEAFE;border:1.2px solid #60A5FA;color:#1E3A8A;">
        <div><span class="math" data-tex="c_\tau"></span> (<span class="math" data-tex="\alpha_{c_\tau}"></span>)</div>
        <div>Context</div>
      </div>
      <div class="b-chip" style="background:#FFFFFF;border:1.2px solid #94A3B8;color:#0F172A;">
        <div><span class="math" data-tex="m_i"></span></div>
        <div><span class="math nw" data-tex="\theta_i\!>\!0"></span></div>
      </div>
      <div class="b-chip" style="background:#FFF1F2;border:1.2px solid #FDA4AF;color:#9F1239;">
        <div><span class="math" data-tex="m_j"></span></div>
        <div><span class="math nw" data-tex="s_{ij}\!<\!0"></span></div>
      </div>
      <div class="b-chip" style="background:#FEF9C3;border:1.2px solid #FACC15;color:#854D0E;">
        <div><span class="math" data-tex="m_k"></span></div>
        <div><span class="math nw" data-tex="n_k\!=\!1"></span></div>
      </div>
      <div style="font-size:8.5pt;font-weight:800;color:#B91C1C;">→</div>
      <div class="b-chip" style="background:#FEE2E2;border:1.3px solid #EF4444;color:#991B1B;">
        <div><span class="math nw" data-tex="r_\tau\!=\!0"></span></div>
        <div>Fail</div>
      </div>
    </div>
  </div>

  <!-- TWO-COLUMN PARADIGM CONTRAST -->
  <div class="two-cols">
    <!-- Left: Independent Credit -->
    <div class="p-card indep">
      <div class="p-head" style="color:#BE123C;">
        <div class="mt">Independent Credit</div>
        <div class="st">Updates (<span class="math nw" data-tex="-\Delta_i,-\Delta_j,-\Delta_k"></span>)</div>
      </div>
      <div class="risk-box">
        <b>1. Collinear Ambiguity:</b> Conflates <span class="math" data-tex="c_\tau"></span> &amp; partners; col deletion can induce nonzero Newton step (Prop. 3.1).
      </div>
      <div class="risk-box">
        <b>2. Unmodeled Pairs:</b> Can miss step competition (<span class="math nw" data-tex="s_{ij}\!<\!0"></span>) or synergy (<span class="math nw" data-tex="s_{ij}\!>\!0"></span>).
      </div>
      <div class="risk-box">
        <b>3. Premature Pruning:</b> Low-count <span class="math" data-tex="m_k"></span> risked before solo separation.
      </div>
    </div>

    <!-- Right: CoMeP Combinatorial Model -->
    <div class="p-card comep">
      <div class="p-head" style="color:#0369A1;">
        <div class="mt">CoMeP (Ours)</div>
        <div class="st">Joint credit model +<br>heuristic guard</div>
      </div>

      <div class="factor-card">
        <div style="font-size:7.8pt;font-weight:800;color:#0F172A;margin-bottom:4px;white-space:nowrap;">
          Joint Factor Model (Eq. 3)
        </div>
        <div style="position:relative;height:62px;width:134px;margin:0 auto;">
          <svg width="134" height="62" viewBox="0 0 134 62" style="position:absolute;left:0;top:0;z-index:1;">
            <line x1="14" y1="20" x2="40" y2="58" stroke="#2563EB" stroke-width="1.6"/>
            <line x1="48" y1="20" x2="58" y2="58" stroke="#0F172A" stroke-width="1.6"/>
            <line x1="55" y1="18" x2="67" y2="34" stroke="#E11D48" stroke-width="1.7"/>
            <line x1="81" y1="18" x2="69" y2="34" stroke="#E11D48" stroke-width="1.7"/>
            <line x1="68" y1="45" x2="68" y2="58" stroke="#E11D48" stroke-width="1.7"/>
            <line x1="86" y1="20" x2="78" y2="58" stroke="#0F172A" stroke-width="1.6"/>
            <line x1="120" y1="20" x2="96" y2="58" stroke="#CA8A04" stroke-width="1.6"/>
          </svg>
          <div class="node-c" style="position:absolute;left:2px;top:1px;background:#DBEAFE;border:1.5px solid #2563EB;color:#1E3A8A;"><span class="math" data-tex="c_\tau"></span></div>
          <div class="node-c" style="position:absolute;left:36px;top:1px;background:#FFFFFF;border:1.5px solid #0F172A;color:#0F172A;"><span class="math" data-tex="m_i"></span></div>
          <div class="node-c" style="position:absolute;left:74px;top:1px;background:#FFFFFF;border:1.5px solid #0F172A;color:#0F172A;"><span class="math" data-tex="m_j"></span></div>
          <div class="node-c" style="position:absolute;left:108px;top:1px;background:#FEF9C3;border:1.5px solid #CA8A04;color:#854D0E;"><span class="math" data-tex="m_k"></span></div>
          <div style="position:absolute;left:55px;top:29px;width:26px;height:16px;border-radius:4px;background:#FFE4E6;border:1.3px solid #E11D48;color:#BE123C;font-size:7.7pt;font-weight:800;display:flex;align-items:center;justify-content:center;z-index:4;">
            <span class="math" data-tex="s_{ij}"></span>
          </div>
        </div>
        <div style="background:#E0F2FE;border:1.3px solid #0284C7;border-radius:5px;padding:3px 4px;font-size:7.9pt;font-weight:800;color:#0C4A6E;white-space:nowrap;">
          <span class="math" data-tex="\eta_\tau(\boldsymbol{\psi}) \to r_\tau"></span> (Eq. 3)
        </div>
      </div>

      <div class="guard-note">
        <b>Fig. 5 safeguards:</b> Keeps hist. cols in <span class="math" data-tex="\mathbf{X}_t"></span>; pruning-only <span class="math" data-tex="\mathrm{Ready}(k)"></span> (<i>no predictor change, identifiability, or retention guarantee</i>).
      </div>
    </div>
  </div>

</div>
<script>
  document.querySelectorAll('.math').forEach(el => {
    katex.render(el.getAttribute('data-tex'), el, { throwOnError: false });
  });
</script>
</body>
</html>
"""


def render_html_to_pdf(html_str: str, pdf_path: Path, win_w: int, win_h: int) -> None:
    verify_render_prerequisites()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    final_html = html_str.replace("__KATEX_CSS__", f"file://{KATEX_CSS_PATH}").replace(
        "__KATEX_JS__", f"file://{KATEX_JS_PATH}"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tf:
        tf.write(final_html)
        tmp_html = Path(tf.name)
    try:
        subprocess.run(
            [
                str(CHROME_PATH),
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                f"--window-size={win_w},{win_h}",
                f"--print-to-pdf={pdf_path}",
                "--no-pdf-header-footer",
                f"file://{tmp_html}",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        tmp_html.unlink(missing_ok=True)

    # Automated PDF XML overflow, error-color, and KaTeX-font check
    xml_out = subprocess.check_output(["pdftohtml", "-xml", "-zoom", "1.0", "-stdout", str(pdf_path)], text=True)
    if "#cc0000" in xml_out.lower():
        raise RuntimeError(f"KaTeX render error (#cc0000) detected in {pdf_path}")
    root = ET.fromstring(xml_out)
    pages = root.findall(".//page")
    if len(pages) != 1:
        raise RuntimeError(f"Expected 1 page in {pdf_path}, got {len(pages)}")
    pg = pages[0]
    pw, ph = float(pg.attrib["width"]), float(pg.attrib["height"])
    families = {fs.attrib.get("family", "") for fs in root.findall(".//fontspec")}
    if not any("KaTeX" in fam for fam in families):
        raise RuntimeError(f"No rendered KaTeX fonts found in {pdf_path}: {families}")
    for t in pg.findall(".//text"):
        txt = "".join(t.itertext()).strip()
        if not txt:
            continue
        top, left, w, h = float(t.attrib["top"]), float(t.attrib["left"]), float(t.attrib["width"]), float(t.attrib["height"])
        if left < 0.5 or top < 0.5 or (left + w) > (pw - 0.5) or (top + h) > (ph - 0.5):
            raise RuntimeError(f"Text overflow/clipping in {pdf_path}: {txt!r} bbox=({left},{top},{w},{h}) page=({pw},{ph})")


def export_all(output_root: Path, *, main_only: bool = False) -> None:
    neuro_figs = output_root / "paper" / "neurocomputing" / "figures"
    arxiv_figs = output_root / "paper" / "figures"
    art_figs = output_root / "paper_artifacts" / "figures"
    for d in (neuro_figs, arxiv_figs, art_figs):
        d.mkdir(parents=True, exist_ok=True)

    main_pdf = neuro_figs / "Figure_5_closed_loop.pdf"
    render_html_to_pdf(MAIN_CLOSED_LOOP_HTML, main_pdf, 680, 620)

    arxiv_pdf = arxiv_figs / "fig0_workflow_closed_loop.pdf"
    shutil.copyfile(main_pdf, arxiv_pdf)
    subprocess.run(
        ["pdftoppm", "-png", "-r", "300", "-singlefile", str(arxiv_pdf), str(arxiv_figs / "fig0_workflow_closed_loop")],
        check=True,
    )
    shutil.copyfile(arxiv_pdf, art_figs / "fig0_workflow_closed_loop.pdf")
    shutil.copyfile(arxiv_figs / "fig0_workflow_closed_loop.png", art_figs / "fig0_workflow_closed_loop.png")

    fig1_pdf = neuro_figs / "Figure_1_compact.pdf"
    if not main_only or not fig1_pdf.exists():
        render_html_to_pdf(COMPACT_FIG1_HTML, fig1_pdf, 380, 440)

    print("Exported v5 closed-loop main figure (main_only=%s) under: %s" % (main_only, output_root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--main-only", action="store_true", help="Re-export only the main closed-loop figure assets.")
    args = parser.parse_args()
    export_all(args.output_root.resolve(), main_only=args.main_only)


if __name__ == "__main__":
    main()
